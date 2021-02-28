from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers import MaxPooling2D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np

from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

import pandas as pd

# Uncomment below if you want to not have warnings!

import os, warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""# Preparing Data"""

TRIP_SIZE = 32
df = pd.read_csv('cars_over100.csv')

df = df.fillna(0)

df[['velocity', 'acceleration']].isnull().values.any()

mean_lat = df['lat'].mean()
std_lat = df['lat'].std()
mean_long = df['long'].mean()
std_long = df['long'].std()

mean_vel = df['velocity'].mean()
std_vel = df['velocity'].std()

mean_acc = df['acceleration'].mean()
std_acc = df['acceleration'].std()


print(mean_long, std_long,mean_lat, std_lat, mean_vel, std_vel, std_acc, mean_acc)

df['lat'] = (df['lat']-mean_lat)/std_lat
df['long'] = (df['long']-mean_long)/std_long
df['velocity'] = (df['velocity']-mean_vel)/std_vel
df['acceleration'] = (df['acceleration']-mean_acc)/std_acc


def get_traj(df):
    by_traj = []
    for id in df['trajectory_id'].unique():
        by_traj.append(df.loc[df['trajectory_id'] == id][['lat', 'long']].reset_index().values.tolist())
    return by_traj


def get_routes(trajectories):
    routes = []
    for traj in trajectories:
        curr_route = []
        for elem in traj:
            if (len(curr_route) % TRIP_SIZE == 0 and len(curr_route) > 0):
                routes.append(curr_route)
                curr_route = []
            curr_route.append(elem[1:])
    return routes


def get_diff_routes(trajectories):
    routes = []
    for traj in trajectories:
        curr_route = []
        prev_elem = []
        for elem in traj:
            if len(curr_route) % TRIP_SIZE == 0:
                if len(curr_route) > 0:
                    routes.append(curr_route)
                curr_route = [[0.0, 0.0]]
            else:
                curr_route.append(np.asarray(elem[1:]) - np.asarray(prev_elem))
            prev_elem = elem[1:]
    return routes


def get_both_routes(trajectories):
    routes = []
    for traj in trajectories:
        curr_route = []
        prev_elem = []
        for elem in traj:
            if len(curr_route) % TRIP_SIZE == 0:
                if len(curr_route) > 0:
                    routes.append(curr_route)
                curr_route = [[elem[1], elem[2], 0.0, 0.0]]
            else:
                curr_route.append(np.hstack([np.asarray(elem[1:]), np.asarray(elem[1:]) - np.asarray(prev_elem)]))
            prev_elem = elem[1:]
    return routes


def stand_driver(driver_routes, mean_x=None, mean_y=None, std_x=None, std_y=None):
    if mean_x is None:
        x_all = driver_routes[:, :, 0].flatten()
        y_all = driver_routes[:, :, 1].flatten()

        mean_x = np.mean(x_all)
        std_x = np.std(x_all)
        mean_y = np.mean(y_all)
        std_y = np.std(y_all)

    result = driver_routes.copy()
    result[:, :, 0] = (driver_routes[:, :, 0] - mean_x) / std_x
    result[:, :, 1] = (driver_routes[:, :, 1] - mean_y) / std_y
    return result


def stand_driver2(driver_routes, mean_x=None, mean_y=None, std_x=None, std_y=None,
                  mean_diff_x=None, mean_diff_y=None, std_diff_x=None, std_diff_y=None):
    if mean_x is None:
        x_all = driver_routes[:, :, 0].flatten()
        y_all = driver_routes[:, :, 1].flatten()
        x_diff_all = driver_routes[:, :, 2].flatten()
        y_diff_all = driver_routes[:, :, 3].flatten()

        mean_x = np.mean(x_all)
        std_x = np.std(x_all)
        mean_y = np.mean(y_all)
        std_y = np.std(y_all)
        mean_diff_x = np.std(x_diff_all)
        std_diff_x = np.std(x_diff_all)
        mean_diff_y = np.std(y_diff_all)
        std_diff_y = np.std(y_diff_all)

    result = driver_routes.copy()
    result[:, :, 0] = (driver_routes[:, :, 0] - mean_x) / std_x
    result[:, :, 1] = (driver_routes[:, :, 1] - mean_y) / std_y
    result[:, :, 2] = (driver_routes[:, :, 2] - mean_diff_x) / std_diff_x
    result[:, :, 3] = (driver_routes[:, :, 3] - mean_diff_y) / std_diff_y
    return result


def unstandardize(mean, std, value):
    return (value * std) + mean


mahal_roc = []
mahal_pr = []
gmm_roc = []
gmm_pr = []
egbd_roc = []
egbd_pr = []

roc_auc_scores = []
prauc_scores = []

NUM_VALS = 4

ids_to_remove = [20, 65, 78, 84, 85, 106, 112, 125, 163]
for id_remove in ids_to_remove:
    if id_remove == 163:

        removed_df = df.loc[df['subfolder'] == id_remove]

        remaining_df = df.loc[df['subfolder'] != id_remove]

        remaining_traj = get_traj(remaining_df)
        removed_traj = get_traj(removed_df)

        remaining_routes = np.asarray(get_both_routes(remaining_traj))

        mean_lat = np.mean(remaining_routes[:, :, 0])
        mean_long = np.mean(remaining_routes[:, :, 1])
        mean_lat_diff = np.mean(remaining_routes[:, :, 2])
        mean_long_diff = np.mean(remaining_routes[:, :, 3])

        std_lat = np.std(remaining_routes[:, :, 0])
        std_long = np.std(remaining_routes[:, :, 1])
        std_lat_diff = np.std(remaining_routes[:, :, 2])
        std_long_diff = np.std(remaining_routes[:, :, 3])

        remaining_routes = stand_driver2(remaining_routes)

        removed_routes = stand_driver2(np.asarray(get_both_routes(removed_traj)),
                                       mean_lat, mean_long, std_lat, std_long,
                                       mean_lat_diff, mean_long_diff, std_lat_diff, std_long_diff)

        print("remaining traj", remaining_routes[0][0:3])
        print("removed traj", removed_routes[0][0:3])

        print("Routes In Anomalous Class:", len(removed_routes))

        X_train, X_test, _, _ = train_test_split(
            remaining_routes, np.zeros(len(remaining_routes)), test_size=0.2, random_state=42)

        '''Reshaping the data'''
        X_train = np.reshape(X_train, (len(X_train), TRIP_SIZE, NUM_VALS, 1))

        X_test_reshaped = np.reshape(X_test, (len(X_test) * TRIP_SIZE, NUM_VALS))
        np.savetxt("data/X_test_%d.txt" % id_remove, X_test_reshaped, fmt='%5s', delimiter=",")
        X_train_reshaped = np.reshape(X_train, (len(X_train) * TRIP_SIZE, NUM_VALS))
        np.savetxt("data/X_train_%d.txt" % id_remove, X_train_reshaped, fmt='%5s', delimiter=",")

        """# Model"""

        latent_dim = 100
        input_shape = (int(TRIP_SIZE), NUM_VALS, 1)

        # Generator

        modelG = Sequential()

        modelG.add(Dense(1024, input_dim=latent_dim))
        modelG.add(BatchNormalization(momentum=0.8))
        modelG.add(LeakyReLU(alpha=0.2))
        modelG.add(Dense(int(TRIP_SIZE / 4) * 1 * 128))
        modelG.add(BatchNormalization(momentum=0.8))
        modelG.add(LeakyReLU(alpha=0.2))
        modelG.add(Reshape((int(TRIP_SIZE / 4), 1, 128)))
        modelG.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"))
        modelG.add(BatchNormalization(momentum=0.8))
        modelG.add(LeakyReLU(alpha=0.2))
        modelG.add(Conv2DTranspose(1, kernel_size=4, strides=2, padding="same", activation='linear'))

        z = Input(shape=(latent_dim,))
        gen_img = modelG(z)
        generator = Model(z, gen_img)

        # Encoder

        modelE = Sequential()

        modelE.add(Conv2D(32, kernel_size=(3, 2), padding="same", input_shape=input_shape))
        modelE.add(BatchNormalization(momentum=0.8))
        modelE.add(Activation("relu"))
        modelE.add(MaxPooling2D(pool_size=(2, 2)))
        modelE.add(Conv2D(64, kernel_size=(3, 2), padding="same"))
        modelE.add(BatchNormalization(momentum=0.8))
        modelE.add(Activation("relu"))
        modelE.add(MaxPooling2D(pool_size=(2, 2)))
        modelE.add(Conv2D(128, kernel_size=(3, 2), padding="same"))
        modelE.add(BatchNormalization(momentum=0.8))
        modelE.add(Activation("relu"))
        modelE.add(Flatten())
        modelE.add(Dense(latent_dim))

        img = Input(shape=input_shape)
        z = modelE(img)
        encoder = Model(img, z)

        # Discriminator
        z = Input(shape=(latent_dim,))
        img = Input(shape=input_shape)

        modelDx = Conv2D(64, kernel_size=(3, 2))(img)
        modelDx = LeakyReLU(alpha=0.2)(modelDx)
        modelDx = Dropout(0.5)(modelDx)
        modelDx = Conv2D(64, kernel_size=(3, 1))(modelDx)
        modelDx = BatchNormalization(momentum=0.8)(modelDx)
        modelDx = LeakyReLU(alpha=0.2)(modelDx)
        modelDx = Dropout(0.5)(modelDx)
        modelDx = Flatten()(modelDx)

        modelDz = Dense(512)(z)
        modelDz = LeakyReLU(alpha=0.2)(modelDz)
        modelDz = Dropout(0.5)(modelDz)

        d_in = concatenate([modelDx, modelDz])

        modelD = Dense(1024)(d_in)
        modelD = LeakyReLU(alpha=0.2)(modelD)
        modelD = Dropout(0.5)(modelD)
        dis_fm = Model([z, img], modelD)
        validity = Dense(1, activation="sigmoid")(modelD)

        discriminator = Model([z, img], validity)

        learn_rate = 0.00001

        optimizer = Adam(learn_rate, 0.5)

        # Build and compile the discriminator
        discriminator.compile(loss=['binary_crossentropy'],
                              optimizer=optimizer,
                              metrics=['accuracy'])

        discriminator.trainable = False

        # Generate image from sampled noise
        z = Input(shape=(latent_dim,))
        img_ = generator(z)

        # Encode image
        img = Input(shape=input_shape)
        z_ = encoder(img)

        # Latent -> img is fake, and img -> latent is valid
        fake = discriminator([z, img_])
        valid = discriminator([z_, img])

        # Set up and compile the combined model
        # Trains generator to fool the discriminator
        bigan_generator = Model([z, img], [fake, valid])
        bigan_generator.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
                                optimizer=optimizer)

        batch_size = 128
        sample_interval = 50
        epochs = 40000

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        g_loss_list = []
        d_loss_list = []

        for epoch in range(epochs + 1):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample noise and generate img
            z = np.random.normal(size=(batch_size, latent_dim))
            imgs_ = generator.predict(z)

            # Select a random batch of images and encode
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            z_ = encoder.predict(imgs)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (z -> img is valid and img -> z is is invalid)
            g_loss = bigan_generator.train_on_batch([z, imgs], [valid, fake])

            # if epoch % 1 == 0:
            # Train the discriminator (img -> z is valid, z -> img is fake)
            d_loss_real = discriminator.train_on_batch([z_, imgs], valid)
            d_loss_fake = discriminator.train_on_batch([z, imgs_], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss_list.append(g_loss)
            d_loss_list.append(d_loss)

            # Plot the progress
            print ("%d - %d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (id_remove, epoch, d_loss[0], 100*d_loss[1], g_loss[0]))

            if epoch % 10000 == 0 and epoch > 0:
                learn_rate /= 2
                K.set_value(optimizer.lr, learn_rate)

            # If at save interval => save generated image samples

            if epoch % 5000 == 0 and epoch > 0:
                z = np.random.normal(size=(1, latent_dim))
                rand_img = generator.predict(z)[0]
                route = [[0.0, 0.0]]
                prev_x = 0.0
                prev_y = 0.0
                for i in range(1, len(rand_img)):
                    diff_x = unstandardize(mean_lat_diff, std_lat_diff, rand_img[i, 0, 0])
                    diff_y = unstandardize(mean_long_diff, std_long_diff, rand_img[i, 1, 0])
                    route.append([prev_x + diff_x, prev_y + diff_y])
                    prev_x += diff_x
                    prev_y += diff_y
                route_np = np.asarray(route)
                plt.plot(route_np[:, 0], route_np[:, 1])
                plt.savefig("imgs/img_%d_%d.png" % (id_remove, epoch))
                plt.close()

                z = np.random.normal(size=(1, latent_dim))
                rand_img = generator.predict(z)
                rand_img = np.reshape(rand_img[0, :, 2:], (TRIP_SIZE, 2))
                np.savetxt("imgs/map_%d_%d" % (id_remove, epoch), rand_img)

        plt.plot(np.asarray(g_loss_list)[:, 0], label='G loss')
        plt.plot(np.asarray(d_loss_list)[:, 0], label='D loss')
        plt.plot(np.asarray(d_loss_list)[:, 1], label='D accuracy')
        plt.legend(bbox_to_anchor=(1, 1))
        plt.savefig("loss_all/loss_%d.png" % id_remove, bbox_inches='tight', pad_inches=1)
        plt.close()

        loss_all = np.asarray([np.asarray(g_loss_list)[:, 0], np.asarray(d_loss_list)[:, 0], np.asarray(d_loss_list)[:, 1]])
        np.savetxt("loss_all/loss_%d.txt" % id_remove, loss_all, fmt='%5s', delimiter=",")

        """# Save Model"""

        generator.save_weights('models/gen_weights_%d.h5' % id_remove)
        with open('models/gen_architecture_%d.json' % id_remove, 'w') as f:
            f.write(generator.to_json())

        encoder.save_weights('models/enc_weights_%d.h5' % id_remove)
        with open('models/enc_architecture_%d.json' % id_remove, 'w') as f:
            f.write(encoder.to_json())

        discriminator.save_weights('models/dis_weights_%d.h5' % id_remove)
        with open('models/dis_architecture_%d.json' % id_remove, 'w') as f:
            f.write(discriminator.to_json())

        X_test_remaining = X_test.copy()
        X_test = np.vstack([removed_routes, X_test_remaining])
        X_test = np.reshape(X_test, (len(X_test), TRIP_SIZE, NUM_VALS, 1))

        Y_test = np.hstack([np.ones(len(removed_routes)), np.zeros(len(X_test_remaining))])

        weight = 0.9

        z_gen_ema = encoder.predict(X_test)
        reconstruct_ema = generator.predict(z_gen_ema)
        l_generator_ema = discriminator.predict([z_gen_ema, reconstruct_ema])

        init = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init)

        # Cross entropy

        dis_score = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(l_generator_ema), logits=l_generator_ema)

        dis_score_val = sess.run(dis_score)

        val_list = []

        for i in range(0, len(X_test)):
            # Finding the anomaly score
            delta = X_test[i] - reconstruct_ema[i]
            delta_flat = np.ndarray.flatten(delta)
            gen_score = np.linalg.norm(delta_flat)

            # Total anomaly score
            value = (1 - weight) * gen_score + weight * dis_score_val[i][0]
            val_list.append(value)

        val_arr = np.asarray(val_list)
        val_probs = val_arr / max(val_arr)

        roc_auc = roc_auc_score(Y_test, val_probs)
        prauc = average_precision_score(Y_test, val_probs)
        roc_auc_scores.append(roc_auc)
        prauc_scores.append(prauc)

        print("ROC AUC SCORE FOR %d: %f" % (id_remove, roc_auc))
        print("PRAUC SCORE FOR %d: %f" % (id_remove, prauc))

        np.savetxt('auc/roc_auc_scores.txt', roc_auc_scores, fmt='%5s', delimiter=",")
        plt.scatter(np.arange(len(roc_auc_scores)), roc_auc_scores)
        plt.savefig("auc/roc_auc_scores.png")
        plt.close()

        np.savetxt('auc/prauc_scores.txt', prauc_scores, fmt='%5s', delimiter=",")
        plt.scatter(np.arange(len(prauc_scores)), prauc_scores)
        plt.savefig("auc/prauc_scores.png")
        plt.close()