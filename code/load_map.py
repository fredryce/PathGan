import os,glob,sys
from matplotlib import pyplot as plt
from xml.dom import minidom
import math
import numpy as np
import pandas


MAP_XML = "./../map/london-seg4/100/london-seg4.100.sumocfg"

from networkxbuild import NXSUMO

class Junctions(object):
	def __init__(self, coord, junction_id):
		#each junction would contain a utility matrix showing
		self.junction_id = junction_id
		self.coord=coord
		self.adjacent_edges_to = [] #what edges this junction goes to
		self.adjacent_edges_from = [] # what edges goes to this junction
		self.utility = {}
		self.x = self.coord[0]
		self.y = self.coord[1]
		self.number_players = 0
		self.adjacent_junctions = [] # adjacent junctions can be traveled to. can be used to calculate the probability when player in this cell
		self.cost = 5


class MapData(object):
	def __init__(self):
		self.sumonet = NXSUMO()
		self.junctions = {}
	def plot_map(self):
		pass


	@staticmethod
	def get_distance(x2,y2,x1,y1):
		return math.sqrt((x2 - x1)**2+(y2 - y1)**2)

	def calculate_distance(self, junc_from, junc_to):
		return MapData.get_distance(self.junctions[junc_to].x, self.junctions[junc_to].y, self.junctions[junc_from].x, self.junctions[junc_from].y)


	def parse_map(self, csv_format=False, show=False): #new parse map for grid london
		coords = []
		print("parsing map...")
		edge_file = MAP_XML.replace(".sumocfg", ".edg.xml")
		node_file = MAP_XML.replace(".sumocfg", ".nod.xml")

		assert os.path.exists(edge_file) and os.path.exists(node_file), f"Check node file and edge file {edge_file} {node_file}"

		print("parsing edge file..")
		edge_xml = minidom.parse(edge_file)
		print("parsing node file...")
		node_xml = minidom.parse(node_file)

		edge_list = [x for x in edge_xml.getElementsByTagName('edge')]
		junction_list = [x for x in node_xml.getElementsByTagName('node')]
		
		for item in junction_list:
			junct_id = item.attributes['id'].value
			self.junctions[junct_id] = Junctions((float(item.attributes['x'].value), float(item.attributes['y'].value)), item.attributes['id'].value)
			coords.append([float(item.attributes['x'].value), float(item.attributes['y'].value)])

		for item in edge_list:
			self.sumonet.add(item.attributes['from'].value, item.attributes['to'].value, weight=self.calculate_distance(item.attributes['from'].value, item.attributes['to'].value))

		if csv_format:
			pd.DataFrame(coords,columns=["x", "y"]).to_csv("mapjunctions.csv")
		if show:
			coords=np.array(coords)
			plt.scatter(coords.T[0], coords.T[1],color=(0,0,0), alpha=1, s=1)

			plt.show()

		print(coords)

		print(f"Parsing completed.... {len(junction_list)} junctions ")

if __name__ == "__main__":
	map_value = MapData()
	map_value.parse_map(csv_format=False, show=True)
	#map_value.sumonet.show()