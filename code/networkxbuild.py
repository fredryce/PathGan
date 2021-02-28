import networkx as nx
from matplotlib import pyplot as plt
class NXSUMO(object):
    def __init__(self):
        #self.G = nx.Graph()
        self.G=nx.DiGraph(directed=True)
        #node to ndoe
    def add(self, start, end, weight):
        self.G.add_edge(start, end, weight=weight)

    def find_shortest_path(self, start, end): #return list
        return(nx.shortest_path(self.G, start, end, weight='weight'))

    def find_shortest_length(self, start, end):
        return(nx.shortest_path_length(self.G, start,end, weight="weight"))

    def sub_new(self, origin_node, cutoff, dest_node=None):
        #return(nx.single_source_shortest_path_length(self.G, origin_node, cutoff))
        #return(nx.single_source_dijkstra(self.G, origin_node, target=dest_node ,cutoff=cutoff))
        return(nx.single_source_dijkstra_path_length(self.G, origin_node, cutoff=cutoff))
    def show(self):
        options = {
        'node_color': 'white',
        'node_size': 1000,
        'width': 3,
        'arrowstyle': '-|>',
        'arrowsize': 12,
        }
        nx.draw_networkx(self.G, cmap = plt.get_cmap('jet'), arrows=True, **options)
        plt.show()



if __name__ == "__main__":

    options = {
    'node_color': 'white',
    'node_size': 1000,
    'width': 3,
    'arrowstyle': '-|>',
    'arrowsize': 12,
    }
    test = NXSUMO()

    test.add('A', 'B', weight=3)
    test.add('B', 'D', weight=2)
    test.add('A', 'C', weight=1)
    test.add('C', 'D', weight=2)
    test.add('C', 'E', weight=1)
    test.add('E', 'A', weight=5)
    test.add('D', 'B', weight=2)
    test.add('A', 'F', weight=5)
    test.add('F', 'E', weight=2)

    print(test.find_shortest_path("A","D"))
    #print(test.find_shortest_length("A","D"))
    path = test.sub_new("A", 100)
    print(path)
    
    nx.draw_networkx(test.G, cmap = plt.get_cmap('jet'), arrows=True, **options)
    plt.show()