from node import Node
from constant_node import ConstantNode

class Layer:
    def __init__(self, layer_index, node_count):
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstantNode(layer_index, node_count))

    def set_output(self, data):
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calculate_output(self):
        for node in self.nodes[:-1]:
            node.calculate_output()

    def dump(self):
        for node in self.nodes:
            print(node)
