from utils import get_connection_str, get_output, get_hidden_layer_delta, sigmoid
from functools import reduce

class Node:
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        self.output = output

    def append_downstream_connection(self, connection):
        self.downstream.append(connection)

    def append_upstream_connection(self, connection):
        self.upstream.append(connection)

    def calculate_output(self):
        output = reduce(get_output, self.upstream, 0)
        self.output = sigmoid(output)

    def calculate_hidden_layer_delta(self):
        downstream_delta = reduce(get_hidden_layer_delta, self.downstream, 0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calculate_output_layer_delta(self, label):
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(get_connection_str, self.downstream, '')
        upstream_str = reduce(get_connection_str, self.upstream, '')
        return node_str + '\n\tdownstream: ' + downstream_str + '\n\tupsteam: ' + upstream_str
