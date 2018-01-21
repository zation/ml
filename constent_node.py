from utils import get_hidden_layer_delta, get_connection_str

class ConstantNode:
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 0

    def append_downstream_connection(self, connection):
        seld.downstream.append(connection)

    def calculate_hidden_layer_delta(self):
        downstream_delta = reduce(get_hidden_layer_delta, self.downstream, 0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        node_str = '%u - %u: output: 1' % (self.layout_index, self.node_index)
        downstream_str = reduce(get_connection_str, self.downstream, '')
        return node_str + '\n\tdownsteam: ' + downstream_str
