from connections import Connections
from layer import Layer
from connection import Connection


class Network:
    def __init__(self, layers_node_count):
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers_node_count)
        node_count = 0
        for i in range(layer_count):
            self.layers.append(Layer(i, layers_node_count[i]))
        for layer in range(layer_count - 1):
            connections = [Connection(upstream_node, downstream_node)
                           for upstream_node in self.layers[layer].nodes
                           for downstream_node in self.layers[layer + 1].nodes[:-1]]
            for connection in connections:
                self.connections.append_connection(connection)
                connection.downstream_node.append_upstream_connection(connection)
                connection.upstream_node.append_downstream_connection(connection)

    def train(self, labels, dataset, rate, iteration):
        for i in range(iteration):
            for d in range(len(dataset)):
                self.train_one_sample(labels[d], dataset[d], rate)

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calculate_delta(label)
        self.update_weight(rate)

    def calculate_delta(self, label):
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calculate_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calculate_hidden_layer_delta()

    def update_weight(self, rate):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for connection in node.downstream:
                    connection.update_weight(rate)

    def calculate_gradient(self):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for connection in node.downstream:
                    connection.calculate_gradient()

    def get_gradient(self, label, sample):
        self.predict(sample)
        self.calculate_delta(label)
        self.calculate_gradient()

    def predict(self, sample):
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calculate_output()
        return map(lambda node: node.output, self.layers[-1].nodes[:-1])

    def dump(self):
        for layer in self.layers:
            layer.dump()
