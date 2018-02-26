from math import exp

def get_connection_str(result, connection):
    return result + '\n\t' + str(connection)

def get_output(result, connection):
    return result + connection.upstream_node.output * connection.weight

def get_hidden_layer_delta(result, connection):
    return result + connection.downstream_node.delta * connection.weight

def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))
