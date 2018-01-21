def network_error(vector1, vector2):
    return 0.5 * reduce(lambda a, b: a + b, map(lambda v: (v[0] - v[1]) * (v[0] - v[1]), zip(vector1, vector2)))

def gradient_check(network, sample_feature, sample_label):
    network.get_gradient(sample_feature, sample_label)

    for connection in network.connections.connections:
        actual_gradient = connection.get_gradient()

        epsilon = 0.0001
        connection.weight += epsilon
        error1 = network_error(network.predict(sample_feature), sample_label)

        connection.weight -= 2 * epsilon
        error2 = network_error(network.predict(sample_feature), sample_label)

        expected_gradient = (error2 - error1) / (@ * epsilon)

        print('expected graident: \t%f\nactual graident: \t%f' % (expected_gradient, actual_gradient))
