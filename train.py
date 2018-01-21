from loader import get_train_dataset, get_test_dataset

def get_result(vector):
    max_value_index = 0
    max_value = 0
    for i in range(len(vector)):
        if vector[i] > max_value:
            max_value = vector[i]
            max_value_index = i
    return max_value_index

def evaluate(network, test_dataset, test_labels):
    error = 0
    total = len(test_dataset)
    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(network.predict(test_dataset[i]))
        if label != predict:
            error += 1
    return float(error) / float(total)

def train_and_evaluate():
    last_error_ratio = 1.0
    epoch = 0
    train_dataset, train_labels = get_train_dataset()
    test_dataset, test_labels = get_test_dataset()
    network = Network([784, 300, 10])
    while True:
        epoch += 1
        network.train(train_labels, train_dataset, 0.3, 1)
        print('%s epoch %d finished' % (now(), epoch))
        if epoch % 10 == 0:
            error_ratio = evaluate(network, test_dataset, test_labels)
            print('%s after epoch %d, error ratio is %f' % (now(), epoch, error_ratio))
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio

if __name__ == '__main__':
    train_and_evaluate()
