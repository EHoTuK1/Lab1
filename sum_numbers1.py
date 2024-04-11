import numpy as np

class ArtificialNeuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def activate(self, inputs):
        return self.sigmoid(np.dot(inputs, self.weights) + self.bias)

    def train(self, inputs, target_output, learning_rate):
        output = self.activate(inputs)
        error = target_output - output
        gradient = error * output * (1 - output)  # Производная сигмоиды
        delta_weight = learning_rate * inputs * gradient
        self.bias += learning_rate * error
        self.weights += delta_weight


weights = np.random.rand(2)
bias = 0

neuron = ArtificialNeuron(weights, bias)

inputs = np.array([[1, 0], [0, 1], [1, 1], [1, 2], [2, 1], [2, 2], [3, 2],[2, 3]])
target_outputs = np.array([[1], [1], [2], [3], [3], [4], [5], [5]])
learning_rate = 0.1

for i in range(10000):
    for j in range(len(inputs)):
        neuron.train(inputs[j], target_outputs[j], learning_rate)

output = neuron.activate(np.array([1, 8]))
print(output)
