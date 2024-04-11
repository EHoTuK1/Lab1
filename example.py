import numpy as np

class ArtificialNeuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def activate(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return 1 / (1 + np.exp(-weighted_sum))

    def train(self, inputs, target_output, learning_rate):
        output = self.activate(inputs)
        error = target_output - output
        gradient = output * (1.0 - output)
        delta_weight = learning_rate * gradient * error * inputs
        self.bias += learning_rate * error * gradient
        self.weights += delta_weight


weights = np.random.rand(2)
bias = np.random.rand()

neuron = ArtificialNeuron(weights, bias)

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target_outputs = np.array([[0], [1], [1], [1]])
learning_rate = 0.1

for i in range(10000):
    for j in range(len(inputs)):
        neuron.train(inputs[j], target_outputs[j], learning_rate)
output = neuron.activate(np.array([1,1]))
print(output)
print(weights[0])
print(bias)