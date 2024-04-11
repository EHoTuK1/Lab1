import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_weights = np.random.rand(input_size, hidden_size)
        self.hidden_bias = np.random.rand(hidden_size)
        self.output_weights = np.random.rand(hidden_size, output_size)
        self.output_bias = np.random.rand(output_size)

    def activate(self, inputs):
        hidden_output = 1 / (1 + np.exp(-(np.dot(inputs, self.hidden_weights) + self.hidden_bias)))
        output = 1 / (1 + np.exp(-(np.dot(hidden_output, self.output_weights) + self.output_bias)))
        return output

    def train(self, inputs, target_outputs, learning_rate):
        hidden_output = 1 / (1 + np.exp(-(np.dot(inputs, self.hidden_weights) + self.hidden_bias)))
        output = 1 / (1 + np.exp(-(np.dot(hidden_output, self.output_weights) + self.output_bias)))

        output_error = target_outputs - output
        output_delta = output_error * output * (1 - output)

        hidden_error = np.dot(output_delta, self.output_weights.T)
        hidden_delta = hidden_error * hidden_output * (1 - hidden_output)

        self.output_weights += learning_rate * np.dot(hidden_output.T, output_delta)
        self.output_bias += learning_rate * np.sum(output_delta, axis=0)

        self.hidden_weights += learning_rate * np.dot(inputs.T, hidden_delta)
        self.hidden_bias += learning_rate * np.sum(hidden_delta, axis=0)


neural_net = NeuralNetwork(input_size=2, hidden_size=4, output_size=2)

inputs = np.array([[0.75, 0.34], [0.2, 0.9], [0.1, 0.5]])
target_outputs = np.array([[0.34, 0.75], [0.9, 0.2], [0.5, 0.1]])
learning_rate = 0.01
iterations = 100000

for i in range(iterations):
    neural_net.train(inputs, target_outputs, learning_rate)

test_input = np.array([[0.76, 0.34], [0.13, 0.98]])
output = neural_net.activate(test_input)
print(test_input)
print(output)
