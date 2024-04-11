import numpy as np
import matplotlib.pyplot as plt

class ArtificialNeuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def activate(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

    def train(self, inputs, target_output, learning_rate):
        output = self.activate(inputs)
        error = target_output - output
        delta_weight = learning_rate * np.dot(inputs.T, error)
        self.bias += learning_rate * error
        self.weights += delta_weight
        #print(inputs, target_output, output, self.weights, error, delta_weight)

weights = np.random.rand(1, 1)
#bias = np.random.rand()
bias = 0

neuron = ArtificialNeuron(weights, bias)

inputs = np.array([[i * 15] for i in range(0, 25)])
target_outputs = np.deg2rad(inputs)
learning_rate = 1e-7

weights_history = []
bias_history = []
error_history = []

iterations = 15000

for i in range(iterations):
    for j in range(len(inputs)):
        neuron.train(inputs[j], target_outputs[j], learning_rate)

        weights_history.append(neuron.weights[0][0])
        bias_history.append(neuron.bias)
        error = target_outputs[j] - neuron.activate(inputs[j])
        error_history.append(error)

test_input = np.array([[0], [45], [90], [180], [270], [360]])
predicted_output = neuron.activate(test_input)

for i in range(len(predicted_output)):
    print(predicted_output[i], np.deg2rad(test_input[i]))

plt.figure(figsize=(7, 7))
plt.subplot(3, 1, 1)
plt.plot(weights_history)
plt.title('Изменение весов во время обучения')
plt.xlabel('Итерация')
plt.ylabel('Вес')
plt.xlim(0, iterations)

plt.subplot(3, 1, 2)
plt.plot(error_history)
plt.title('Изменение ошибки во время обучения')
plt.xlabel('Итерация')
plt.ylabel('Ошибка')
plt.xlim(0, iterations)

plt.tight_layout()
plt.show()
