import numpy as np
import matplotlib.pyplot as plt

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
        gradient = output * (1 - output)
        delta_weights = learning_rate * error * gradient * inputs
        delta_bias = learning_rate * error * gradient
        self.bias += delta_bias
        self.weights += delta_weights

weights = np.random.rand(2) * 0.05
bias = np.random.rand() * 0.05
neuron = ArtificialNeuron(weights, bias)

learning_rate = 0.0000001
iterations = 100

inputs = []
outputs = []
for i in range(100):
    for j in range(100):
        input_val = [i/100, j/100]
        output_val = [j/100, i/100]
        inputs.append(input_val)
        outputs.append(output_val)
input_data = np.array(inputs)
output_data = np.array(outputs)
indices = np.arange(len(input_data))

weights_history = []
bias_history = []
error_history = []

for i in range(iterations):
    np.random.shuffle(indices)
    inputs = input_data[indices]
    output_data = output_data[indices]
    for j in range(len(input_data)):
        neuron.train(input_data[j], output_data[j], learning_rate)
        weights_history.append(neuron.weights)
        bias_history.append(neuron.bias)
        error = output_data[j] - neuron.activate(input_data[j])
        error_history.append(np.linalg.norm(error))
    print(i)

print(neuron.activate([0.35, 0.72]))
print(neuron.weights)

plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)
plt.plot(weights_history)
plt.title('Weights during training')
plt.xlabel('Iteration')
plt.ylabel('Weights')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(error_history)
plt.title('Error ')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.grid(True)

plt.tight_layout()
plt.show()
