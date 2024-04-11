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

weights = np.random.rand(1, 1)
bias = 0

neuron = ArtificialNeuron(weights, bias)

inputs = np.array([[0], [45], [90], [180], [270], [360]])
target_outputs = np.deg2rad(inputs)
learning_rate = 0.00000001

weights_history = []
bias_history = []
error_history = []

import numpy as np
import matplotlib.pyplot as plt

# Определение класса ArtificialNeuron и остального кода...

learning_rate = 0.000001
min_error = float('inf')
best_learning_rate = None

while learning_rate >= 1e-15:  # Проверка на достижение минимального learning_rate
    print("Testing learning rate:", learning_rate)
    total_error = 0

    # Прогоняем 10000 вычислений несколько раз
    for _ in range(5):
        neuron = ArtificialNeuron(weights, bias)
        error_history = []
        for i in range(10000):
            for j in range(len(inputs)):
                neuron.train(inputs[j], target_outputs[j], learning_rate)
                error = target_outputs[j] - neuron.activate(inputs[j])
                error_history.append(error)
        total_error += np.mean(np.abs(error_history))  # Суммируем среднюю ошибку на каждой итерации обучения
    if total_error is None:
        continue
    # Средняя ошибка для текущего learning_rate
    avg_error = total_error / 5

    # Обновляем минимальную ошибку и лучшее значение learning_rate, если это так
    if avg_error < min_error:
        min_error = avg_error
        best_learning_rate = learning_rate
        best_weights = neuron.weights

    # Уменьшаем learning_rate в 10 раз
    learning_rate /= 10

print("Best learning rate:", best_learning_rate)
print("Best weights:", best_weights)
test_input = [i*15 for i in range(0,25)]
test_input = np.array(test_input).reshape(-1,1)
predicted_output = neuron.activate(test_input)

for i in range(len(predicted_output)):
    print(predicted_output[i], np.deg2rad(test_input[i]))






plt.figure(figsize=(12, 4))
# Подграфик для визуализации изменения весов
plt.subplot(1, 2, 1)
plt.plot(weights_history)
plt.title('Изменение весов во время обучения')
plt.xlabel('Итерация')
plt.ylabel('Вес')
plt.xlim(0, 10000)

# Подграфик для визуализации изменения смещения
plt.subplot(1, 2, 2)
plt.plot(error_history)
plt.title('Изменение ошибки во время обучения')
plt.xlabel('Итерация')
plt.ylabel('Ошибка')
plt.xlim(0, 10000)

plt.tight_layout()
plt.show()


