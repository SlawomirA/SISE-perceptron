# coding=utf-8
""" Warstwa w pełni połączona — Dense. """

import numpy as np

from Layer import Layer


class DenseLayer(Layer):
    def __init__(self, input_units: int, output_units: int, learning_rate: float = 0.1):
        """
        Warstwa głęboka jest warstwą w której każdy neuron warstwy poprzedniej jest łączony z każdym neuronem tej warstwy.
        Neuron wylicza swoją odpowiedź na podstawie wzoru f(x) = W*x + b gdzie:
        y - odp neuronu
        W - waga
        x - wejście
        b - bias (na początku 0)
        :param input_units: liczba wejść
        :param output_units: liczba neuronów
        :param learning_rate: współczynnnik nauki
        """
        super().__init__()
        self.learning_rate: float = learning_rate
        self.weights: np.ndarray = np.random.normal(loc=0.0, scale=np.sqrt(2 / (input_units + output_units)), size=(input_units, output_units))
        self.biases: np.ndarray = np.zeros(output_units)

    def forward(self, input_layer: np.ndarray) -> np.ndarray:
        """
        Oblicza odpowiedź sieci na podstawie f(x) = Wx+b
        :param input_layer: dane wejściowe
        :return: wyjście, lista float
        """
        return np.dot(input_layer, self.weights) + self.biases

    def backward(self, input_layer: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        Wylicza gradient błędu wejścia.  d f / d x = d f / d dense * d dense / d x
        gdzier d dense/ d x = weights transposed
        :param input_layer: Wyjście sieci
        :param grad_output: Gradient błędu wyjścia
        :return: gradient błedu wejścia
        """
        grad_input = np.dot(grad_output, self.weights.T)

        #Wylicza gradient wag na podstawie gradientu wyjścia
        grad_weights = np.dot(input_layer.T, grad_output)
        #Średnia gradientu wyjścia, potem wyrzuć gradienty wag, w ten sposób zostawiając gradienty biasu
        grad_biases = grad_output.mean(axis=0) * input_layer.shape[0]
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape

        #Aktualizacja wag i biasu
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input
