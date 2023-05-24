# coding=utf-8
""" Warstwa w pełni połączona — Dense. """

import numpy as np

from Layer import Layer


class DenseLayer(Layer):
    def __init__(self, input_units: int, output_units: int, learning_rate: float = 0.1):
        # A dense layer is a layer which performs a learned affine transformation:
        # f(x) = <W*x> + b
        super().__init__()
        self.learning_rate: float = learning_rate
        self.weights: np.ndarray = np.random.normal(loc=0.0, scale=np.sqrt(2 / (input_units + output_units)), size=(input_units, output_units))
        self.biases: np.ndarray = np.zeros(output_units)

    def forward(self, input_layer: np.ndarray) -> np.ndarray:
        # Perform an affine transformation:
        # f(x) = <W*x> + b
        # input shape: [batch, input_units]
        # output shape: [batch, output units]
        return np.dot(input_layer, self.weights) + self.biases

    def backward(self, input_layer: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = np.dot(grad_output, self.weights.T)

        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(input_layer.T, grad_output)
        grad_biases = grad_output.mean(axis=0) * input_layer.shape[0]
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape

        # Here we perform a stochastic gradient descent step.
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input
