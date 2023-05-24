# coding=utf-8
""" Warstwa ReLU. """

import numpy as np

from Layer import Layer


class ReLULayer(Layer):
    def __init__(self):
        # ReLU layer simply applies elementwise rectified linear unit to all inputs
        super().__init__()

    def forward(self, input_layer: np.ndarray) -> np.ndarray:
        # Apply elementwise ReLU to [batch, input_units] matrix
        return np.maximum(0, input_layer)

    def backward(self, input_layer: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        # Compute gradient of loss w.r.t. ReLU input
        return grad_output * (input_layer > 0)
