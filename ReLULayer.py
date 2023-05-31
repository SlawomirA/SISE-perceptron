# coding=utf-8
""" Warstwa ReLU. """

import numpy as np

from Layer import Layer


class ReLULayer(Layer):
    def __init__(self):
        """
        Warstwa zwraca zero jeśli wartość jest mniejsza od 0, w innym przypadku zwraca tę wartość.
        """
        super().__init__()

    def forward(self, input_layer: np.ndarray) -> np.ndarray:
        """
        Oblicza odpowiedź warstwy (wyjście dla wejścia)
        :param input_layer:
        :return:
        """
        return np.maximum(0, input_layer)

    def backward(self, input_layer: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        Oblicza wartość gradientu straty dla obecnej warstwy
        :param input_layer:
        :param grad_output:
        :return:
        """
        return grad_output * (input_layer > 0)
