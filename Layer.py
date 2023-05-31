# coding=utf-8
""" Bazowa klasa dla warstw sieci MLP. """
import numpy as np

from abc import abstractmethod


class Layer:
    # Podstawowa, bazowa klasa dla wszystkich warstw. Każda warstwa jest w stanie wykonać dwie czynności:
    # - przetworzyć dane wejściowe w celu uzyskania danych wyjściowych: output = layer.forward(input)
    # - propagować gradienty przez samą siebie: grad_input = layer.backward(input, grad_output)
    # Niektóre warstwy mają również parametry, których można się nauczyć i które są aktualizowane podczas layer.backward.

    @abstractmethod
    def __init__(self):
        pass

    def forward(self, input_layer: np.ndarray) -> np.ndarray:
        return input_layer

    def backward(self, input_layer: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        return np.dot(grad_output, np.eye(input_layer.shape[1]))
