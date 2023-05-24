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
        # Takes input data of shape [batch, input_units], returns output data [batch, output_units]
        # A dummy layer just returns whatever it gets as input.
        return input_layer

    def backward(self, input_layer: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        # Performs a backpropagation step through the layer, with respect to the given input.
        # To compute loss gradients w.r.t input, we need to apply chain rule (backprop):
        # d loss / d x  = (d loss / d layer) * (d layer / d x)
        # Luckily, we already receive d loss / d layer as input, so you only need to multiply it by d layer / d x.
        # If our layer has parameters (e.g. dense layer), we also need to update them here using d loss / d layer
        # The gradient of a dummy layer is precisely grad_output, but we'll write it more explicitly
        return np.dot(grad_output, np.eye(input_layer.shape[1]))
