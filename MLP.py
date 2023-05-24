# coding=utf-8
""" Sieć MLP. """

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pickle

from tqdm import trange
from IPython.display import clear_output
from typing import List, Tuple

from DenseLayer import DenseLayer
from ImageData import ImageData
from ReLULayer import ReLULayer


class MLP:
    def __init__(self, sharpe: List[Tuple[int, int]] = None, learning_rate: float = 0.1, model_path: str = ""):
        np.random.seed(42)

        if sharpe is None:
            sharpe = [(784, 100), (0, 0), (100, 200), (0, 0), (200, 10)]
        self.network = []
        for layer_sharpe in sharpe:
            self.network.append(DenseLayer(layer_sharpe[0], layer_sharpe[1], learning_rate) if layer_sharpe[0] != 0 else ReLULayer())

        if model_path != "":
            try:
                self.__dict__.update(self.load(model_path).__dict__)
            except TypeError:
                self.__dict__.update(self.load(model_path))

    def save(self, file_path: str):
        with open(file_path, "wb") as mypicklefile:
            pickle.dump(self, mypicklefile)

    @staticmethod
    def load(file_path: str):
        with open(file_path, "rb") as mypicklefile:
            return pickle.load(mypicklefile)

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True)

    @staticmethod
    def softmax_crossentropy(received_responses: np.ndarray, expected_responses: np.ndarray) -> np.ndarray:
        # Compute crossentropy from logits[batch,n_classes] and ids of correct answers
        return - received_responses[np.arange(len(received_responses)), expected_responses] + np.log(np.sum(np.exp(received_responses), axis=-1))

    @staticmethod
    def grad_softmax_crossentropy(received_responses: np.ndarray, expected_responses: np.ndarray) -> np.ndarray:
        # Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers
        ones_for_answers = np.zeros_like(received_responses)
        ones_for_answers[np.arange(len(received_responses)), expected_responses] = 1
        return (- ones_for_answers + MLP.softmax(received_responses)) / received_responses.shape[0]

    def forward(self, X: np.ndarray) -> List[np.ndarray]:
        # Compute activations of all network layers by applying them sequentially.
        # Return a list of activations for each layer.
        activations = []
        input_x = X
        # Looping through each layer
        for layer in self.network:
            activations.append(layer.forward(input_x))
            # Updating input to last layer output
            input_x = activations[-1]

        assert len(activations) == len(self.network)
        return activations

    def predict(self, imput_images: np.ndarray, show_all_classes: bool = False) -> np.ndarray:
        # Compute network predictions. Returning indices of largest Logit probability
        responses = self.forward(imput_images)[-1]
        return responses if show_all_classes else responses.argmax(axis=-1)

    def train(self, input_images: np.ndarray, expected_labels: np.ndarray) -> np.ndarray:
        # Train our network on a given batch of input_values and expected_values.
        # We first need to run forward to get all layer activations.
        # Then we can run layer.backward going from last to first layer.
        # After we have called backward for all layers, all Dense layers have already made one gradient step.

        # Get the layer activations
        layer_activations = self.forward(input_images)
        layer_inputs = [input_images] + layer_activations  # layer_input[i] is an input for network[i]
        received_responses = layer_activations[-1]

        # Compute the loss and the initial gradient
        loss = MLP.softmax_crossentropy(received_responses, expected_labels)
        loss_grad = MLP.grad_softmax_crossentropy(received_responses, expected_labels)

        # Propagate gradients through the network
        # Reverse propogation as this is backprop
        for i in range(len(self.network))[::-1]:
            layer = self.network[i]
            loss_grad = layer.backward(layer_inputs[i], loss_grad)  # grad w.r.t. input, also weight updates
        return np.mean(loss)

    def generate_minibatches(self, image_data: ImageData, batch_size: int = 32, shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        assert len(image_data.train_images) == len(image_data.train_labels)
        indices = np.random.permutation(len(image_data.train_images)) if shuffle else None
        for start_idx in trange(0, len(image_data.train_images) - batch_size + 1, batch_size):
            excerpt = indices[start_idx:start_idx + batch_size] if shuffle else slice(start_idx, start_idx + batch_size)
            yield image_data.train_images[excerpt], image_data.train_labels[excerpt]

    def run(self, image_data: ImageData = None, epochs: int = 5, verbose: bool = True, show_plot: bool = True,
            model_path: str = "") -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if image_data is None:
            image_data: ImageData = ImageData()
            image_data.load_mnist_dataset()

        train_log = []
        validation_log = []
        for epoch in range(epochs):
            for input_values, expected_values in self.generate_minibatches(image_data):
                self.train(input_values, expected_values)

            train_log.append(np.mean(self.predict(image_data.train_images) == image_data.train_labels))
            validation_log.append(np.mean(self.predict(image_data.validation_images) == image_data.validation_labels))
            if verbose:
                clear_output()
                print(f"Epoch: {epoch}.")
                print(f"Train accuracy: {train_log[-1]}.")
                print(f"Validation accuracy: {validation_log[-1]}.")
        if show_plot:
            plt.plot(train_log, label='train accuracy')
            plt.plot(validation_log, label='validation accuracy')
            plt.legend(loc='best')
            plt.grid()
            plt.show()

        if model_path != "":
            self.save(model_path)

        return train_log, validation_log
