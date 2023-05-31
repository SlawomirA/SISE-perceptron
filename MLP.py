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

        #Sprawdzanie, czy podano kształt sieci, jeśli nie, zastosuj domyślny (gęsta-aktywacji-gęsta-aktywacji-wyjściowa)
        #(Ile wejść dla każdego z neuronów, liczba neuronów)
        if sharpe is None:
            sharpe = [(784, 100), (0, 0), (100, 200), (0, 0), (200, 10)]
        self.network = []
        #Przechodzenie po wszystkich kształtach warstwy, jeśli pierwsza liczba jest 0 to jest to warstwa aktywacji
        for layer_sharpe in sharpe:
            self.network.append(DenseLayer(layer_sharpe[0], layer_sharpe[1], learning_rate) if layer_sharpe[0] != 0 else ReLULayer())
        #Ścieżka do istniejącego wytrenowanego modelu
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
        """
        Wylicza jak duży błąd popełnił model, funkcja straty
        :param x:lista floatów, tyle ile neuronów jest w warstwie
        :return: Błąd odpowiedzi warstwy
        """
        return np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True)

    @staticmethod
    def softmax_crossentropy(received_responses: np.ndarray, expected_responses: np.ndarray) -> np.ndarray:
        """
        Entropia krzyżowa jest to wartość oceniająca wartość odpowiedzi podczas procesu nauki CE = - sum(y * log(p))
        :param received_responses: Odpowiedzi które otrzymano
        :param expected_responses: Odpowiedzi które oczekiwano
        :return:
        """
        return - received_responses[np.arange(len(received_responses)), expected_responses] + np.log(np.sum(np.exp(received_responses), axis=-1))

    @staticmethod
    def grad_softmax_crossentropy(received_responses: np.ndarray, expected_responses: np.ndarray) -> np.ndarray:
        """
        Gradient dla funkcji straty (softmax + entropia krzyżowa). Gradient to pochodna cząstkowa funkcji wielu
        zmiennych względem każdej zmiennej. Wzór to pochodna z softmax_crossentropy czyli (-1+softmax)/y
        :param received_responses:
        :param expected_responses:
        :return:
        """
        ones_for_answers = np.zeros_like(received_responses)
        ones_for_answers[np.arange(len(received_responses)), expected_responses] = 1
        return (- ones_for_answers + MLP.softmax(received_responses)) / received_responses.shape[0]

    def forward(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Obliczanie wartości sieci neuronowej przy podanych werjściach
        :param X:
        :return:
        """
        # Wyliczanie odpowiedzi sieci
        activations = []
        input_x = X
        for layer in self.network:
            activations.append(layer.forward(input_x))
            # Przypisz obecne wyjście sieci jako wejście kolejnej
            input_x = activations[-1]

        #Liczba odpowiedzi musi być równa liczbie wartsw
        assert len(activations) == len(self.network)
        return activations

    def predict(self, imput_images: np.ndarray, show_all_classes: bool = False) -> np.ndarray:
        """
        Oblicza odpowiedź sieci dla każdej z klas na podstawie podanego wejścia.
        :param imput_images: Tablica zawierająca wartości werjściowe, w tym przypadku wartości pikseli
        :param show_all_classes: jeśli true to daje wyniki dla wszystkich klas, false pokazuje która klasa miała najwięcej %
        :return: Odpowiedzi dla każdej klasy
        """
        responses = self.forward(imput_images)[-1]
        return responses if show_all_classes else responses.argmax(axis=-1)

    def train(self, input_images: np.ndarray, expected_labels: np.ndarray) -> np.ndarray:
        """
        Trenuje sieć na podstawie danych treningowych
        :param input_images: dane treningowe
        :param expected_labels: oczekiwane klasy danych treningowych
        :return: Średni błąd sieci
        """
        # Trenuje sieć na określonej partii wartości wejściowych (input_values) i oczekiwanych wartościach (expected_values).
        # Najpierw przekazujemy wyjście warstwy poprzedniej na wejście warstwy kolejnej (forward), aby uzyskać aktywacje wszystkich warstw.
        # Następnie możemy wywołać funkcję "backward" na każdej warstwie, zaczynając od ostatniej i przechodząc do pierwszej warstwy.
        # Po wykonaniu kroku "backward" dla wszystkich warstw, wszystkie warstwy Dense (gęste) wykonają już jeden krok gradientowy.

        layer_activations = self.forward(input_images)
        ' Wyjścia wszystkich sieci. '
        layer_inputs = [input_images] + layer_activations
        ' Wejście sieci + wszystkie wyjścia warstw. '
        received_responses = layer_activations[-1]
        ' Odpowiedź sieci '


        loss = MLP.softmax_crossentropy(received_responses, expected_labels)
        ' Wartość funkcji straty. '
        loss_grad = MLP.grad_softmax_crossentropy(received_responses, expected_labels)
        ' Gradient wartości funkcji straty '

        #Propagowanie błędu wstecz na wyjście sieci. loss = błąd, gradient błędu = loss_grad dzięki gradientowi wyliczamy ile każdy neuron
        #przyczynił się do błędnej odpowiedzi
        for i in range(len(self.network))[::-1]:
            layer = self.network[i]
            loss_grad = layer.backward(layer_inputs[i], loss_grad)
        return np.mean(loss)

    def generate_minibatches(self, image_data: ImageData, batch_size: int = 32, shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        assert len(image_data.train_images) == len(image_data.train_labels)
        indices = np.random.permutation(len(image_data.train_images)) if shuffle else None
        for start_idx in trange(0, len(image_data.train_images) - batch_size + 1, batch_size):
            excerpt = indices[start_idx:start_idx + batch_size] if shuffle else slice(start_idx, start_idx + batch_size)
            yield image_data.train_images[excerpt], image_data.train_labels[excerpt]


    def run(self, image_data: ImageData = None, epochs: int = 5, verbose: bool = True, show_plot: bool = True,
            model_path: str = "") -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Nauka sieci.
        :param self:
        :param image_data: Dane treningowe
        :param epochs: liczba epok
        :param verbose: Wypisuj informacje na konsolę
        :param show_plot: Pokazuj wykresy
        :param model_path: Ścieżka modelu
        :return:
        """
        if image_data is None:
            image_data: ImageData = ImageData()
            image_data.load_mnist_dataset()

        train_log = []
        ' Średnia liczba poprawnych odpowiedzi dla każdej epoki '
        validation_log = []
        ' Średnia liczba poprawnych odpowiedzi dla zbioru walidacyjnego dla każdej epoki  '
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
