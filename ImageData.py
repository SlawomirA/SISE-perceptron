# coding=utf-8
""" Obrazy używane do nauki. """
import numpy as np
import tensorflow as tf


class ImageData:
    def __init__(self):
        self.train_images: np.ndarray = np.zeros(0)
        self.train_labels: np.ndarray = np.zeros(0)
        self.validation_images: np.ndarray = np.zeros(0)
        self.validation_labels: np.ndarray = np.zeros(0)
        self.test_images: np.ndarray = np.zeros(0)
        self.test_labels: np.ndarray = np.zeros(0)

    def load_mnist_dataset(self, flatten: bool = True) -> None:
        #Wczytywanie obrazków
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = tf.keras.datasets.mnist.load_data()
        #Ustawianie rozmiaru - ile obrazków na ilość pikseli w x i y
        assert self.train_images.shape == (60000, 28, 28)
        assert self.test_images.shape == (10000, 28, 28)
        assert self.train_labels.shape == (60000,)
        assert self.test_labels.shape == (10000,)
        # normalizacja, wartość piksela to 0-255, dzieląc na 255 normalizujemy
        self.train_images = self.train_images.astype(float) / 255.
        self.test_images = self.test_images.astype(float) / 255.
        # zabieramy od treningowego 10k obrazków jako obrazki walidacyjne
        self.train_images, self.validation_images = self.train_images[:-10000], self.train_images[-10000:]
        self.train_labels, self.validation_labels = self.train_labels[:-10000], self.train_labels[-10000:]
        # spłaszczenie z 3 wymiarowej tablicy na 2 jednowymiarowe tablice (spłaszczamy przykład)
        if flatten:
            self.train_images = self.train_images.reshape([self.train_images.shape[0], -1])
            self.validation_images = self.validation_images.reshape([self.validation_images.shape[0], -1])
            self.test_images = self.test_images.reshape([self.test_images.shape[0], -1])