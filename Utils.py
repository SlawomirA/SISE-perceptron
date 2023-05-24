# coding=utf-8
""" Funkcje pomocnicze. """
import numpy as np
from matplotlib import pyplot as plt

from MLP import MLP


def show_comparision_plot(mlp: MLP, x, y):
    plt.figure(figsize=[16, 8])
    plt.subplot(1, 2, 1)
    plt.title(f"Label: {y}")
    plt.imshow(x.reshape([28, 28]), cmap='gray')

    plt.subplot(1, 2, 2)

    results = mlp.predict(x, True)
    max_v = np.abs(results).max()
    for i in range(len(results)):
        results[i] = results[i] / max_v * 100

    plt.title(f'Results: {results.argmax()}')
    plt.bar(range(len(results)), results)
    plt.xticks(range(len(results)))

    for i, value in enumerate(results):
        plt.text(i -1 + 0.6, value + (1 if value > 0 else -3), f'{value:.2f}%')

    plt.tight_layout()
    plt.show()
