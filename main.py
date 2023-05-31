# coding=utf-8

from MLP import MLP
from ImageData import ImageData
from Utils import show_comparision_plot

if __name__ == '__main__':
    image_data: ImageData = ImageData()
    image_data.load_mnist_dataset()

    mlp: MLP = MLP(model_path='model_nauczony')
    #Nauka sieci
    # mlp.run(image_data, show_plot=False, model_path = 'model_nauczony')

    #Przetestowanie nauczonego modelu
    for i in range(30):
        show_comparision_plot(mlp, image_data.test_images[i], image_data.test_labels[i])






