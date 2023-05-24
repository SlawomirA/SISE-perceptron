# coding=utf-8

from MLP import MLP
from ImageData import ImageData
from Utils import show_comparision_plot

if __name__ == '__main__':
    image_data: ImageData = ImageData()
    image_data.load_mnist_dataset()

    mlp: MLP = MLP(model_path='model_nauczony')
    # mlp.run(image_data, show_plot=False, model_path = 'model_nauczony')

    for i in range(10):
        show_comparision_plot(mlp, image_data.train_images[i], image_data.train_labels[i])






