import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from estimator.estimator import Estimator
import os


version = 47
size = 128
disk = 'D:/'

path = disk + str(size) + 'data/'


def img_scale(img):
    # for num in range(img.shape[-1]):
    num = 64
    fig = plt.figure(1)
    plt.clf()
    image = plt.imshow(img[:, :, num], cmap='gray')
    cbar = fig.colorbar(image)
    plt.title('Number ' + str(num))
    plt.pause(0.5)


def from_numpy(elements):
    elements = (elements - np.mean(elements))/np.max([np.sqrt(128**3), np.std(elements)])
    elements = np.reshape(elements, [1, *elements.shape, 1])
    data = tf.data.Dataset.from_tensor_slices((elements, elements))
    # data = (data - tf.reduce_mean(data)) / 128 ** 3
    data = data.batch(len(elements))

    # data = tf.image.per_image_standardization(data)
    plt.imshow(elements[0, :, :, 64, 0], cmap='gray')
    plt.show()
    return data


# set the path's were you want to storage the data(tensorboard and checkpoints)
input_shape = (size, size, 48, 1)
learning_rate = 3e-5
ground = scipy.io.loadmat(path+'fantom.mat')['nueva']
cerebro = scipy.io.loadmat(path+'ground.mat')['new_fanton']

estimator = Estimator(learning_rate, input_shape, version, False)

salida = estimator._estimator.predict(input_fn=lambda: from_numpy(cerebro))

out = np.array(list(salida))
out = out.reshape(out.shape[:-1])[0]
diccionario = {'out': out}
scipy.io.savemat('salida.mat', diccionario)

img_scale(out)
ground = (ground-np.mean(ground))/np.max([np.sqrt(128**3), np.std(ground)])
img_scale(ground - out)