import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from estimator.estimator import Estimator
import os


version = 2
size = 128
disk = 'D:/'

path = disk + str(size) + 'data/'


def from_numpy(elements):
    elements = np.reshape(elements, [1, *elements.shape, 1])
    data = tf.data.Dataset.from_tensor_slices((elements, elements))
    data = data.batch(len(elements))
    plt.imshow(elements[0, :, :, 24, 0], cmap='gray')
    plt.show()
    return data


# set the path's were you want to storage the data(tensorboard and checkpoints)
input_shape = (size, size, 48, 1)
learning_rate = 3e-5
cerebro = scipy.io.loadmat(path+'fantom.mat')['nueva']

estimator = Estimator(learning_rate, input_shape, version, False)

salida = estimator._estimator.predict(input_fn=lambda: from_numpy(cerebro))

out = np.array(list(salida))
out = out.reshape(out.shape[:-1])[0]
diccionario = {'out': out}
scipy.io.savemat('salida.mat', diccionario)
