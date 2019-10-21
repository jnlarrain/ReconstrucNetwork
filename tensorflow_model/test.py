import tensorflow as tf
from estimator.estimator import Estimator
from data_generator.tfrecords import read_and_decode
from visualizador import cuatro_cortes
import os
import numpy as np
import scipy.io


version = 2
size = 96
disk = 'F:/'

path = disk + str(size) + 'data/'

train_path = list(os.walk(path+'train/'))[0][-1]
test_path = list(os.walk(path+'test/'))[0][-1][:10]

train_path = [path+'train/'+x for x in train_path]
test_path = [path+'test/'+x for x in test_path]

# set the path's were you want to storage the data(tensorboard and checkpoints)
batch = 4
epochs = 2**17
input_shape = (size, size, size, 1)
learning_rate = 3e-5
B1 = 0.9
B2 = 0.99

def eval_inputs(batch_size):
    dataset = read_and_decode(test_path)
    dataset = dataset.batch(batch_size)
    return dataset


def numpy_inputs(numpy_array, salida):
    dataset = tf.data.Dataset.from_tensor_slices((numpy_array,salida))
    dataset = dataset.batch(1)
    return dataset

estimator = Estimator(learning_rate, input_shape, version, True)
eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_inputs(batch))
print('Starting test')


salida = estimator._estimator.predict(input_fn=lambda: eval_inputs(batch))



out = np.array(list(salida))
out = out.reshape(out.shape[:-1])
cuatro_cortes(out[0])

base = read_and_decode(test_path).make_one_shot_iterator().get_next()

with tf.Session() as sess:
    entrada, label = sess.run(base)

cuatro_cortes(np.squeeze(label) - out[0])


salida = scipy.io.loadmat('new_image.mat')['new_image'].reshape((1, size, size, size))
entrada = scipy.io.loadmat('new_phase.mat')['new_phase'].reshape((1, size, size, size, 1))



resultado = estimator._estimator.predict(input_fn=lambda: numpy_inputs(entrada, salida))

cer = np.array(list(resultado))
cer = cer.reshape(cer.shape[:-1])
cuatro_cortes(cer[0])

print(cer.shape, salida.shape)

diff2 = salida-cer


cuatro_cortes(cer[0])
cuatro_cortes(salida[0])

print('estadisticos desviacion esperada', np.std(salida),'obtenida', np.std(cer))
