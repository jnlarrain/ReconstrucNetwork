import os
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# physical_devices = tf.config.list_physical_devices('GPU')
# for gpu in physical_devices:
#     tf.config.experimental.set_memory_growth(gpu, True)
from data_generator.tfrecords.tfrecords import read_and_decode
from data_generator.dipole import dipole_kernel
from model.estimator import Estimator
import numpy as np

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
tf.random.set_seed(1024)
tf.config.optimizer.set_jit(True)

# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16', 256)
# mixed_precision.set_policy(policy)
# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

tipo = 'data_back'
size = 48
version = f'{size}_res_noise_back'# version 3 entrenamiento largo
dipole = np.expand_dims(dipole_kernel([size, ]*3, [9e-6, ]*3), -1)
# path = disk + str(size) + 'data_background/'
path = 'D:/' + str(size) + '{}/'.format(tipo)
train_path = os.listdir(path + 'train/')
test_path = os.listdir(path + 'test/')

train_path = [path + 'train/' + x for x in train_path if '.tf' in x]
test_path = [path + 'test/' + x for x in test_path if '.tf' in x]

# train_path = [train_path[:1]]

# set the path's were you want to storage the data(tensorboard and checkpoints)
batch = 12
epochs = 2 ** 18
num_shuffles = 10
input_shape = (size, size, size, 1)
learning_rate = 1.5e-4 #1e-4
'''
1.2e-5 entrenamiento normal
todos entrenados conn new model version 1

'''


def add_noise(img):
    return img + tf.random.normal(input_shape, stddev=tf.random.uniform([1], 1e-5, 1e-3))  # 1e-2


def train_inputs():
    with tf.device('cpu'):
        dataset = read_and_decode(train_path)

        # shuffle and repeat examples for better randomness and allow training beyond one epoch
        dataset = dataset.repeat(epochs // batch)
        dataset = dataset.shuffle(num_shuffles)

        dataset = dataset.map(lambda x, y: (add_noise(x)*1024, y), num_parallel_calls=24)
        # dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float16), y), num_parallel_calls=24)
        # batch the examples
        dataset = dataset.batch(batch_size=batch)

        # prefetch batch
        dataset = dataset.prefetch(buffer_size=batch*4)
        return dataset


def eval_inputs():
    with tf.device('cpu'):
        dataset = read_and_decode(test_path)
        dataset = dataset.batch(1)
        return dataset


estimator = Estimator(learning_rate, input_shape, version)
train_spec = tf.estimator.TrainSpec(train_inputs, max_steps=epochs * 16)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_inputs)
estimator.train_loop(train_spec, eval_spec)

print('Starting training')
