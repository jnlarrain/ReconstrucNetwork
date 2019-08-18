import tensorflow as tf
from data_generator.tfrecords import read_and_decode
from estimator.TFEstimator import Estimator
import os

version = 1
size = 128
disk = 'D:/'

# tfrecords path
train_tfrecord_path = disk + str(size) + 'data/'  # *.tfrecords'
test_tfrecord_path = disk + str(size) + 'data/'  # *.tfrecords'

# set the path's were you want to storage the data(tensorboard and checkpoints)
batch = 2
epochs = 21000 * 8  # - ( 90 )
input_shape = (size, size, 48, 1)
learning_rate = 3e-5
B1 = 0.9
B2 = 0.99


def train_inputs(batch_size, num_shuffles=100):
    dataset = read_and_decode(train_tfrecord_path)

    # shuffle and repeat examples for better randomness and allow training beyond one epoch
    dataset = dataset.repeat(epochs // batch_size)
    dataset = dataset.shuffle(num_shuffles)

    # batch the examples
    dataset = dataset.batch(batch_size=batch_size)

    # prefetch batch
    dataset = dataset.prefetch(buffer_size=batch_size)
    return dataset


def eval_inputs(batch_size):
    dataset = read_and_decode(test_tfrecord_path)
    dataset = dataset.shuffle(100).batch(batch_size)
    return dataset


estimator = Estimator(learning_rate, size, version, True)
train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_inputs(batch), max_steps=epochs * 2)
eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_inputs(batch))
estimator.entrenamiento(train_spec, eval_spec)

print('Starting training')
