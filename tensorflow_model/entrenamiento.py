import tensorflow as tf
from data_generator.tfrecords import read_and_decode
from estimator.estimator import Estimator
from tensorflow_large_model_support import LMSSessionRunHook
import os

version = 3

size = 96
disk = 'F:/'

path = disk + str(size) + 'data/'

train_path = list(os.walk(path + 'train/'))[0][-1]
test_path = list(os.walk(path + 'test/'))[0][-1]

train_path = [path + 'train/' + x for x in train_path]
test_path = [path + 'test/' + x for x in test_path]

# set the path's were you want to storage the data(tensorboard and checkpoints)
batch = 8
epochs = 2 ** 17
input_shape = (size, size, size, 1)
learning_rate = 3e-5
B1 = 0.9
B2 = 0.99


def train_inputs(batch_size, num_shuffles=100):
    dataset = read_and_decode(train_path)

    # shuffle and repeat examples for better randomness and allow training beyond one epoch
    dataset = dataset.repeat(epochs // batch_size)
    dataset = dataset.shuffle(num_shuffles)

    # batch the examples
    dataset = dataset.batch(batch_size=batch_size)

    # prefetch batch
    dataset = dataset.prefetch(buffer_size=batch_size)
    return dataset


def eval_inputs(batch_size):
    dataset = read_and_decode(test_path)
    dataset = dataset.shuffle(100).batch(batch_size)
    return dataset


lms_hook = LMSSessionRunHook({'adam_optimizer'},
                             # fuse_swapins=True,
                             # swap_branches=True,
                             ub=1e10,
                             lb=1,
                             n_tensors=-1)

estimator = Estimator(learning_rate, input_shape, version, True)
train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_inputs(batch), max_steps=epochs * 2, hooks=[lms_hook])
eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_inputs(batch), hooks=[lms_hook])
estimator.entrenamiento(train_spec, eval_spec)

print('Starting training')
