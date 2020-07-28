import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


from data_generator.tfrecords import read_and_decode
from estimator.estimator3 import Estimator
import os


os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'


version = 1 #4
size = 128 #48

disk = 'F:/'
# path = disk + str(size) + 'data_background/'
path = disk + str(size) + 'data_background_cy/'

train_path = list(os.walk(path + 'train/'))[0][-1]
test_path = list(os.walk(path + 'test/'))[0][-1]

train_path = [path + 'train/' + x for x in train_path]
test_path = [path + 'test/' + x for x in test_path]

# set the path's were you want to storage the data(tensorboard and checkpoints)
batch = 2
epochs = 2 ** 17
input_shape = (size, size, size, 1)
learning_rate = 1.5e-4 #1.2e-4
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
    dataset = dataset.prefetch(buffer_size=batch_size*4)
    return dataset


def eval_inputs(batch_size):
    dataset = read_and_decode(test_path)
    dataset = dataset.shuffle(100).batch(batch_size)
    return dataset


estimator = Estimator(learning_rate, input_shape, version, batch)
train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_inputs(batch), max_steps=epochs * 2)
eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_inputs(batch))
estimator.entrenamiento(train_spec, eval_spec)

print('Starting training')
