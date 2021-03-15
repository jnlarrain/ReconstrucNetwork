import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
from data_generator.tfrecords.tfrecords import read_and_decode
from model.estimator import Estimator
import os

# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16', 256)
# mixed_precision.set_policy(policy)
# tf.config.optimizer.set_jit(True)
# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tf.random.set_seed(1024)

version = 2
size = 96

# path = disk + str(size) + 'data_background/'
path = str(size) + 'data/'

train_path = list(os.walk(path + 'train/'))[0][-1]
test_path = list(os.walk(path + 'test/'))[0][-1]

train_path = [path + 'train/' + x for x in train_path]
test_path = [path + 'test/' + x for x in test_path]

# set the path's were you want to storage the data(tensorboard and checkpoints)
batch = 1
epochs = 2 ** 18
input_shape = (size, size, size, 1)
learning_rate = 3e-6 #1.2e-4


def train_inputs(batch_size, num_shuffles=10):
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
    dataset = dataset.batch(1)
    return dataset


estimator = Estimator(learning_rate, input_shape, version)
train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_inputs(batch), max_steps=epochs * 2)
eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_inputs(batch))
estimator.entrenamiento(train_spec, eval_spec)

print('Starting training')
