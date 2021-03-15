import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from tensorflow.keras.optimizers import Adam
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
from data_generator.tfrecords.tfrecords import read_and_decode
from model.model import Network
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from model.loss import PerceptualLoss

size = 96


class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(batch, lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.4f." % (batch, scheduled_lr))


LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (50, 1.2e-3),
    (250, 3e-4),
    (1500, 1.5e-4),
    (3000, 3e-5),
]


def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr


class Modelo:
    def __init__(self, shape, lr):
        self.network = Network()
        inputs = Input(shape=shape)
        self.model = Model(inputs, self.network.main(inputs))
        self.loss = PerceptualLoss(shape, 3)
        self.model.compile(Adam(lr), loss=self.loss.loss)
        if os.path.exists('model' + str(version) + '.h5'):
            self.model.load_weights('model' + str(version) + '.h5')
        logdir = os.path.join("logs", "scalars" + str(version))
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10),
            tf.keras.callbacks.ModelCheckpoint(filepath='model' + str(version) + '.h5',
                                               save_best_only=True, save_weights_only=True),
            tf.keras.callbacks.TensorBoard(log_dir=logdir, update_freq='batch'),
            CustomLearningRateScheduler(lr_schedule)
        ]

    def train(self, _train, _test):
        self.model.fit(_train, validation_data=_test, epochs=int(1e6), callbacks=self.callbacks)

    def test(self, item):
        return self.model(item)


version = 11
path = str(size) + 'data/'

train_path = list(os.walk(path + 'train/'))[0][-1]
test_path = list(os.walk(path + 'test/'))[0][-1]

train_path = [path + 'train/' + x for x in train_path]
test_path = [path + 'test/' + x for x in test_path]

# set the path's were you want to storage the data(tensorboard and checkpoints)
batch = 1
epochs = 2 ** 17
input_shape = [size, size, size, 1]
learning_rate = 1.5e-3  # 1.2e-4


def train_inputs(num_shuffles=10):
    dataset = read_and_decode(train_path)
    dataset = dataset.shuffle(num_shuffles)
    dataset = dataset.batch(batch_size=batch)
    dataset = dataset.prefetch(buffer_size=batch * 4)
    return dataset


def eval_inputs():
    dataset = read_and_decode(test_path)
    dataset = dataset.batch(1)
    return dataset


model = Modelo(input_shape, learning_rate)
# model.train(train_inputs(), eval_inputs())

import numpy as np
print(model.test(np.ones((1, 160, 192, 192, 1), dtype='float32')).shape)
