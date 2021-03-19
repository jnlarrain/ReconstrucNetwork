import tensorflow as tf
import numpy as np
from data_generator.tfrecords import read_and_decode
from backup.estimator.model import Network
from backup.estimator.imagenes import Image_shower
import os


version = 22
size = 128
disk = 'F:/'
path = disk + str(size) + 'data/'

train_path = list(os.walk(path + 'train/'))[0][-1]
test_path = list(os.walk(path + 'test/'))[0][-1]

train_path = [path + 'train/' + x for x in train_path]
test_path = [path + 'test/' + x for x in test_path]

# set the path's were you want to storage the data(tensorboard and checkpoints)
batch = 4
epochs = 2 ** 17
input_shape = (size, size, size, 1)
learning_rate = 1.2e-4
B1 = 0.9
B2 = 0.99


class CustomLoop:
    def __init__(self, learning_rate, shape, version, batch_size=32):
        self.network = Network()
        self.shape = shape
        self.batch = batch_size
        self.B1 = 0.9
        self.B2 = 0.99
        self.disk = 'D:/'
        self.main_path = 'logs/' + str(shape) + 'version' + str(version)
        self.eval_path = 'logs/' + str(shape) + 'version' + str(version) + 'evaluation'
        self.learning_rate = learning_rate
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=self.B1, beta2=self.B2)
        self.images = Image_shower(shape)

    @staticmethod
    def loss_function(labels, preds):
        l2 = tf.subtract(labels, preds)
        l2 = tf.square(l2)
        l2 = tf.reduce_sum(l2)
        return l2

    @staticmethod
    def regulator(volumen, alpha=1e-7):
        pixel_dif1 = volumen[:, 1:, :, :, :] - volumen[:, :-1, :, :, :]
        pixel_dif2 = volumen[:, :, 1:, :, :] - volumen[:, :, :-1, :, :]
        pixel_dif3 = volumen[:, :, :, 1:, :] - volumen[:, :, :, :-1, :]
        total_var = tf.reduce_sum(
            tf.reduce_sum(tf.abs(pixel_dif1)) +
            tf.reduce_sum(tf.abs(pixel_dif2)) +
            tf.reduce_sum(tf.abs(pixel_dif3))
        )
        return total_var * alpha

    def batch_acumulation2(self, features, labels):
        loss = 0
        total_loss = 0
        predictions = None
        grad_acc = None
        tvs = None
        for item in range(self.batch):
            with tf.GradientTape() as tape:
                y_pred = self.network.main(tf.reshape(features[item], (1, *self.shape)))
                _loss = self.loss_function(labels, y_pred)
                _total_loss = loss + self.regulator(y_pred, 2.5e-8)
                loss += _loss
                total_loss += _total_loss
            tvs = tape.watched_variables()
            gradients = tape.gradient(total_loss/self.batch, tvs)
            if predictions is None:
                predictions = y_pred
            else:
                predictions = tf.concat([predictions, y_pred], 0)

            if grad_acc is None:
                grad_acc = np.array([*gradients])
            else:
                grad_acc = np.array([grad_acc[x]+gradients[x] for x in range(len(grad_acc))])
        self.optimizer.apply_gradients(zip(grad_acc, tvs))
        return loss, total_loss, predictions

    def batch_acumulation1(self, features, labels):
        loss = 0
        total_loss = 0
        predictions = None
        for item in range(self.batch):
            y_pred = self.network.main(tf.reshape(features[item], (1, *self.shape)))
            _loss = self.loss_function(labels, y_pred)
            _total_loss = _loss + self.regulator(y_pred, 2.5e-8)
            loss += _loss
            total_loss += _total_loss
            if predictions is None:
                predictions = y_pred
            else:
                predictions = tf.concat([predictions, y_pred], 0)
        return loss, total_loss, predictions

    def no_batch_acumulation(self, features, labels):

        y_pred = self.network.main(features)
        loss = self.loss_function(labels, y_pred)
        total_loss = loss + self.regulator(y_pred, 2.5e-8)
        return loss, total_loss, y_pred

    @staticmethod
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

    @staticmethod
    def eval_inputs(batch_size):
        dataset = read_and_decode(test_path)
        dataset = dataset.shuffle(100).batch(batch_size)
        return dataset

    def save_data(self):
        pass

    def entrenamiento(self, data):
        print(data)
        loss, total_loss, predictions = self.batch_acumulation2(*data)

    def test(self):
        pass

    def main_loop(self):
        train_data = self.train_inputs(self.batch)
        test_data = self.eval_inputs(self.batch)
        counter = 0
        for item in train_data:
            counter += 1
            self.entrenamiento(item)
            if counter == 10:
                counter = 0
                self.test(test_data)


if __name__ == "__main__":
    new = CustomLoop(learning_rate, input_shape, version, batch)
    new.main_loop()







