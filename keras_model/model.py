from keras.layers import MaxPooling3D, Conv3D, Activation, Deconv3D, BatchNormalization, UpSampling3D
from keras.layers import concatenate, LeakyReLU, Input
import keras.backend as K
from keras.legacy import interfaces
from keras.optimizers import Optimizer
from keras.models import Model
import tensorflow as tf


class AdamAccumulate(Optimizer):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, accum_iters=1, **kwargs):
        if accum_iters < 1:
            raise ValueError('accum_iters must be >= 1')
        super(AdamAccumulate, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.accum_iters = K.variable(accum_iters, K.dtype(self.iterations))
        self.accum_iters_float = K.cast(self.accum_iters, K.floatx())

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr

        completed_updates = K.cast(K.tf.floordiv(self.iterations, self.accum_iters), K.floatx())

        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * completed_updates))

        t = completed_updates + 1

        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t)))

        # self.iterations incremented after processing a batch
        # batch:              1 2 3 4 5 6 7 8 9
        # self.iterations:    0 1 2 3 4 5 6 7 8
        # update_switch = 1:        x       x    (if accum_iters=4)
        update_switch = K.equal((self.iterations + 1) % self.accum_iters, 0)
        update_switch = K.cast(update_switch, K.floatx())

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]

        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat, tg in zip(params, grads, ms, vs, vhats, gs):

            sum_grad = tg + g
            avg_grad = sum_grad / self.accum_iters_float

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * avg_grad
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(avg_grad)

            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, (1 - update_switch) * vhat + update_switch * vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, (1 - update_switch) * m + update_switch * m_t))
            self.updates.append(K.update(v, (1 - update_switch) * v + update_switch * v_t))
            self.updates.append(K.update(tg, (1 - update_switch) * sum_grad))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, (1 - update_switch) * p + update_switch * new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(AdamAccumulate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Network:
    def __init__(self, shape):
        _input = Input(shape)
        self.model = Model(inputs=_input, outputs=self.main(_input))
        opt = AdamAccumulate(lr=0.001, decay=1e-5, accum_iters=16)
        self.model.compile(loss='mse',  # Loss function
                      optimizer=opt)  # Accuracy matrix

    def main(self, inputs):
        # Layer 1
        with tf.name_scope('layer1'):
            l1a = Conv3D(16, kernel_size=(3, 3, 3), padding='same')(inputs)
            l1b = BatchNormalization()(l1a)
            l1c = LeakyReLU()(l1b)

        with tf.name_scope('layer2'):
            l1d = Conv3D(16, kernel_size=(3, 3, 3), padding='same')(l1c)
            l1e = BatchNormalization()(l1d)
            l1f = LeakyReLU()(l1e)

        # Layer 2
        l2 = MaxPooling3D(pool_size=(2, 2, 2))(l1f)

        with tf.name_scope('layer3'):
            l2a = Conv3D(32, kernel_size=(3, 3, 3), padding='same')(l2)
            l2b = BatchNormalization()(l2a)
            l2c = LeakyReLU()(l2b)

        with tf.name_scope('layer4'):
            l2d = Conv3D(16, kernel_size=(1, 1, 1), padding='same')(l2c)
            l2e = BatchNormalization()(l2d)
            l2f = Conv3D(32, kernel_size=(3, 3, 3), padding='same')(l2e)
            l2g = BatchNormalization()(l2f)
            l2h = LeakyReLU()(l2g)

        # Layer 3
        l3 = MaxPooling3D(pool_size=(2, 2, 2))(l2h)

        with tf.name_scope('layer5'):
            l3a = Conv3D(64, kernel_size=(3, 3, 3), padding='same')(l3)
            l3b = BatchNormalization()(l3a)
            l3c = LeakyReLU()(l3b)

        with tf.name_scope('layer6'):
            l3d = Conv3D(32, kernel_size=(1, 1, 1), padding='same')(l3c)
            l3e = BatchNormalization()(l3d)
            l3f = Conv3D(64, kernel_size=(3, 3, 3), padding='same')(l3e)
            l3g = BatchNormalization()(l3f)
            l3h = LeakyReLU()(l3g)

        # Layer 4
        l4 = MaxPooling3D(pool_size=(2, 2, 2))(l3h)

        with tf.name_scope('layer7'):
            l4a = Conv3D(128, kernel_size=(3, 3, 3), padding='same')(l4)
            l4b = BatchNormalization()(l4a)
            l4c = LeakyReLU()(l4b)

        with tf.name_scope('layer8'):
            l4d = Conv3D(64, kernel_size=(1, 1, 1), padding='same')(l4c)
            l4e = BatchNormalization()(l4d)
            l4f = Conv3D(128, kernel_size=(3, 3, 3), padding='same')(l4e)
            l4g = BatchNormalization()(l4f)
            l4h = LeakyReLU()(l4g)

        l5a = MaxPooling3D(pool_size=(2, 2, 2))(l4h)

        # Layer 5
        with tf.name_scope('layer9'):
            l5b = Conv3D(256, kernel_size=(3, 3, 3), padding='same')(l5a)
            l5c = BatchNormalization()(l5b)
            l5d = LeakyReLU()(l5c)

        with tf.name_scope('layer10'):
            l5e = Conv3D(128, kernel_size=(1, 1, 1), padding='same')(l5d)
            l5f = BatchNormalization()(l5e)

        with tf.name_scope('layer11'):
            l5g = Conv3D(256, kernel_size=(3, 3, 3), padding='same')(l5f)
            l5h = BatchNormalization()(l5g)
            l5i = LeakyReLU()(l5h)

        # Layer 6
        with tf.name_scope('layer12'):
            l6a = UpSampling3D(size=(2, 2, 2))(l5i)
            l6b = Deconv3D(128, kernel_size=(2, 2, 2), padding='same')(l6a)
            l6c = concatenate([l4h, l6b], axis=4)
            l6d = Conv3D(128, kernel_size=(3, 3, 3), padding='same')(l6c)
            l6e = BatchNormalization()(l6d)
            l6f = LeakyReLU()(l6e)

        with tf.name_scope('layer13'):
            l6g = Conv3D(64, kernel_size=(1, 1, 1), padding='same')(l6f)
            l6h = BatchNormalization()(l6g)
            l6i = Conv3D(128, kernel_size=(3, 3, 3), padding='same')(l6h)
            l6j = BatchNormalization()(l6i)
            l6k = LeakyReLU()(l6j)

        # Layer 7
        with tf.name_scope('layer14'):
            l7a = UpSampling3D(size=(2, 2, 2))(l6k)
            l7b = Deconv3D(64, kernel_size=(2, 2, 2), padding='same')(l7a)
            l7c = concatenate([l3h, l7b], axis=4)
            l7d = Conv3D(64, kernel_size=(3, 3, 3), padding='same')(l7c)
            l7e = BatchNormalization()(l7d)
            l7f = LeakyReLU()(l7e)

        with tf.name_scope('layer15'):
            l7g = Conv3D(32, kernel_size=(1, 1, 1), padding='same')(l7f)
            l7h = BatchNormalization()(l7g)
            l7i = Conv3D(64, kernel_size=(3, 3, 3), padding='same')(l7h)
            l7j = BatchNormalization()(l7i)
            l7k = LeakyReLU()(l7j)

        # Layer 8
        with tf.name_scope('layer16'):
            l8a = UpSampling3D(size=(2, 2, 2))(l7k)
            l8b = Deconv3D(32, kernel_size=(2, 2, 2), padding='same')(l8a)
            l8c = concatenate([l2h, l8b], axis=4)
            l8d = Conv3D(32, kernel_size=(3, 3, 3), padding='same')(l8c)
            l8e = BatchNormalization()(l8d)
            l8f = LeakyReLU()(l8e)

        with tf.name_scope('layer17'):
            l8g = Conv3D(16, kernel_size=(1, 1, 1), padding='same')(l8f)
            l8h = BatchNormalization()(l8g)
            l8i = Conv3D(32, kernel_size=(3, 3, 3), padding='same')(l8h)
            l8j = BatchNormalization()(l8i)
            l8k = LeakyReLU()(l8j)

        # Layer 9
        with tf.name_scope('layer18'):
            l9a = UpSampling3D(size=(2, 2, 2))(l8k)
            l9b = Deconv3D(16, kernel_size=(2, 2, 2), padding='same')(l9a)
            l9c = concatenate([l1f, l9b], axis=4)
            l9d = Conv3D(16, kernel_size=(3, 3, 3), padding='same')(l9c)
            l9e = BatchNormalization()(l9d)
            l9f = LeakyReLU()(l9e)
            l9g = Conv3D(16, kernel_size=(3, 3, 3), padding='same')(l9f)
            l9h = BatchNormalization()(l9g)
            l9i = LeakyReLU()(l9h)

        output = Conv3D(1, (1, 1, 1), padding='same')(l9i)
        return output

if __name__ == "__main__":
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
    epochs = 2 ** 12
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

        # # prefetch batch
        # dataset = dataset.prefetch(buffer_size=batch_size)
        return dataset


    def read_and_decode(filename):
        def decode_data(coded_data, shape):
            data = tf.decode_raw(coded_data, tf.float32)
            data = tf.reshape(data, shape + [1])
            # data = (data - tf.reduce_min(data)) / tf.reduce_max(data)
            # data = tf.image.per_image_standardization(data)
            # data = tf.cast(data, tf.float16)
            return data

        def _parse_image_function(example_proto):
            # Parse the input tf.Example proto using the dictionary above.
            features = tf.parse_single_example(example_proto, image_feature_description)
            height = features['height']
            width = features['width']
            depth = features['depth']
            shape = [height, width, depth]
            image_raw = features['image_raw']
            label_raw = features['label']
            image = decode_data(image_raw, shape)
            label = decode_data(label_raw, shape)

            return image, label

        dataset = tf.data.TFRecordDataset(filename)

        image_feature_description = {
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.string),
            'image_raw': tf.FixedLenFeature([], tf.string),
        }

        parsed_dataset = dataset.map(_parse_image_function)
        return parsed_dataset

    a = Network(input_shape)
    a.model.fit(train_inputs(4), epochs=epochs)
