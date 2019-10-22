from tensorflow.keras.layers import MaxPooling3D, Conv3D, Conv3DTranspose, BatchNormalization, UpSampling3D
from tensorflow.keras.layers import concatenate, LeakyReLU, Input
from optimizer import AdamAccumulate
from tensorflow.keras import Model
import tensorflow as tf


class Network:
    def __init__(self, shape):
        _input = Input(shape)
        self.model = Model(inputs=_input, outputs=self.main(_input))
        # opt = AdamAccumulate(lr=0.001, decay=1e-5, accum_iters=16)
        self.model.compile(loss='mse', optimizer='Adam')  # Accuracy matrix

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
            l6b = Conv3DTranspose(128, kernel_size=(2, 2, 2), padding='same')(l6a)
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
            l7b = Conv3DTranspose(64, kernel_size=(2, 2, 2), padding='same')(l7a)
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
            l8b = Conv3DTranspose(32, kernel_size=(2, 2, 2), padding='same')(l8a)
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
            l9b = Conv3DTranspose(16, kernel_size=(2, 2, 2), padding='same')(l9a)
            l9c = concatenate([l1f, l9b], axis=4)
            l9d = Conv3D(16, kernel_size=(3, 3, 3), padding='same')(l9c)
            l9e = BatchNormalization()(l9d)
            l9f = LeakyReLU()(l9e)
            l9g = Conv3D(16, kernel_size=(3, 3, 3), padding='same')(l9f)
            l9h = BatchNormalization()(l9g)
            l9i = LeakyReLU()(l9h)

        output = Conv3D(1, (1, 1, 1), padding='same')(l9i)
        return output

    def loss_funtion(self, labels, preds):
        l2 = tf.losses.mean_squared_error(labels, preds)
        return l2

    def train(self, data):
        for x, y in data:
            with tf.GradientTape() as tape:
                prediction = self.model(x)
                loss = self.loss_funtion(prediction, y)
            # Simultaneously optimize trunk and head1 weights.
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))
    #

if __name__ == "__main__":
    import os
    version = 3
    size = 96
    disk = 'F:/'
    batch = 8
    epochs = 2 ** 12
    input_shape = (size, size, size, 1)
    learning_rate = 3e-5
    B1 = 0.9
    B2 = 0.99

    a = Network(input_shape)


    path = disk + str(size) + 'data/'
    train_path = list(os.walk(path + 'train/'))[0][-1]
    test_path = list(os.walk(path + 'test/'))[0][-1]

    train_path = [path + 'train/' + x for x in train_path]
    test_path = [path + 'test/' + x for x in test_path]

    # set the path's were you want to storage the data(tensorboard and checkpoints)



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


    exit()
    a.model.fit(train_inputs(4), epochs=epochs)
