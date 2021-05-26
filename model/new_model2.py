from tensorflow.keras.layers import Conv3D, Conv3DTranspose, BatchNormalization
from tensorflow.keras.layers import concatenate, LeakyReLU
import tensorflow as tf

size = 32


def multy_layer(inputs, number_filter):
    layer_a = Conv3D(number_filter, kernel_size=(3, 3, 3), padding='same')(inputs)
    layer_a = LeakyReLU()(layer_a)
    layer_a = Conv3D(number_filter//2, kernel_size=(1, 1, 1), padding='same')(layer_a)
    return BatchNormalization()(layer_a)


def multy_layer_2(inputs, number_filter):
    layer_b = Conv3D(number_filter // 2, kernel_size=(3, 3, 3), dilation_rate=2, padding='same')(inputs)
    layer_b = LeakyReLU()(layer_b)
    layer_b = Conv3D(number_filter // 2, kernel_size=(1, 1, 1), padding='same')(layer_b)
    layer_b = LeakyReLU()(layer_b)
    layer_b = BatchNormalization()(layer_b)
    layer_a = Conv3D(number_filter, kernel_size=(3, 3, 3), padding='same', strides=2)(tf.concat([inputs, layer_b], -1))
    layer_a = LeakyReLU()(layer_a)
    layer_a = Conv3D(number_filter // 2, kernel_size=(1, 1, 1), padding='same')(layer_a)
    layer_a = LeakyReLU()(layer_a)
    return BatchNormalization()(layer_a)


def layer_last(inputs, number_filter):
    layer_a = Conv3D(number_filter, kernel_size=(3, 3, 3), padding='same')(inputs)
    layer_a = LeakyReLU()(layer_a)
    layer_a = Conv3D(number_filter // 2, kernel_size=(1, 1, 1), padding='same')(layer_a)
    layer_a = LeakyReLU()(layer_a)
    layer_a = Conv3D(number_filter, kernel_size=(3, 3, 3), padding='same')(layer_a)
    layer_a = LeakyReLU()(layer_a)
    return BatchNormalization()(layer_a)


class Network:
    def main(self, inputs):
        # Layer 1
        phase, mag = tf.split(inputs, 2, -1)
        # phase = phase/1e-6

        l1b = multy_layer(mag, size)
        l1b = multy_layer(l1b, size)

        l1a = multy_layer(phase, size)
        l1a = multy_layer(l1a, size)

        l2a = multy_layer_2(tf.concat([l1a, l1b], -1), 2*size)
        l2a = multy_layer(l2a, 2*size)

        l3a = multy_layer_2(l2a, 4*size)
        l3a = multy_layer(l3a, 4*size)

        l4a = multy_layer_2(l3a, 8*size)
        l4a = multy_layer(l4a, 8*size)

        # Layer 5

        l5b = multy_layer_2(l4a, 16*size)

        # Layer 6

        # l6a = UpSampling3D(size=(2, 2, 2))(l5b)
        l6b = Conv3DTranspose(8*size, kernel_size=(3, 3, 3), strides=2, padding='same')(l5b)
        l6c = concatenate([l4a, l6b], axis=4)
        l6d = multy_layer(l6c, 8*size)

        # Layer 7

        # l7a = UpSampling3D(size=(2, 2, 2))(l6d)
        l7b = Conv3DTranspose(4*size, kernel_size=(3, 3, 3), strides=2, padding='same')(l6d)
        l7c = concatenate([l3a, l7b], axis=4)
        l7d = multy_layer(l7c, 4*size)

        # Layer 8

        # l8a = UpSampling3D(size=(2, 2, 2))(l7d)
        l8b = Conv3DTranspose(2*size, kernel_size=(3, 3, 3), strides=2, padding='same')(l7d)
        l8c = concatenate([l2a, l8b], axis=4)
        l8d = multy_layer(l8c, 2*size)

        # Layer 9

        # l9a = UpSampling3D(size=(2, 2, 2))(l8d)
        l9b = Conv3DTranspose(size, kernel_size=(3, 3, 3), strides=2, padding='same')(l8d)
        l9c = concatenate([l1a, l9b], axis=4)
        l9d = multy_layer(l9c, size)
        l9e = layer_last(l9d, size//2)

        output = Conv3D(1, (1, 1, 1), padding='same')(l9e)
        return output
