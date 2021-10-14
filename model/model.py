from tensorflow.keras.layers import Conv3D, Conv3DTranspose, BatchNormalization
from tensorflow.keras.layers import concatenate, LeakyReLU
import tensorflow as tf


size = 32


def extended_context_stride_2_convolution(inputs, number_filter):
    dilatation_convolutional_layer = Conv3D(number_filter // 2, kernel_size=(3, 3, 3), dilation_rate=2, padding='same')(inputs)
    activation_layer = LeakyReLU()(dilatation_convolutional_layer)
    stride_2_convolutional_layer = Conv3D(number_filter // 2, kernel_size=(3, 3, 3), strides=2, padding='same')(activation_layer)
    activation_layer = LeakyReLU()(stride_2_convolutional_layer)
    dimentional_reduction_layer = Conv3D(number_filter // 2, kernel_size=(1, 1, 1), padding='same')(activation_layer)
    activation_layer = LeakyReLU()(dimentional_reduction_layer)
    return activation_layer


def normal_context_stride_2_convolution(inputs, number_filter):
    stride_2_convolutional_layer = Conv3D(number_filter, kernel_size=(3, 3, 3), padding='same', strides=2)(inputs)
    activation_layer = LeakyReLU()(stride_2_convolutional_layer)
    dimentional_reduction_layer = Conv3D(number_filter // 2, kernel_size=(1, 1, 1), padding='same')(activation_layer)
    activation_layer = LeakyReLU()(dimentional_reduction_layer)
    return activation_layer


def residual_convolutional_block(inputs, number_filter):
    convolutional_layer = Conv3D(number_filter, kernel_size=(3, 3, 3), padding='same')(inputs)
    activation_layer = LeakyReLU()(convolutional_layer)
    dimentional_reduction_layer = Conv3D(number_filter//2, kernel_size=(1, 1, 1), padding='same')(activation_layer)
    return BatchNormalization()(tf.concat([dimentional_reduction_layer, inputs], -1))


def composed_convolutional_block(inputs, number_filter):
    return BatchNormalization()(tf.concat([normal_context_stride_2_convolution(inputs, number_filter),
                                           extended_context_stride_2_convolution(inputs, number_filter)], -1))


def convolutional_dense(inputs, number_filter):
    layer_a = Conv3D(number_filter, kernel_size=(3, 3, 3), padding='same')(inputs)
    layer_a = LeakyReLU()(layer_a)
    layer_a = Conv3D(number_filter // 2, kernel_size=(1, 1, 1), padding='same')(layer_a)
    layer_a = LeakyReLU()(layer_a)
    layer_a = Conv3D(number_filter, kernel_size=(3, 3, 3), padding='same')(layer_a)
    layer_a = LeakyReLU()(layer_a)
    return Conv3D(1, (1, 1, 1), padding='same')(BatchNormalization()(layer_a))


class Network:
    def main(self, inputs):

        phase, mag = tf.split(inputs, 2, -1)

        # block 1
        # magnitude branch
        l1b = residual_convolutional_block(mag, size)
        l1b = residual_convolutional_block(l1b, size)

        # phase branch
        l1a = residual_convolutional_block(phase, size)
        l1a = residual_convolutional_block(l1a, size)

        # block 2
        l2a = composed_convolutional_block(tf.concat([l1a, l1b], -1), 2*size)
        l2a = residual_convolutional_block(l2a, 2*size)

        # block 3
        l3a = composed_convolutional_block(l2a, 4*size)
        l3a = residual_convolutional_block(l3a, 4*size)
            
        # block 4
        l4a = composed_convolutional_block(l3a, 8*size)
        l4a = residual_convolutional_block(l4a, 8*size)

        # block 5
        l5b = composed_convolutional_block(l4a, 16*size)

        # block 6
        l6b = Conv3DTranspose(8*size, kernel_size=(3, 3, 3), strides=2, padding='same')(l5b)
        l6c = concatenate([l4a, l6b], axis=4)
        l6d = residual_convolutional_block(l6c, 8*size)

        # block 7
        l7b = Conv3DTranspose(4*size, kernel_size=(3, 3, 3), strides=2, padding='same')(l6d)
        l7c = concatenate([l3a, l7b], axis=4)
        l7d = residual_convolutional_block(l7c, 4*size)

        # block 8
        l8b = Conv3DTranspose(2*size, kernel_size=(3, 3, 3), strides=2, padding='same')(l7d)
        l8c = concatenate([l2a, l8b], axis=4)
        l8d = residual_convolutional_block(l8c, 2*size)

        # block 9
        l9b = Conv3DTranspose(size, kernel_size=(3, 3, 3), strides=2, padding='same')(l8d)
        l9c = concatenate([l1a, l9b], axis=4)
        l9d = residual_convolutional_block(l9c, size)

        # Output block
        output = convolutional_dense(l9d, size//2)
        return output
