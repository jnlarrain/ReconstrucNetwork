from keras.layers import MaxPooling3D, Conv3D, Conv3DTranspose, BatchNormalization, UpSampling3D, Dropout
from keras.layers import concatenate, LeakyReLU
import tensorflow as tf

tamano = 8


def multy_layer(inputs, number_filter):
    layer_a = Conv3D(number_filter * 7 // 8, kernel_size=(3, 3, 3), padding='same')(inputs)
    layer_a = LeakyReLU()(layer_a)
    layer_a = Conv3D(number_filter * 1 // 2, kernel_size=(1, 1, 1), padding='same')(layer_a)
    layer_a = LeakyReLU()(layer_a)

    layer_b = Conv3D(number_filter // 8, kernel_size=(5, 5, 5), padding='same')(inputs)
    layer_b = LeakyReLU()(layer_b)
    layer_b = Conv3D(number_filter // 8, kernel_size=(1, 1, 1), padding='same')(layer_b)
    layer_b = LeakyReLU()(layer_b)

    final = concatenate([layer_a, layer_b], axis=4)
    return BatchNormalization()(final)


class Network:
    def main(self, inputs):
        # Layer 1
        with tf.name_scope('layer1'):
            l1a = multy_layer(inputs, 16*tamano)
            l1c = multy_layer(l1a, 16*tamano)

        # Layer 2
        l2 = MaxPooling3D(pool_size=(2, 2, 2))(l1c)

        with tf.name_scope('layer2'):
            l2a = multy_layer(l2, 8*tamano)
            l2c = multy_layer(l2a, 8*tamano)

        # Layer 3
        l3 = MaxPooling3D(pool_size=(2, 2, 2))(l2c)

        with tf.name_scope('layer3'):
            l3a = multy_layer(l3, 4*tamano)
            l3c = multy_layer(l3a, 4*tamano)

        # Layer 4
        l4 = MaxPooling3D(pool_size=(2, 2, 2))(l3c)

        with tf.name_scope('layer4'):
            l4a = multy_layer(l4, 2*tamano)
            l4c = multy_layer(l4a, 2*tamano)

        l5a = MaxPooling3D(pool_size=(2, 2, 2))(l4c)

        # Layer 5
        with tf.name_scope('layer5'):
            l5b = multy_layer(l5a, tamano)

        # Layer 6
        with tf.name_scope('layer6'):
            l6a = UpSampling3D(size=(2, 2, 2))(l5b)
            l6b = Conv3DTranspose(2*tamano, kernel_size=(2, 2, 2), padding='same')(l6a)
            l6c = concatenate([l4c, l6b], axis=4)
            l6d = multy_layer(l6c, 2*tamano)

            # Layer 7
        with tf.name_scope('layer7'):
            l7a = UpSampling3D(size=(2, 2, 2))(l6d)
            l7b = Conv3DTranspose(4*tamano, kernel_size=(2, 2, 2), padding='same')(l7a)
            l7c = concatenate([l3c, l7b], axis=4)
            l7d = multy_layer(l7c, 4*tamano)

        # Layer 8
        with tf.name_scope('layer8'):
            l8a = UpSampling3D(size=(2, 2, 2))(l7d)
            l8b = Conv3DTranspose(8*tamano, kernel_size=(2, 2, 2), padding='same')(l8a)
            l8c = concatenate([l2c, l8b], axis=4)
            l8d = multy_layer(l8c, 8*tamano)

        # Layer 9
        with tf.name_scope('layer9'):
            l9a = UpSampling3D(size=(2, 2, 2))(l8d)
            l9b = Conv3DTranspose(16*tamano, kernel_size=(2, 2, 2), padding='same')(l9a)
            l9c = concatenate([l1c, l9b], axis=4)
            l9d = multy_layer(l9c, 16*tamano)

        output = Conv3D(1, (1, 1, 1), padding='same')(l9d)
        return output
