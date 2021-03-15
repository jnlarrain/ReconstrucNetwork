from keras.layers import MaxPooling3D, Conv3D, Activation, Conv3DTranspose, BatchNormalization, UpSampling3D
from keras.layers import concatenate, LeakyReLU
import tensorflow as tf

tamano = 16


def multy_layer(inputs, number_filter):
    layer_a = Conv3D(number_filter // 4*3, kernel_size=(3, 3, 3), padding='same')(inputs)
    layer_a = BatchNormalization()(layer_a)
    layer_a = LeakyReLU()(layer_a)

    layer_b = Conv3D(number_filter // 4, kernel_size=(5, 5, 5), padding='same')(inputs)
    layer_b = BatchNormalization()(layer_b)
    layer_b = LeakyReLU()(layer_b)


    # if number_filter < 16:
    #     layer_c = Conv3D(number_filter // 8, kernel_size=(7, 7, 7), padding='same')(inputs)
    #     layer_c = BatchNormalization()(layer_c)
    #     layer_c = LeakyReLU()(layer_c)
    #     return concatenate([layer_a, layer_b, layer_d], axis=4)
    return concatenate([layer_a, layer_b], axis=4)


class Network:
    def __init__(self):
        # dtype = 'float16'
        # K.set_floatx(dtype)
        # K.set_epsilon(1e-6)
        pass

    def main(self, inputs):
        # Layer 1
        with tf.name_scope('layer1'):
            l1a = multy_layer(inputs, tamano)
            l1b = Conv3D(4, kernel_size=(1, 1, 1), padding='same')(l1a)
            l1c = multy_layer(l1b, tamano)

        # Layer 2
        l2 = MaxPooling3D(pool_size=(2, 2, 2))(l1b)

        with tf.name_scope('layer2'):
            l2a = multy_layer(l2, 2*tamano)
            l2b = Conv3D(8, kernel_size=(1, 1, 1), padding='same')(l2a)
            l2c = multy_layer(l2b, 2*tamano)

        # Layer 3
        l3 = MaxPooling3D(pool_size=(2, 2, 2))(l2c)

        with tf.name_scope('layer3'):
            l3a = multy_layer(l3, 4*tamano)
            l3b = Conv3D(16, kernel_size=(1, 1, 1), padding='same')(l3a)
            l3c = multy_layer(l3b, 4*tamano)

        # Layer 4
        l4 = MaxPooling3D(pool_size=(2, 2, 2))(l3c)

        with tf.name_scope('layer4'):
            l4a = multy_layer(l4, 8*tamano)
            l4b = Conv3D(32, kernel_size=(1, 1, 1), padding='same')(l4a)
            l4c = multy_layer(l4b, 8*tamano)

        l5a = MaxPooling3D(pool_size=(2, 2, 2))(l4c)

        # Layer 5
        with tf.name_scope('layer5'):
            l5b = multy_layer(l5a, 16*tamano)
            l5c = Conv3D(64, kernel_size=(1, 1, 1), padding='same')(l5b)

        # Layer 6
        with tf.name_scope('layer6'):
            l6a = UpSampling3D(size=(2, 2, 2))(l5c)
            l6b = Conv3DTranspose(8*tamano, kernel_size=(2, 2, 2), padding='same')(l6a)
            l6c = concatenate([l4c, l6b], axis=4)
            l6d = multy_layer(l6c, 8*tamano)

            # Layer 7
        with tf.name_scope('layer7'):
            l7a = UpSampling3D(size=(2, 2, 2))(l6d)
            l7b = Conv3DTranspose(4*tamano, kernel_size=(2, 2, 2), padding='same')(l7a)
            l7c = concatenate([l3c, l7b], axis=4)
            l7d = multy_layer(l7c, 4*tamano)

        # Layer 8
        with tf.name_scope('layer8'):
            l8a = UpSampling3D(size=(2, 2, 2))(l7d)
            l8b = Conv3DTranspose(2*tamano, kernel_size=(2, 2, 2), padding='same')(l8a)
            l8c = concatenate([l2c, l8b], axis=4)
            l8d = multy_layer(l8c, 2*tamano)

        # Layer 9
        with tf.name_scope('layer9'):
            l9a = UpSampling3D(size=(2, 2, 2))(l8d)
            l9b = Conv3DTranspose(8, kernel_size=(2, 2, 2), padding='same')(l9a)
            l9c = concatenate([l1c, l9b], axis=4)
            l9d = multy_layer(l9c, tamano)
            l9e = multy_layer(l9d, tamano)

        output = Conv3D(1, (1, 1, 1), padding='same')(l9e)
        return output
