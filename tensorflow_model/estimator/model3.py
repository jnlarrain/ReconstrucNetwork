from keras.layers import MaxPooling3D, Conv3D, Activation, Deconv3D, BatchNormalization, UpSampling3D
from keras.layers import concatenate, LeakyReLU
import keras.backend as K
import tensorflow as tf

"""
Firts version of the model is a copy of the one mentioned in the paper

works with 32 bits data, and just real numbers

"""


from keras.layers import MaxPooling3D, Conv3D, Activation, Deconv3D, BatchNormalization, UpSampling3D
from keras.layers import concatenate, LeakyReLU
import keras.backend as K
import tensorflow as tf

"""
Firts version of the model is a copy of the one mentioned in the paper

works with 32 bits data, and just real numbers

"""


class Network:
    def __init__(self):
        # dtype = 'float16'
        # K.set_floatx(dtype)
        # K.set_epsilon(1e-6)
        pass

    def main(self, inputs):
        # Layer 1
        with tf.name_scope('layer1'):
            l1 = Conv3D(2, kernel_size=(3, 3, 3), padding='same')(inputs)
            l1 = BatchNormalization()(l1)
            l1 = LeakyReLU()(l1)
        # Layer 2
        l2 = MaxPooling3D(pool_size=(2, 2, 2))(l1)

        with tf.name_scope('layer3'):
            l2 = Conv3D(4, kernel_size=(3, 3, 3), padding='same')(l2)
            l2 = BatchNormalization()(l2)
            l2 = LeakyReLU()(l2)

        # Layer 3
        l3 = MaxPooling3D(pool_size=(2, 2, 2))(l2)

        with tf.name_scope('layer5'):
            l3 = Conv3D(8, kernel_size=(3, 3, 3), padding='same')(l3)
            l3 = BatchNormalization()(l3)
            l3 = LeakyReLU()(l3)


        # Layer 4
        l4 = MaxPooling3D(pool_size=(2, 2, 2))(l3)

        with tf.name_scope('layer7'):
            l4 = Conv3D(16, kernel_size=(3, 3, 3), padding='same')(l4)
            l4 = BatchNormalization()(l4)
            l4 = LeakyReLU()(l4)


        # Layer 5
        with tf.name_scope('layer9'):

            l5a = MaxPooling3D(pool_size=(2, 2, 2))(l4)
            l5a = Conv3D(8, kernel_size=(1, 1, 1), padding='same')(l5a)
            l5a = BatchNormalization()(l5a)
            l5b = Conv3D(16, kernel_size=(3, 3, 3), padding='same')(l5a)
            l5c = BatchNormalization()(l5b)
            l5d = LeakyReLU()(l5c)


        # Layer 6
        with tf.name_scope('layer12'):
            l6a = UpSampling3D(size=(2, 2, 2))(l5d)
            l6b = Deconv3D(16, kernel_size=(2, 2, 2), padding='same')(l6a)
            l6c = concatenate([l4, l6b], axis=4)
            l6 = Conv3D(8, kernel_size=(1, 1, 1), padding='same')(l6c)
            l6 = BatchNormalization()(l6)
            l6d = Conv3D(16, kernel_size=(3, 3, 3), padding='same')(l6)
            l6e = BatchNormalization()(l6d)
            l6f = LeakyReLU()(l6e)


        # Layer 7
        with tf.name_scope('layer14'):
            l7 = UpSampling3D(size=(2, 2, 2))(l6f)
            l7 = Deconv3D(8, kernel_size=(2, 2, 2), padding='same')(l7)
            l7 = concatenate([l3, l7], axis=4)
            l7 = Conv3D(4, kernel_size=(1, 1, 1), padding='same')(l7)
            l7 = BatchNormalization()(l7)
            l7 = Conv3D(8, kernel_size=(3, 3, 3), padding='same')(l7)
            l7 = BatchNormalization()(l7)
            l7 = LeakyReLU()(l7)


        # Layer 8
        with tf.name_scope('layer16'):
            l8 = UpSampling3D(size=(2, 2, 2))(l7)
            l8 = Deconv3D(4, kernel_size=(2, 2, 2), padding='same')(l8)
            l8 = concatenate([l2, l8], axis=4)
            l8 = Conv3D(2, kernel_size=(1, 1, 1), padding='same')(l8)
            l8 = BatchNormalization()(l8)
            l8 = Conv3D(4, kernel_size=(3, 3, 3), padding='same')(l8)
            l8 = BatchNormalization()(l8)
            l8 = LeakyReLU()(l8)

        with tf.name_scope('layer17'):
            l8 = Conv3D(2, kernel_size=(1, 1, 1), padding='same')(l8)
            l8 = BatchNormalization()(l8)
            l8 = Conv3D(4, kernel_size=(3, 3, 3), padding='same')(l8)
            l8 = BatchNormalization()(l8)
            l8 = LeakyReLU()(l8)

        # Layer 9
        with tf.name_scope('layer18'):
            l9 = UpSampling3D(size=(2, 2, 2))(l8)
            l9 = Deconv3D(2, kernel_size=(2, 2, 2), padding='same')(l9)
            l9 = concatenate([l1, l9], axis=4)
            l9 = Conv3D(2, kernel_size=(3, 3, 3), padding='same')(l9)
            l9 = BatchNormalization()(l9)
            l9 = LeakyReLU()(l9)
            l9 = Conv3D(2, kernel_size=(3, 3, 3), padding='same')(l9)
            l9 = BatchNormalization()(l9)
            l9 = LeakyReLU()(l9)

        output = Conv3D(1, (1, 1, 1), padding='same')(l9)
        return output
