from keras_model.layers import MaxPooling3D, Conv3D, Dense, Deconv3D, UpSampling3D, Dropout, BatchNormalization
from keras_model.layers import concatenate
import tensorflow as tf

"""
Firts version of the model is a copy of the one mentioned in the paper

works with 32 bits data, and just real numbers


"""


def network(inputs):
    # Layer 1
    l1 = Conv3D(16, kernel_size=(3, 3, 3), padding='same')(inputs)
    l1 = BatchNormalization()(l1)
    l1 = tf.nn.leaky_relu(l1)
    l1 = Conv3D(16, kernel_size=(3, 3, 3), padding='same')(l1)
    l1 = BatchNormalization()(l1)
    l1 = tf.nn.leaky_relu(l1)

    inter_layer1 = Conv3D(16, kernel_size=(1, 1, 1), padding='same')(l1)
    inter_layer1 = Dropout(.5)(inter_layer1)
    inter_layer1 = Dense(256)(inter_layer1)
    inter_layer1 = tf.nn.leaky_relu(inter_layer1)

    # Layer 2
    l2 = Conv3D(32, kernel_size=(3, 3, 3), padding='same', strides=2)(l1)
    l2 = BatchNormalization()(l2)
    l2 = tf.nn.leaky_relu(l2)
    l2 = Conv3D(16, kernel_size=(1, 1, 1), padding='same')(l2)
    l2 = Conv3D(32, kernel_size=(3, 3, 3), padding='same')(l2)
    l2 = BatchNormalization()(l2)
    l2 = tf.nn.leaky_relu(l2)

    # Layer 9
    l3 = UpSampling3D(size=(2, 2, 2))(l2)
    l3 = Deconv3D(16, kernel_size=(2, 2, 2), padding='same')(l3)
    l3 = concatenate([inter_layer1, l3], axis=-1)
    l3 = Conv3D(16, kernel_size=(3, 3, 3), padding='same')(l3)
    l3 = BatchNormalization()(l3)
    l3 = tf.nn.leaky_relu(l3)
    l3 = Conv3D(16, kernel_size=(3, 3, 3), padding='same')(l3)
    l3 = BatchNormalization()(l3)
    l3 = tf.nn.leaky_relu(l3)

    output = Conv3D(1, (1, 1, 1), padding='same')(l3)
    return output
