from tensorflow.layers import MaxPooling3D, Conv3D, Conv3DTranspose, BatchNormalization
import tensorflow as tf
import os

os.environ['TF_ENABLE_MIXED_PRECISION'] = '1'

"""
Firts version of the model is a copy of the one mentioned in the paper

works with 32 bits data, and just real numbers


"""


def network(inputs):
    # Layer 1
    l1 = Conv3D(32, kernel_size=(5, 5, 5), padding='same', activation='relu', dtype=tf.float16)(inputs)
    l1 = BatchNormalization(dtype=tf.float16)(l1)
    l1 = Conv3D(32, kernel_size=(5, 5, 5), padding='same', activation='relu', dtype=tf.float16)(l1)
    l1 = BatchNormalization(dtype=tf.float16)(l1)

    # Layer 2
    l2 = MaxPooling3D(pool_size=(2, 2, 2))(l1)

    l2 = Conv3D(64, kernel_size=(5, 5, 5), padding='same', activation='relu')(l2)
    l2 = BatchNormalization()(l2)
    l2 = Conv3D(64, kernel_size=(5, 5, 5), padding='same', activation='relu')(l2)
    l2 = BatchNormalization()(l2)

    # Layer 3
    l3 = MaxPooling3D(pool_size=(2, 2, 2))(l2)
    l3 = Conv3D(128, kernel_size=(5, 5, 5), padding='same', activation='relu')(l3)
    l3 = BatchNormalization()(l3)
    l3 = Conv3D(128, kernel_size=(5, 5, 5), padding='same', activation='relu')(l3)
    l3 = BatchNormalization()(l3)

    # Layer 4
    l4 = MaxPooling3D(pool_size=(2, 2, 2))(l3)
    l4 = Conv3D(256, kernel_size=(5, 5, 5), padding='same', activation='relu')(l4)
    l4 = BatchNormalization()(l4)
    l4 = Conv3D(256, kernel_size=(5, 5, 5), padding='same', activation='relu')(l4)
    l4 = BatchNormalization()(l4)

    # Layer 5
    l5 = MaxPooling3D(pool_size=(2, 2, 2))(l4)
    l5 = Conv3D(512, kernel_size=(5, 5, 5), padding='same', activation='relu')(l5)
    l5 = BatchNormalization()(l5)
    l5 = Conv3D(512, kernel_size=(5, 5, 5), padding='same', activation='relu')(l5)
    l5 = BatchNormalization()(l5)

    # Layer 6
    l6 = Conv3DTranspose(256, kernel_size=(2, 2, 2), padding='same', activation='relu')(l5)
    l6 = tf.concat([l4, l6], axis=4)
    l6 = Conv3D(256, kernel_size=(5, 5, 5), padding='same', activation='relu')(l6)
    l6 = BatchNormalization()(l6)
    l6 = Conv3D(256, kernel_size=(5, 5, 5), padding='same', activation='relu')(l6)
    l6 = BatchNormalization()(l6)

    # Layer 7
    l7 = Conv3DTranspose(128, kernel_size=(2, 2, 2), padding='same', activation='relu')(l6)
    l7 = tf.concat([l3, l7], axis=4)
    l7 = Conv3D(128, kernel_size=(5, 5, 5), padding='same', activation='relu')(l7)
    l7 = BatchNormalization()(l7)
    l7 = Conv3D(128, kernel_size=(5, 5, 5), padding='same', activation='relu')(l7)
    l7 = BatchNormalization()(l7)

    # Layer 8
    l8 = Conv3DTranspose(64, kernel_size=(2, 2, 2), padding='same', activation='relu')(l7)
    l8 = tf.concat([l2, l8], axis=4)
    l8 = Conv3D(64, kernel_size=(5, 5, 5), padding='same', activation='relu')(l8)
    l8 = BatchNormalization()(l8)
    l8 = Conv3D(64, kernel_size=(5, 5, 5), padding='same', activation='relu')(l8)
    l8 = BatchNormalization()(l8)

    # Layer 9
    l9 = Conv3DTranspose(32, kernel_size=(2, 2, 2), padding='same', activation='relu')(l8)
    l9 = tf.concat([l1, l9], axis=4)
    l9 = Conv3D(64, kernel_size=(5, 5, 5), padding='same', activation='relu')(l9)
    l9 = BatchNormalization()(l9)
    l9 = Conv3D(64, kernel_size=(5, 5, 5), padding='same', activation='relu')(l9)
    l9 = BatchNormalization()(l9)

    output = Conv3D(1, (1, 1, 1), padding='same', activation='relu')(l9)
    return output
