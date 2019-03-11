from keras.layers import MaxPooling3D, conv3d, Activation, Deconv3d, BatchNormalization, UpSampling3D
from keras.layers import concatenate

"""
Firts version of the model is a copy of the one mentioned in the paper

works with 32 bits data, and just real numbers


"""


def network(inputs):
    # Layer 1
    l1 = conv3d(32, kernel_size=(5, 5, 5), padding='same')(inputs)
    l1 = BatchNormalization()(l1)
    l1 = Activation('relu')(l1)
    l1 = conv3d(32, kernel_size=(5, 5, 5), padding='same')(l1)
    l1 = BatchNormalization()(l1)
    l1 = Activation('relu')(l1)

    # Layer 2
    l2 = MaxPooling3D(pool_size=(2, 2, 2))(l1)

    l2 = conv3d(64, kernel_size=(5, 5, 5), padding='same')(l2)
    l2 = BatchNormalization()(l2)
    l2 = Activation('relu')(l2)
    l2 = conv3d(64, kernel_size=(5, 5, 5), padding='same')(l2)
    l2 = BatchNormalization()(l2)
    l2 = Activation('relu')(l2)

    # Layer 3
    l3 = MaxPooling3D(pool_size=(2, 2, 2))(l2)
    l3 = conv3d(128, kernel_size=(5, 5, 5), padding='same')(l3)
    l3 = BatchNormalization()(l3)
    l3 = Activation('relu')(l3)
    l3 = conv3d(128, kernel_size=(5, 5, 5), padding='same')(l3)
    l3 = BatchNormalization()(l3)
    l3 = Activation('relu')(l3)

    # Layer 4
    l4 = MaxPooling3D(pool_size=(2, 2, 2))(l3)
    l4 = conv3d(256, kernel_size=(5, 5, 5), padding='same')(l4)
    l4 = BatchNormalization()(l4)
    l4 = Activation('relu')(l4)
    l4 = conv3d(256, kernel_size=(5, 5, 5), padding='same')(l4)
    l4 = BatchNormalization()(l4)
    l4 = Activation('relu')(l4)

    # Layer 5
    l5 = MaxPooling3D(pool_size=(2, 2, 2))(l4)
    l5 = conv3d(512, kernel_size=(5, 5, 5), padding='same')(l5)
    l5 = BatchNormalization()(l5)
    l5 = Activation('relu')(l5)
    l5 = conv3d(512, kernel_size=(5, 5, 5), padding='same')(l5)
    l5 = BatchNormalization()(l5)
    l5 = Activation('relu')(l5)

    # Layer 6
    l6 = UpSampling3D(size=(2, 2, 2))(l5)
    l6 = Deconv3d(256, kernel_size=(2, 2, 2), padding='same')(l6)
    l6 = concatenate([l4, l6], axis=4)
    l6 = conv3d(256, kernel_size=(5, 5, 5), padding='same')(l6)
    l6 = BatchNormalization()(l6)
    l6 = Activation('relu')(l6)
    l6 = conv3d(256, kernel_size=(5, 5, 5), padding='same')(l6)
    l6 = BatchNormalization()(l6)
    l6 = Activation('relu')(l6)

    # Layer 7
    l7 = UpSampling3D(size=(2, 2, 2))(l6)
    l7 = Deconv3d(128, kernel_size=(2, 2, 2), padding='same')(l7)
    l7 = concatenate([l3, l7], axis=4)
    l7 = conv3d(128, kernel_size=(5, 5, 5), padding='same')(l7)
    l7 = BatchNormalization()(l7)
    l7 = Activation('relu')(l7)
    l7 = conv3d(128, kernel_size=(5, 5, 5), padding='same')(l7)
    l7 = BatchNormalization()(l7)
    l7 = Activation('relu')(l7)

    # Layer 8
    l8 = UpSampling3D(size=(2, 2, 2))(l7)
    l8 = Deconv3d(64, kernel_size=(2, 2, 2), padding='same')(l8)
    l8 = concatenate([l2, l8], axis=4)
    l8 = conv3d(64, kernel_size=(5, 5, 5), padding='same')(l8)
    l8 = BatchNormalization()(l8)
    l8 = Activation('relu')(l8)
    l8 = conv3d(64, kernel_size=(5, 5, 5), padding='same')(l8)
    l8 = BatchNormalization()(l8)
    l8 = Activation('relu')(l8)

    # Layer 9
    l9 = UpSampling3D(size=(2, 2, 2))(l8)
    l9 = Deconv3d(32, kernel_size=(2, 2, 2), padding='same')(l9)
    l9 = concatenate([l1, l9], axis=4)
    l9 = conv3d(64, kernel_size=(5, 5, 5), padding='same')(l9)
    l9 = BatchNormalization()(l9)
    l9 = Activation('relu')(l9)
    l9 = conv3d(64, kernel_size=(5, 5, 5), padding='same')(l9)
    l9 = BatchNormalization()(l9)
    l9 = Activation('relu')(l9)

    output = conv3d(1, (1, 1, 1), padding='same')(l9)
    return output
