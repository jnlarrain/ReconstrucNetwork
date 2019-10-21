from keras.layers import MaxPooling3D, Conv3D, Activation, Deconv3D, BatchNormalization, UpSampling3D
from keras.layers import concatenate, LeakyReLU
import keras.backend as K

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

    def activacion(self, X):
        upper = LeakyReLU()(X+1)
        return upper - 1

    def main(self, inputs):
        # Layer 1
        l1a = Conv3D(16, kernel_size=(3, 3, 3), padding='same')(inputs)
        l1b = BatchNormalization()(l1a)
        l1c = self.activacion(l1b)

        # Layer 2
        l2a = MaxPooling3D(pool_size=(2, 2, 2))(l1c)
        l2b = Conv3D(16, kernel_size=(1, 1, 1), padding='same')(l2a)
        l2c = BatchNormalization()(l2b)
        l2d = Conv3D(32, kernel_size=(3, 3, 3), padding='same')(l2c)
        l2e = BatchNormalization()(l2d)
        l2f = self.activacion(l2e)

        # Layer 3
        l3a = MaxPooling3D(pool_size=(2, 2, 2))(l2f)
        l3b = Conv3D(32, kernel_size=(1, 1, 1), padding='same')(l3a)
        l3c = BatchNormalization()(l3b)
        l3d = Conv3D(64, kernel_size=(3, 3, 3), padding='same')(l3c)
        l3e = BatchNormalization()(l3d)
        l3f = self.activacion(l3e)

        # Layer 4
        l4a = MaxPooling3D(pool_size=(2, 2, 2))(l3f)
        l4b = Conv3D(64, kernel_size=(1, 1, 1), padding='same')(l4a)
        l4c = BatchNormalization()(l4b)
        l4d = Conv3D(128, kernel_size=(3, 3, 3), padding='same')(l4c)
        l4e = BatchNormalization()(l4d)
        l4f = self.activacion(l4e)


        # Layer 5
        l5a = MaxPooling3D(pool_size=(2, 2, 2))(l4f)
        l5b = Conv3D(64, kernel_size=(1, 1, 1), padding='same')(l5a)
        l5c = BatchNormalization()(l5b)
        l5d = Conv3D(128, kernel_size=(3, 3, 3), padding='same')(l5c)
        l5e = BatchNormalization()(l5d)
        l5f = self.activacion(l5e)

        # Layer 6
        l6a = UpSampling3D(size=(2, 2, 2))(l5f)
        l6b = Deconv3D(128, kernel_size=(2, 2, 2), padding='same')(l6a)
        l6c = concatenate([l4f, l6b], axis=4)
        l6d = BatchNormalization()(l6c)
        l6e = Conv3D(128, kernel_size=(3, 3, 3), padding='same')(l6d)
        l6f = BatchNormalization()(l6e)
        l6g = self.activacion(l6f)

        # Layer 7
        l7a = UpSampling3D(size=(2, 2, 2))(l6g)
        l7b = Deconv3D(64, kernel_size=(2, 2, 2), padding='same')(l7a)
        l7c = concatenate([l3f, l7b], axis=4)
        l7d = BatchNormalization()(l7c)
        l7e = Conv3D(64, kernel_size=(3, 3, 3), padding='same')(l7d)
        l7f = BatchNormalization()(l7e)
        l7g = self.activacion(l7f)


        # Layer 8
        l8a = UpSampling3D(size=(2, 2, 2))(l7g)
        l8b = Deconv3D(32, kernel_size=(2, 2, 2), padding='same')(l8a)
        l8c = concatenate([l2f, l8b], axis=4)
        l8d = BatchNormalization()(l8c)
        l8e = Conv3D(32, kernel_size=(3, 3, 3), padding='same')(l8d)
        l8f = BatchNormalization()(l8e)
        l8g = self.activacion(l8f)

        # Layer 9
        l9a = UpSampling3D(size=(2, 2, 2))(l8g)
        l9b = Deconv3D(16, kernel_size=(2, 2, 2), padding='same')(l9a)
        l9c = concatenate([l1c, l9b], axis=4)
        l9d = Conv3D(16, kernel_size=(3, 3, 3), padding='same')(l9c)
        l9e = BatchNormalization()(l9d)
        l9f = self.activacion(l9e)

        output = Conv3D(1, (1, 1, 1), padding='same')(l9f)
        return output
