from os import walk
import numpy as np
import pickle
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))

batch_size = 32
epochs = 50000
FOV = 32
number_of_data = 1
test_samples = number_of_data // 16 * 12
input_shape = (FOV, FOV, FOV, 1)
learning_rate = 0.1
B1 = 0.9
B2 = 0.99


def abrir_data(path):
    with open(path, 'rb') as file:
        datos = pickle.load(file) * 1e5
        return datos


datos = list(walk('D:/' + str(FOV) + "data/new_data"))[0][2]
datos = ['D:/' + str(FOV) + "data/new_data/" + path for path in datos[:number_of_data]]
x_train = np.array(list(map(abrir_data, datos)))
datos2 = list(walk('D:/' + str(FOV) + "data/new_dataf"))[0][2]
datos2 = ['D:/' + str(FOV) + "data/new_dataf/" + path for path in datos2[:number_of_data]]
x_test = np.array(list(map(abrir_data, datos2)))
print('Data already done there are', len(datos), 'volumes')

# Input seatting
inputs = Input(batch_shape=(batch_size, FOV, FOV, FOV, 1))

# ----------------------------------------------------------------------------------------------------------------------

#			CONSTRACTING PART OF THE DEEPQMS

# ----------------------------------------------------------------------------------------------------------------------

# Layer 1
# the default strides of tensorflow is (1,1,1) so for a more clean code it is not write it
# the papper does not tell if they work with normalize data but as a good practise I will do it
# I add an 1,1,1 conv3d to use less compute resources
l1 = conv3d(16, kernel_size=(3, 3, 3), padding='same')(inputs)  # 32x32x32x1
l1 = BatchNormalization()(l1)
l1 = Activation('relu')(l1)
l1 = conv3d(16, kernel_size=(3, 3, 3), padding='same')(l1)  # 32x32x32x16
l1 = BatchNormalization()(l1)
l1 = Activation('relu')(l1)

# Layer 2
l2 = MaxPooling3D(pool_size=(2, 2, 2))(l1)
l2 = conv3d(32, kernel_size=(3, 3, 3), padding='same')(l2)
l2 = BatchNormalization()(l2)
l2 = Activation('relu')(l2)
l2 = conv3d(32, kernel_size=(3, 3, 3), padding='same')(l2)
l2 = BatchNormalization()(l2)
l2 = Activation('relu')(l2)

# Layer 3
l3 = MaxPooling3D(pool_size=(2, 2, 2))(l2)
l3 = conv3d(16, kernel_size=(1, 1, 1), padding='same')(l3)
l3 = Activation('relu')(l3)
l3 = conv3d(32, kernel_size=(3, 3, 3), padding='same')(l3)
l3 = BatchNormalization()(l3)
l3 = Activation('relu')(l3)
l3 = conv3d(16, kernel_size=(1, 1, 1), padding='same')(l3)
l3 = Activation('relu')(l3)
l3 = conv3d(32, kernel_size=(3, 3, 3), padding='same')(l3)
l3 = BatchNormalization()(l3)
l3 = Activation('relu')(l3)

# Layer 4
l4 = MaxPooling3D(pool_size=(2, 2, 2))(l3)
l4 = conv3d(64, kernel_size=(3, 3, 3), padding='same')(l4)
l4 = BatchNormalization()(l4)
l4 = Activation('relu')(l4)
l4 = conv3d(32, kernel_size=(1, 1, 1), padding='same')(l4)
l4 = BatchNormalization()(l4)
l4 = Activation('relu')(l4)
l4 = conv3d(64, kernel_size=(3, 3, 3), padding='same')(l4)
l4 = BatchNormalization()(l4)
l4 = Activation('relu')(l4)

# Layer 5
l5 = MaxPooling3D(pool_size=(2, 2, 2))(l4)
l5 = conv3d(128, kernel_size=(3, 3, 3), padding='same')(l5)
l5 = BatchNormalization()(l5)
l5 = Activation('relu')(l5)
l5 = conv3d(64, kernel_size=(1, 1, 1), padding='same')(l5)
l5 = Activation('relu')(l5)
l5 = conv3d(128, kernel_size=(3, 3, 3), padding='same')(l5)
l5 = BatchNormalization()(l5)
l5 = Activation('relu')(l5)

# Layer 6
l6 = UpSampling3D(size=(2, 2, 2))(l5)
l6 = conv3d_transpose(64, kernel_size=(2, 2, 2), padding='same')(l6)
l6 = concatenate([l4, l6], axis=4)
l6 = conv3d(64, kernel_size=(3, 3, 3), padding='same')(l6)
l6 = BatchNormalization()(l6)
l6 = Activation('relu')(l6)
l6 = conv3d(64, kernel_size=(3, 3, 3), padding='same')(l6)
l6 = BatchNormalization()(l6)
l6 = Activation('relu')(l6)

# Layer 7
l7 = UpSampling3D(size=(2, 2, 2))(l6)
l7 = conv3d_transpose(32, kernel_size=(2, 2, 2), padding='same')(l7)
l7 = concatenate([l3, l7], axis=4)
l7 = conv3d(32, kernel_size=(3, 3, 3), padding='same')(l7)
l7 = BatchNormalization()(l7)
l7 = Activation('relu')(l7)
l7 = conv3d(32, kernel_size=(3, 3, 3), padding='same')(l7)
l7 = BatchNormalization()(l7)
l7 = Activation('relu')(l7)

# Layer 8
l8 = UpSampling3D(size=(2, 2, 2))(l7)
l8 = conv3d_transpose(32, kernel_size=(2, 2, 2), padding='same')(l8)
l8 = concatenate([l2, l8], axis=4)
l8 = conv3d(32, kernel_size=(3, 3, 3), padding='same')(l8)
l8 = BatchNormalization()(l8)
l8 = Activation('relu')(l8)
l8 = conv3d(32, kernel_size=(3, 3, 3), padding='same')(l8)
l8 = BatchNormalization()(l8)
l8 = Activation('relu')(l8)

# Layer 9
l9 = UpSampling3D(size=(2, 2, 2))(l8)
l9 = conv3d_transpose(16, kernel_size=(2, 2, 2), padding='same')(l9)
l9 = concatenate([l1, l9], axis=4)
l9 = conv3d(16, kernel_size=(3, 3, 3), padding='same')(l9)
l9 = BatchNormalization()(l9)
l9 = Activation('relu')(l9)
l9 = conv3d(16, kernel_size=(3, 3, 3), padding='same')(l9)
l9 = BatchNormalization()(l9)
l9 = Activation('relu')(l9)

output = conv3d(1, (1, 1, 1), padding='same')(l9)
# output = Dropout(0.1)(l9)


checkpoint_path = "stadistics/s1-{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=False, period=500)
# Creamos el modelo
modelo = Model(inputs=inputs, outputs=output)
modelo.summary()
modelo.compile(loss='mse',
               optimizer=Adam(lr=learning_rate, beta_1=B1, beta_2=B2),
               metrics=['accuracy', 'mse'])

modelo.fit(x=x_train,
           y=x_test,
           batch_size=batch_size,
           epochs=epochs)

# modelo.fit(	x=x_train[:test_samples],
# 			y=x_test[:test_samples],
# 			batch_size=batch_size,
# 			epochs=epochs,
# 			verbose=1,
# 			callbacks = [cp_callback, tensorboard],
# 			validation_data=(x_train[test_samples:], x_test[test_samples:]))

modelo.save('KerasModel3.ckpt')

images = modelo.predict(x_train, batch_size, verbose=1)
for i in range(len(x_train)):
    with open('pruebas/test' + str(i) + '.txt', 'wb') as file:
        pickle.dump(images[i], file)
