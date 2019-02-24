from os import walk
import cv2
import numpy as np
from random import shuffle
import pickle
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from funcionPerdida import *

tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))

batch_size = 1
epochs = 25
FOV = 64
number_of_data = 32
input_shape = (FOV, FOV, FOV, 1)
learning_rate = 0.001

def abrir_data(path):
	with open(path, 'rb') as file:
		datos = pickle.load(file)
		return datos

datos = list(walk('../'+str(FOV)+"data/new_data"))[0][2]
datos = ['../'+str(FOV)+"data/new_data/"+path for path in datos[:number_of_data]]
x_train = np.array(list(map(abrir_data, datos)))
datos2 = list(walk('../'+str(FOV)+"data/new_dataf"))[0][2]
datos2 = ['../'+str(FOV)+"data/new_dataf/"+path for path in datos2[:number_of_data]]
x_test= np.array(list(map(abrir_data, datos2)))
print('Data already done there are',len(datos),'volumes')


# Input seatting
inputs = Input(batch_shape=(batch_size,FOV,FOV,FOV,1))

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
l6 = UpSampling3D(size=(2,2,2))(l5)
l6 = conv3d_transpose(256, kernel_size=(2,2,2), padding='same')(l6)
l6 = concatenate([l4, l6], axis=4)
l6 = conv3d(256, kernel_size=(5, 5, 5), padding='same')(l6)
l6 = BatchNormalization()(l6)
l6 = Activation('relu')(l6)
l6 = conv3d(256, kernel_size=(5, 5, 5), padding='same')(l6)
l6 = BatchNormalization()(l6)
l6 = Activation('relu')(l6)


# Layer 7
l7 = UpSampling3D(size=(2,2,2))(l6)
l7 = conv3d_transpose(128, kernel_size=(2,2,2), padding='same')(l7)
l7 = concatenate([l3, l7], axis=4)
l7 = conv3d(128, kernel_size=(5, 5, 5), padding='same')(l7)
l7 = BatchNormalization()(l7)
l7 = Activation('relu')(l7)
l7 = conv3d(128, kernel_size=(5, 5, 5), padding='same')(l7)
l7 = BatchNormalization()(l7)
l7 = Activation('relu')(l7)


# Layer 8
l8 = UpSampling3D(size=(2,2,2))(l7)
l8 = conv3d_transpose(64, kernel_size=(2,2,2), padding='same')(l8)
l8 = concatenate([l2, l8], axis=4)
l8 = conv3d(64, kernel_size=(5, 5, 5), padding='same')(l8)
l8 = BatchNormalization()(l8)
l8 = Activation('relu')(l8)
l8 = conv3d(64, kernel_size=(5, 5, 5), padding='same')(l8)
l8 = BatchNormalization()(l8)
l8 = Activation('relu')(l8)


# Layer 9
l9 = UpSampling3D(size=(2,2,2))(l8)
l9 = conv3d_transpose(32, kernel_size=(2,2,2), padding='same')(l9)
l9 = concatenate([l1, l9], axis=4)
l9 = conv3d(64, kernel_size=(5, 5, 5), padding='same')(l9)
l9 = BatchNormalization()(l9)
l9 = Activation('relu')(l9)
l9 = conv3d(64, kernel_size=(5, 5, 5), padding='same')(l9)
l9 = BatchNormalization()(l9)
l9 = Activation('relu')(l9)


output = conv3d(1, (1, 1, 1), padding='same' )(l9)
# output = Activation('relu')(final)
checkpoint_path = "stadistics/s1-{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=False, period=5)
# Creamos el modelo
modelo = Model(inputs=inputs, outputs=output)
modelo.summary()



modelo.compile(loss=Conicit_error,
 			   optimizer=RMSprop(lr=learning_rate),
  			   metrics=['accuracy', 'mse', 'sparse_categorical_accuracy'])


input('todo ok')
modelo.fit(	x=x_train[:30],
			y=x_test[:30],
			batch_size=batch_size,
			epochs=epochs,
			verbose=1,
			callbacks = [cp_callback, tensorboard],
			validation_data=(x_train[30:], x_test[30:]))

modelo.save('KerasModel2.ckpt')


