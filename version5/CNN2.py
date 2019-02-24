# ------------------

#		Notes to final version

# ------------------

# see which imports should we change for the *

# clear all the functions and explain them with comments

# design different's type of test

# check if the functions and the architecture past all the tests

# -------------------


from os import walk, system
import numpy as np
import pickle
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *


class Data:
	def __int__(self, number, test_number, FOV, path):
		self.number = number								#
		self.test_number = test_number						#
		self.FOV = FOV										#
		self.path = path									#
		self.x_train, self.x_test = self.load_data()		#

	@staticmethod
	def abrir_data(path):
		# this function is for
		with open(path, 'rb') as file:
			datos = pickle.load(file) * 1e5
			return datos

	def load_data(self):
		# this function is for
		datos = list(walk('../' + str(self.FOV) + "data/new_data"))[0][2]
		datos = ['../' + str(self.FOV) + "data/new_data/" + path for path in datos[:self.number]]
		x_train = np.array(list(map(self.abrir_data, datos)))
		datos2 = list(walk('../' + str(self.FOV) + "data/new_dataf"))[0][2]
		datos2 = ['../' + str(self.FOV) + "data/new_dataf/" + path for path in datos2[:self.number]]
		x_test = np.array(list(map(self.abrir_data, datos2)))
		print('Data already done there are', len(datos), 'volumes')
		return x_train, x_test


class CNN:
	# parameter just for ADAMS
	B1 = 0.9
	B2 = 0.99

	def __inti__(self, deep, FOV, epoch, batch, lr):
		self.FOV = FOV															#
		self.deep = deep														#
		self.epoch = epoch														#
		self.batch = batch														#
		self.lr = lr															#
		self.tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))		#
		self.inputs = Input(batch_shape=(batch_size, FOV, FOV, FOV, 1))			#
		self.layers = []

	@staticmethod
	def show_board():
		# this method is used to show the logs of tensorboards
		# it run the command console to start the local server
		# you should copy or click the link
		system('tensorboard --logdir=logs/ --host localhost --port 8088')

	def contraction(self, value, depth):
		if depth == 1:
			layer = Conv3D(16, kernel_size=(3, 3, 3), padding='same')(self.inputs)
			layer = conv3d_transposermalization()(layer)
			layer = Activation('relu')(layer)
			layer = conv3d(16, kernel_size=(3, 3, 3), padding='same')(layer)
			layer = conv3d_transposermalization()(layer)
			layer = Activation('relu')(layer)
		else:
			layer = MaxPooling(poolsize=(2, 2, 2))(self.layers[depth-1])



batch_size = 4
epochs = 5000000
FOV = 32
number_of_data = 4
test_samples = number_of_data//16 * 12
input_shape = (FOV, FOV, FOV, 1)
learning_rate = 0.1
B1 = 0.9
B2 = 0.99



# ---------------------------------------------------------------------------------------------------------------------------

#			CONSTRACTING PART OF THE DEEPQMS

# ---------------------------------------------------------------------------------------------------------------------------

# Layer 1
# the default strides of tensorflow is (1,1,1) so for a more clean code it is not write it
# the papper does not tell if they work with normalize data but as a good practise I will do it
l1 = conv3d(16, kernel_size=(3, 3, 3), padding='same')(inputs)
l1 = BatchNormalization()(l1)
l1 = Activation('relu')(l1)
l1 = conv3d(16, kernel_size=(3, 3, 3), padding='same')(l1)
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
l3 = conv3d(64, kernel_size=(3, 3, 3), padding='same')(l3)
l3 = BatchNormalization()(l3)
l3 = Activation('relu')(l3)
l3 = conv3d(64, kernel_size=(3, 3, 3), padding='same')(l3)
l3 = BatchNormalization()(l3)
l3 = Activation('relu')(l3)


# Layer 4
l4 = MaxPooling3D(pool_size=(2, 2, 2))(l3)
l4 = conv3d(128, kernel_size=(3, 3, 3), padding='same')(l4)
l4 = BatchNormalization()(l4)
l4 = Activation('relu')(l4)
l4 = conv3d(128, kernel_size=(3, 3, 3), padding='same')(l4)
l4 = BatchNormalization()(l4)
l4 = Activation('relu')(l4)



# Layer 5
l5 = MaxPooling3D(pool_size=(2, 2, 2))(l4)
l5 = conv3d(256, kernel_size=(3, 3, 3), padding='same')(l5)
l5 = BatchNormalization()(l5)
l5 = Activation('relu')(l5)
l5 = conv3d(256, kernel_size=(3, 3, 3), padding='same')(l5)
l5 = BatchNormalization()(l5)
l5 = Activation('relu')(l5)


# Layer 6
l6 = UpSampling3D(size=(2,2,2))(l5)
l6 = conv3d_transpose(128, kernel_size=(2,2,2), padding='same')(l6)
l6 = concatenate([l4, l6], axis=4)
l6 = conv3d(128, kernel_size=(3, 3, 3), padding='same')(l6)
l6 = BatchNormalization()(l6)
l6 = Activation('relu')(l6)
l6 = conv3d(128, kernel_size=(3, 3, 3), padding='same')(l6)
l6 = BatchNormalization()(l6)
l6 = Activation('relu')(l6)


# Layer 7
l7 = UpSampling3D(size=(2,2,2))(l6)
l7 = conv3d_transpose(64, kernel_size=(2,2,2), padding='same')(l7)
l7 = concatenate([l3, l7], axis=4)
l7 = conv3d(64, kernel_size=(3, 3, 3), padding='same')(l7)
l7 = BatchNormalization()(l7)
l7 = Activation('relu')(l7)
l7 = conv3d(64, kernel_size=(3, 3, 3), padding='same')(l7)
l7 = BatchNormalization()(l7)
l7 = Activation('relu')(l7)


# Layer 8
l8 = UpSampling3D(size=(2,2,2))(l7)
l8 = conv3d_transpose(32, kernel_size=(2,2,2), padding='same')(l8)
l8 = concatenate([l2, l8], axis=4)
l8 = conv3d(32, kernel_size=(3, 3, 3), padding='same')(l8)
l8 = BatchNormalization()(l8)
l8 = Activation('relu')(l8)
l8 = conv3d(32, kernel_size=(3, 3, 3), padding='same')(l8)
l8 = BatchNormalization()(l8)
l8 = Activation('relu')(l8)


# Layer 9
l9 = UpSampling3D(size=(2,2,2))(l8)
l9 = conv3d_transpose(16, kernel_size=(2,2,2), padding='same')(l9)
l9 = concatenate([l1, l9], axis=4)
l9 = conv3d(16, kernel_size=(3, 3, 3), padding='same')(l9)
l9 = BatchNormalization()(l9)
l9 = Activation('relu')(l9)
l9 = conv3d(16, kernel_size=(3, 3, 3), padding='same')(l9)
l9 = BatchNormalization()(l9)
l9 = Activation('relu')(l9)


output = conv3d(1, (1, 1, 1), padding='same' )(l9)
# output = Dropout(0.1)(l9)



checkpoint_path = "stadistics/s1-{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=False, period=500)
# Creamos el modelo
modelo = Model(inputs=inputs, outputs=output)
modelo.summary()

modelo.compile(loss='mse',
 			   optimizer=Adam(lr=learning_rate, beta_1 = B1, beta_2=B2),
  			   metrics=['accuracy', 'mse'])


modelo.fit(	x=x_train,
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
    with open('pruebas/test'+str(i)+'.txt', 'wb') as file:
        pickle.dump(images[i], file)
