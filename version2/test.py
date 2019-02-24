import pickle
from keras.models import load_model
import cv2
import numpy as np
from matplotlib import pyplot as plt
from os import walk
import cv2

def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return keras.losses.mean_squared_error(y_true, y_pred)+abs(dice_coefficient(y_true, y_pred))

modelo = load_model('stadistics/ms-0025.ckpt')

batch_size = 1
epochs = 128
FOV = 64
number_of_data = 32
input_shape = (FOV, FOV, FOV, 1)

def abrir_data(path):
    with open(path, 'rb') as file:
        datos = pickle.load(file)
        return datos

datos = list(walk(str(FOV)+"data/new_data"))[0][2]
datos = [str(FOV)+"data/new_data/"+path for path in datos[9*number_of_data:10*number_of_data]]
x_train = np.array(list(map(abrir_data, datos)))*1e10
datos2 = list(walk(str(FOV)+"data/new_dataf"))[0][2]
datos2 = [str(FOV)+"data/new_dataf/"+path for path in datos2[number_of_data:2*number_of_data]]
x_test= np.array(list(map(abrir_data, datos2)))*1e10
print('Data already done')




images = modelo.predict(x_train, batch_size, verbose=1)
for i in range(len(x_train)):
	with open('pruebas/test'+str(i)+'.txt', 'wb') as file:
		pickle.dump(images[i], file)