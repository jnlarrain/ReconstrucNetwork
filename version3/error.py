import pickle
import cv2
import numpy as np
from matplotlib import pyplot as plt
from os import walk
import cv2


class Data:
	def __init__(self, datos, predicciones, ide):
		self.datos = datos
		self.predicciones = predicciones
		self.ide = ide
		self.error_calculate()

	def error_calculate(self):
		print("La suma de las diferencias es:", sum(np.ravel(abs(self.datos-self.predicciones))))
		print("La suma de las diferencias al cuadrado es:", sum(np.ravel(abs(self.datos**2 - self.predicciones**2)**(1/2))))
		print('Las diferencias graficas de cortes son:')
		a = self.datos.shape
		self.dif_graficas([a[0]//2, a[1]//2, a[2]//2])

	def dif_graficas(self, pos):
		fig=plt.figure(figsize=(2, 3))
		img = self.datos[pos[0], :, :]
		img = np.reshape(img, (img.shape[0], img.shape[1]))
		fig.add_subplot(3, 2, 1)
		plt.imshow(img)
		img = self.datos[:, pos[1], :]
		img = np.reshape(img, (img.shape[0], img.shape[1]))
		fig.add_subplot(3, 2, 3)
		plt.imshow(img)
		img = self.datos[:, :, pos[2]]
		img = np.reshape(img, (img.shape[0], img.shape[1]))
		fig.add_subplot(3, 2, 5)
		plt.imshow(img)
		img = self.predicciones[pos[0], :, :]
		img = np.reshape(img, (img.shape[0], img.shape[1]))
		fig.add_subplot(3, 2, 2)
		plt.imshow(img)
		img = self.predicciones[:, pos[1], :]
		img = np.reshape(img, (img.shape[0], img.shape[1]))
		fig.add_subplot(3, 2, 4)
		plt.imshow(img)
		img = self.predicciones[:, :, pos[2]]
		img = np.reshape(img, (img.shape[0], img.shape[1]))
		fig.add_subplot(3, 2, 6)
		plt.imshow(img)
		plt.show()




batch_size = 1
epochs = 128
FOV = 64
number_of_data = 32
input_shape = (FOV, FOV, FOV, 1)

def abrir_data(path):
    with open(path, 'rb') as file:
        datos = pickle.load(file)
        return datos

datos = list(walk("pruebas"))[0][2]
datos = ["pruebas/"+path for path in datos]
x_pred = np.array(list(map(abrir_data, datos)))
datos2 = list(walk(str(FOV)+"data/new_dataf"))[0][2]
datos2 = [str(FOV)+"data/new_dataf/"+path for path in datos2[9*number_of_data:10*number_of_data]]
# datos2 = list(walk("pruebas/malas"))[0][2]
# datos2 = ["pruebas/malas/"+path for path in datos2]
x_test= np.array(list(map(abrir_data, datos2)))

for i in range(len(datos)):
	print(i)
	Data(x_test[i], x_pred[i], i)


print('Data already done')






