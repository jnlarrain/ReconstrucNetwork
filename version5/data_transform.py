import numpy as np
from os import walk
import pickle

FOV = 64


"""
Script to transform data from the matlab format to numpy array and save them as python pickle
"""

def limpiar_linea(linea):
    lista = list(map(float, linea.strip().replace(' ', '').split(',')))
    filtrada = [x if not np.isnan(x) else 0 for x in lista]
    return filtrada


nombres = list(filter(lambda x: 'f' not in x, list(walk('D:/' + str(FOV) + "data/data"))[0][2]))
nombres2 = list(filter(lambda x: 'f' in x, list(walk('D:/' + str(FOV) + "data/data"))[0][2]))
nombres = ['D:/' + str(FOV) + "data/data/" + path for path in nombres]
nombres2 = ['D:/' + str(FOV) + "data/data/" + path for path in nombres2]
contador = 0
for i in nombres:
    with open(i, 'r') as file:
        datos = np.reshape(np.array(list(map(limpiar_linea, file.readlines())), dtype='float16'), [FOV, FOV, FOV, 1])
        datos -= np.min(datos)
        datos /= np.max(datos)
    with open('D:/' + str(FOV) + 'data/new_data/' + str(contador) + '.txt', 'wb') as file:
        pickle.dump(datos, file)
    print(contador)
    contador += 1
contador = 0
for i in nombres2:
    with open(i, 'r') as file:
        datos = np.reshape(np.array(list(map(limpiar_linea, file.readlines())), dtype='float16'), [FOV, FOV, FOV, 1])
        datos -= np.min(datos)
        datos /= np.max(datos)
    with open('D:/' + str(FOV) + 'data/new_dataf/' + str(contador) + '.txt', 'wb') as file:
        pickle.dump(datos, file)
    print(contador, 'f')
    contador += 1
