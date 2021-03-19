"""
Archivo para ala generacion de datos sinteticos QSM

Se le puede o no agregar ruido a las imagenes, el ruido puede ser complejo, normal o uniforme. Tambien se le puede
agregar focos de campo fuera del campo de vision de la imagen, y cambiar el comportamineto de la interfacez.

El algoritmo utiliza la formula teorica y no la convolucion del dipolo
"""

import os
import argparse
from tqdm import tqdm
from random import randint, choice
import numpy as np
from data_generator.tfrecords.tfrecords import convert_tfrecords
from data_generator.figures.spherestf import sphere
from data_generator.figures.cylinderstf import cylinder
from data_generator.figures.K_space import calculate_k
from threading import Thread
from os import walk
import ants
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

size = 96
threads_number = 128

# tfrecords path
train_tfrecord_path = 'F:/' + str(size) + 'data/train/'
test_tfrecord_path = 'F:/' + str(size) + 'data/test/'


if not os.path.exists('F:/' + str(size) + 'data'):
    os.mkdir('F:/' + str(size) + 'data')

if not os.path.exists(train_tfrecord_path):
    os.mkdir(train_tfrecord_path)

if not os.path.exists(test_tfrecord_path):
    os.mkdir(test_tfrecord_path)

size = 96
FOV = [size, size, size]
FOV2 = [size * 2, size * 2, size * 2]

datos_test = list(walk(test_tfrecord_path))[0][2]
datos_train = list(walk(train_tfrecord_path))[0][2]

numero_elementos = 1#randint(12, 1024)
range_center = [size // 4, size // 4 * 3]
range_radio = [4, size // 6]
c_range = 8 # rango en que los radios de los cilindros son mas pequeños

N1 = 1  # numero de datos de entrenamiento
N2 = 1     # numero de datos de evaluacion






def new_image(N_archivo, range_radio, range_center):
    """

    Parameters
    ----------
    N_archivo
    range_radio
    range_center

    Returns
    -------

    """
    susceptibilidad = np.zeros(FOV)
    phase = np.zeros(FOV)
    spheres_inner_number = N_archivo - randint(0, N_archivo) + randint(0, N_archivo)
    cylinder_inner_number = N_archivo - randint(0, N_archivo) + randint(0, N_archivo)
    for _ in range(spheres_inner_number):
        k, radio, suscep = randoms(FOV, range_radio, range_center, 1e-2)
        x, y, _ = sphere(k, radio, suscep, -9e-6)
        x = x.numpy()
        x[x == -9e-6] = 0
        susceptibilidad += x #/ 1e-6
        phase += y / 1e-6
    for _ in range(cylinder_inner_number):
        k, radio, suscep, p1, p2 = randoms(FOV, np.array(range_radio)//c_range, range_center, 1e-2, _cylinder=True)
        x, y = cylinder(k, FOV, p1, p2, radio, suscep, -9e-6)
        x = x.numpy()
        x[x == -9e-6] = 0
        susceptibilidad += x #/ 1e-6
        phase += y / 1e-6
    # total = cylinder_inner_number + spheres_inner_number
    return susceptibilidad, phase.numpy()


def add_noise(susceptibilidad, phase):
    """

    Parameters
    ----------
    susceptibilidad
    phase

    Returns
    -------

    """

    '''
    susceptibilidad -> espheras tienen promedio 9pmm(revisar el signo de agua en ppm) puede ser diferente
    susceptibilidad -> fondo no lo toques (fondo 0)
    
    phase -> promedio de señal entorno a 0
    phase -> fondo no lo toco
    
    para calcular el promedio:
     calcular el 
     np.mean(phase)
     phase -= mean?
    
    
    '''

    back_spheres = randint(0, 32)
    back_spheres = out_spheres(back_spheres)
    percent = 1 / randint(256, 384) * np.max(np.abs(phase))
    mascara = susceptibilidad.copy()
    mascara[mascara != 0] = 1

    noise = np.random.normal(size=phase.shape, scale=percent) * mascara
    out_noise = np.random.uniform(-np.pi - np.min(back_spheres), np.pi - np.max(back_spheres),
                                  phase.shape) * (1 - mascara)
    phase += back_spheres + noise + out_noise
    susceptibilidad = np.reshape(susceptibilidad, [1] + FOV)
    phase = np.reshape(phase, [1] + FOV)
    return susceptibilidad, phase


def training_data(file, path):
    """

    Parameters
    ----------
    file
    path

    Returns
    -------

    """
    susceptibilidad, phase = new_image(numero_elementos, range_radio, range_center)
    ants.image_write(ants.from_numpy(np.squeeze(susceptibilidad)), path + str(file) + 'sus.nii.gz')
    ants.image_write(ants.from_numpy(np.squeeze(phase)), path + str(file) + 'phase.nii.gz')
    susceptibilidad, phase = add_noise(susceptibilidad, phase)
    ants.image_write(ants.from_numpy(np.squeeze(phase)), path + str(file) + 'phase_final.nii.gz')
    convert_tfrecords(phase, susceptibilidad, path + str(file))


def create(N, path):
    """

    Parameters
    ----------
    N
    path

    Returns
    -------

    """
    threads = []
    for file in tqdm(range(N)):
        # if str(file) + '.tfrecords' not in datos_train:

        threads.append(Thread(target=training_data, args=(file, path)))
        threads[-1].start()
        if file % threads_number == 0:
            for worker in threads:
                worker.join()
            threads = []
    for worker in threads:
        worker.join()


create(N1, train_tfrecord_path)
create(N2, test_tfrecord_path)


