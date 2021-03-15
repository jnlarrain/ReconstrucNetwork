"""
Archivo para ala generacion de datos sinteticos QSM
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

size = 96
threads_number = 128

# tfrecords path
train_tfrecord_path = 'H:/' + str(size) + 'data/train/'
test_tfrecord_path = 'H:/' + str(size) + 'data/test/'
size = 96
FOV = [size, size, size]
FOV2 = [size * 2, size * 2, size * 2]

datos_test = list(walk(test_tfrecord_path))[0][2]
datos_train = list(walk(train_tfrecord_path))[0][2]

numero_elementos = randint(12, 128)
range_center = [size // 4, size // 4 * 3]
range_radio = [4, size // 6]
c_range = 8 # rango en que los radios de los cilindros son mas pequeños

N1 = 1024  # numero de datos de entrenamiento
N2 = 512     # numero de datos de evaluacion

def randoms(_fov, r=None, center=None, chi=None, _choice=False, _cylinder=False):
    """
    Funcion para generar k space
    Parameters
    ----------
    _fov
    r
    center
    chi float
        valor de la susceptibilidad
    _choice bool
        flag para indicar que el rango de centros ya fue seleccionado por lo que solo se debe tomar uno al azar.
    _cylinder

    Returns
    -------

    """
    suscep = np.random.normal(0, chi) * 1e-6 * np.random.choice([-1, 1])
    radio = randint(*r)
    if _cylinder:
        p1 = np.array([randint(*range_center), randint(*range_center), randint(*range_center)], dtype='float32')
        p2 = np.array([randint(*range_center), randint(*range_center), randint(*range_center)], dtype='float32')
        k = (*calculate_k(FOV, FOV, points=p1),)
        return k, radio, suscep, p1, p2
    if _choice:
        centers = [choice(center), choice(center), choice(center)]
    else:
        centers = [randint(*center), randint(*center), randint(*center)]

    k = (*calculate_k(_fov, _fov, center=centers),)
    return k, radio, suscep


def out_spheres(number):
    """

    Parameters
    ----------
    number

    Returns
    -------

    """
    # campo inicial
    background = np.zeros(FOV)
    # rango posible para los centros
    posible_range = [x for x in range(FOV2[0]) if x < size // 8 or x > int(size * 14 / 8)]
    for _ in range(number):
        k, radio, suscep = randoms(FOV2, [1, size // 16 - 1], posible_range, 1, True)
        susceptibilidad, campo, _ = sphere(k, radio, suscep, 0)
        susceptibilidad /= 1e-6
        cnt = FOV[0]
        background += susceptibilidad[cnt // 2:cnt // 2 + cnt, cnt // 2:cnt // 2 + cnt, cnt // 2:cnt // 2 + cnt]
    return background


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
        susceptibilidad += x / 1e-6
        phase += y / 1e-6
    for _ in range(cylinder_inner_number):
        k, radio, suscep, p1, p2 = randoms(FOV, np.array(range_radio)//c_range, range_center, 1e-2, _cylinder=True)
        x, y = cylinder(k, FOV, p1, p2, radio, suscep, -9e-6)
        x = x.numpy()
        x[x == -9e-6] = 0
        susceptibilidad += x / 1e-6
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
    # ants.image_write(ants.from_numpy(np.squeeze(susceptibilidad)), path + str(file) + 'sus.nii.gz')
    # ants.image_write(ants.from_numpy(np.squeeze(phase)), path + str(file) + 'phase.nii.gz')
    susceptibilidad, phase = add_noise(susceptibilidad, phase)
    # ants.image_write(ants.from_numpy(np.squeeze(phase)), path + str(file) + 'phase_final.nii.gz')
    convert_tfrecords(susceptibilidad, phase, path + str(file))


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


