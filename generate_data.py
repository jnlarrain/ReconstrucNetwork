"""
Archivo para ala generacion de datos sinteticos QSM

Se le puede o no agregar ruido a las imagenes, el ruido puede ser complejo, normal o uniforme. Tambien se le puede
agregar focos de campo fuera del campo de vision de la imagen, y cambiar el comportamineto de la interfacez.

El algoritmo utiliza la formula teorica y no la convolucion del dipolo
"""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tqdm import tqdm
from random import randint
from data_generator.tfrecords.tfrecords import convert_tfrecords
from data_generator import add_complex_noise
from data_generator.create_one_image import Data
from threading import Thread
from os import walk
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

size = 48
threads_number = 512
FOV = [size, size, size]


disk = 'D:/'

# tfrecords path
train_tfrecord_path_00 = disk + str(size) + 'data_normal3/train/'
test_tfrecord_path_00 = disk + str(size) + 'data_normal3/test/'
train_tfrecord_path_10 = disk + str(size) + 'data_back/train/'
test_tfrecord_path_10 = disk + str(size) + 'data_back/test/'
# train_tfrecord_path_01 = disk + str(size) + 'data_noise/train/'
# test_tfrecord_path_01 = disk + str(size) + 'data_noise/test/'
# train_tfrecord_path_11 = disk + str(size) + 'data/train/'
# test_tfrecord_path_11 = disk + str(size) + 'data/test/'

# if not os.path.exists(disk + str(size) + 'data'):
#     os.mkdir(disk + str(size) + 'data')

if not os.path.exists(disk + str(size) + 'data_normal2'):
    os.mkdir(disk + str(size) + 'data_normal2')

# if not os.path.exists(disk + str(size) + 'data_back'):
#     os.mkdir(disk + str(size) + 'data_back')
#
# if not os.path.exists(disk + str(size) + 'data_noise'):
#     os.mkdir(disk + str(size) + 'data_noise')

if not os.path.exists(train_tfrecord_path_00):
    os.mkdir(train_tfrecord_path_00)

if not os.path.exists(test_tfrecord_path_00):
    os.mkdir(test_tfrecord_path_00)


if not os.path.exists(train_tfrecord_path_10):
    os.mkdir(train_tfrecord_path_10)

if not os.path.exists(test_tfrecord_path_10):
    os.mkdir(test_tfrecord_path_10)
#
# if not os.path.exists(train_tfrecord_path_11):
#     os.mkdir(train_tfrecord_path_11)
#
# if not os.path.exists(test_tfrecord_path_11):
#     os.mkdir(test_tfrecord_path_11)
#
# if not os.path.exists(train_tfrecord_path_01):
#     os.mkdir(train_tfrecord_path_01)
#
# if not os.path.exists(test_tfrecord_path_01):
#     os.mkdir(test_tfrecord_path_01)


datos_test_00 = list(walk(test_tfrecord_path_00))[0][2]
datos_train_00 = list(walk(train_tfrecord_path_00))[0][2]
datos_test_10 = list(walk(test_tfrecord_path_10))[0][2]
datos_train_10 = list(walk(train_tfrecord_path_10))[0][2]
# datos_test_01 = list(walk(test_tfrecord_path_01))[0][2]
# datos_train_01 = list(walk(train_tfrecord_path_01))[0][2]
# datos_test_11 = list(walk(test_tfrecord_path_11))[0][2]
# datos_train_11 = list(walk(train_tfrecord_path_11))[0][2]

num_cilindros = 32
num_esferas = 128 # 128
susceptibilidad_interna = 1e-7 # 1e-8
susceptibilidad_externa = 0     # 9 * 1e-6
susceptibilidad_foco_externo = 1e-1
fov_foco_ext = [size*3, ]*3
bias = 0

N1 = 4096*16  # numero de datos de entrenamiento
N2 = 4096*4  # numero de datos de evaluacion


def training_data(file, num_esferas, num_cilindros, path):
    """

    Parameters
    ----------
    file
    path

    Returns
    -------

    """
    data = Data(size, susceptibilidad_interna, susceptibilidad_externa, bias, num_esferas, num_cilindros)
    susceptibilidad, phase, magnitud = data.new_image
    susceptibilidad /= 1e-6
    phase /= 1e-6
    if 'train' in path:
        convert_tfrecords(phase, magnitud, susceptibilidad, train_tfrecord_path_00 + str(file))
    else:
        convert_tfrecords(phase, magnitud, susceptibilidad, test_tfrecord_path_00 + str(file))
    # ants.image_write(ants.from_numpy(np.squeeze(susceptibilidad)), path + str(file) + 'sus.nii.gz')
    # ants.image_write(ants.from_numpy(np.squeeze(phase)), path + str(file) + 'phase.nii.gz')
    # ants.image_write(ants.from_numpy(np.squeeze(magnitud)), path + str(file) + 'mag.nii.gz')

    phase += data.foco_externo(8, fov_foco_ext, susceptibilidad_foco_externo, [1, size // 20]) / 1e-6

    if 'train' in path:
        convert_tfrecords(phase, magnitud, susceptibilidad, train_tfrecord_path_10 + str(file))
    else:
        convert_tfrecords(phase, magnitud, susceptibilidad, test_tfrecord_path_10 + str(file))
    #
    # # ants.image_write(ants.from_numpy(np.squeeze(phase)), path + str(file) + 'phase_with_background.nii.gz')
    #
    # # magnitud2, phase = add_complex_noise(magnitud, phase)
    # magnitud2, phase2 = add_complex_noise(magnitud, phase2)
    #
    # if 'train' in path:
    #     convert_tfrecords(phase2, magnitud, susceptibilidad, train_tfrecord_path_01 + str(file))
    # else:
    #     convert_tfrecords(phase2, magnitud, susceptibilidad, test_tfrecord_path_01 + str(file))

    # ants.image_write(ants.from_numpy(np.squeeze(phase)), path + str(file) + 'phase_final.nii.gz')
    # ants.image_write(ants.from_numpy(np.squeeze(magnitud2)), path + str(file) + 'mag_final.nii.gz')

    # convert_tfrecords(phase, magnitud, susceptibilidad, path + str(file))


def create(N, path, test_flag=False):
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
        if (str(file) + '.tfrecords' not in datos_train_00 or test_flag):
            spheres_number = num_esferas - randint(0, num_esferas) + randint(0, num_esferas)
            cylinder_number = num_cilindros - randint(0, num_cilindros) + randint(0, num_cilindros)
            threads.append(Thread(target=training_data, args=(file, spheres_number, cylinder_number, path)))
            threads[-1].start()
            if file % threads_number == 0:
                for worker in threads:
                    worker.join()
                threads = []
    for worker in threads:
        worker.join()


create(N1, train_tfrecord_path_00)
create(N2, test_tfrecord_path_00, True)


