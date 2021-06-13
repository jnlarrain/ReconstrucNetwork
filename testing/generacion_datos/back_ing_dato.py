"""
Archivo para ala generacion de datos sinteticos QSM

Se le puede o no agregar ruido a las imagenes, el ruido puede ser complejo, normal o uniforme. Tambien se le puede
agregar focos de campo fuera del campo de vision de la imagen, y cambiar el comportamineto de la interfacez.

El algoritmo utiliza la formula teorica y no la convolucion del dipolo
"""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


from data_generator import add_complex_noise
from data_generator.figures.spherestf import sphere
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
from data_generator.dipole import dipole_kernel
from data_generator.figures.K_space import calculate_k
from testing_tools import *


def foco_externo(number, fov, _susceptibilidad, _radio=False):
    if not _radio:
        _radio = [1, fov[0] // 20]
    background = np.zeros((160, 192, 192))
    posible_range = [x for x in range(fov[0]) if x < int(fov[0] // 8) or x > int(fov[0] // 8 * 7)]
    for _ in range(number):
        k, radio, suscep = kspace(fov, _radio, posible_range, _susceptibilidad, True)
        suscep = _susceptibilidad
        susceptibilidad, campo, _ = sphere(k, radio, suscep, _susceptibilidad)
        background += campo[80:240, 96:288, 96:288].numpy()
    return background


def kspace( original_fov, radio, range_center, chi, center_choice=(), _cylinder=False):
    suscep = np.random.uniform(0, chi) * np.random.choice([-1, 1])
    radio = np.random.randint(*radio)
    if center_choice:
        centers = [np.random.choice(range_center), np.random.choice(range_center), np.random.choice(range_center)]
    else:
        centers = [np.random.randint(*range_center), np.random.randint(*range_center), np.random.randint(*range_center)]

    k = (*calculate_k(original_fov, original_fov, center=centers),)
    return k, radio, suscep



output_path = 'D:\\files\\ReconstrucNetwork\\outputs\\SNR'
input_path = '/SNR1/Sim1Snr1'
image_pha, _ = open_nii_gz(os.path.join(input_path, 'Frequency.nii.gz'))
image_mag, _ = open_nii_gz(os.path.join(input_path, 'MaskBrainExtracted.nii.gz'))
ground, _ = open_nii_gz(os.path.join(input_path, 'GT', 'Chi.nii.gz'))

image_pha = central_crop(ground, (160, 192, 192))
image_pha = np.pad(np.squeeze(image_pha), [[160, 160], [192, 192], [192, 192]])
dipole = dipole_kernel(image_pha.shape, [9e-6, 9e-6, 9e-6])
image_pha = np.squeeze(central_crop(ifft(fft(image_pha)*dipole), (160, 192, 192)))
image_mag = central_crop(image_mag, (160, 192, 192))
image_mag = np.squeeze(np.where(image_mag > 0, 1., 0.))

gamma = 267.522e6
TE = 3.6e-3
B0 = 3

image_pha /= TE
image_pha /= B0
image_pha /= gamma / 1e6

# image -= np.mean(image)


image_pha += np.squeeze(foco_externo(8, (320, 384, 384), 1e-1, (1, 20)) / 1e-6)
ants.image_write(ants.from_numpy(np.squeeze(image_pha)), 'backgourd_phase.nii.gz')
magnitud2, phase = add_complex_noise(image_mag, image_pha)
ants.image_write(ants.from_numpy(np.squeeze(image_pha)), 'phase_final.nii.gz')







