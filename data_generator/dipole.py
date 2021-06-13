import numpy as np
from numpy import e
import tensorflow as tf


def dipole_kernel(N, spatial_res):
    x = np.linspace(-N[0] / 2, N[0] / 2 - 1, N[0])
    y = np.linspace(-N[1] / 2, N[1] / 2 - 1, N[1])
    z = np.linspace(-N[2] / 2, N[2] / 2 - 1, N[2])
    ky, kx, kz = np.meshgrid(y, x, z)
    kx = (kx / np.max(np.abs(kx))) / spatial_res[0]
    ky = (ky / np.max(np.abs(ky))) / spatial_res[1]
    kz = (kz / np.max(np.abs(kz))) / spatial_res[2]

    k2 = kx ** 2 + ky ** 2 + kz ** 2
    R_tot = np.eye(3)
    kernel = np.fft.fftshift(1 / 3 - (kx * R_tot[2, 0] + ky * R_tot[2, 1] + kz * R_tot[2, 2]) ** 2 / (k2 + e))

    return kernel


@tf.function
def tf_dipole_kernel(N, spatial_res):
    # x = tf.linspace(-N[0] / 2, N[0] / 2 - 1, N[0])
    # y = tf.linspace(-N[1] / 2, N[1] / 2 - 1, N[1])
    # z = tf.linspace(-N[2] / 2, N[2] / 2 - 1, N[2])
    x = tf.cast(tf.range(-N[0] // 2, N[0] // 2), tf.int32)
    y = tf.cast(tf.range(-N[0] // 2, N[0] // 2), tf.int32)
    z = tf.cast(tf.range(-N[0] // 2, N[0] // 2), tf.int32)
    ky, kx, kz = tf.meshgrid(y, x, z, indexing='ij')
    kx = (kx / np.max(tf.abs(kx))) / spatial_res[0]
    ky = (ky / np.max(tf.abs(ky))) / spatial_res[1]
    kz = (kz / np.max(tf.abs(kz))) / spatial_res[2]

    k2 = kx ** 2 + ky ** 2 + kz ** 2
    R_tot = tf.eye(3)
    kernel = tf.signal.fftshift(1 /  3 - (kx * R_tot[2, 0] + ky * R_tot[2, 1] + kz * R_tot[2, 2]) ** 2 / (k2 + e))

    return kernel


if __name__=='__main__':
    import ants
    print(tf_dipole_kernel([12, 12, 12], [1, 1, 1]).shape)
    # ants.image_write(ants.from_numpy(dipole_kernel([96, 96, 96], [1, 1, 1])), 'sample.nii.gz')

