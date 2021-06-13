import ants
import numpy as np
import tensorflow as tf


def delta(size1, size2):
    diff = np.abs(size1 - size2)
    _delta = diff // 2
    return _delta


def central_crop(image, size):
    image = np.squeeze(image)
    shape = image.shape
    dx = delta(shape[0], size[0])
    dy = delta(shape[1], size[1])
    dz = delta(shape[2], size[2])
    image = image[dx:size[0] + dx, dy:size[1] + dy, dz:size[2] + dz]
    return image


def fft(image):
    img = np.fft.fftn(image)
    return img


def ifft(fourier):
    img = np.real(np.fft.ifftn(fourier))
    return img


def open_nii_gz(path):
    data = ants.image_read(path)
    data = ants.reorient_image2(data, 'RAI')
    return np.expand_dims(data.numpy().astype('float32'), [0, -1]), data


def from_numpy(elements):
    data = tf.data.Dataset.from_tensor_slices((elements, elements))
    # data = data.unbatch()
    data = data.batch(1)
    return data


