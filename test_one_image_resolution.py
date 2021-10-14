import numpy as np
from data_generator.figures.K_space import calculate_k
from data_generator.figures.spherestf import sphere
import tensorflow as tf
import ants
from model.estimator import Estimator
from data_generator import add_complex_noise


def create_testing_sphere(fov, centers, inner_sus, out_sus):
    radio = fov[0]//4
    k = (*calculate_k(fov, fov, center=centers),)
    x, y, _ = sphere(k, radio, inner_sus, out_sus)
    return x, y


def central_crop(image, size):
    def delta(size1, size2):
        diff = np.abs(size1 - size2)
        _delta = diff // 2
        return _delta

    image = np.squeeze(image)
    shape = image.shape
    print(shape)
    dx = delta(shape[0], size[0])
    dy = delta(shape[1], size[1])
    dz = delta(shape[2], size[2])
    image = image[dx:size[0] + dx, dy:size[1] + dy, dz:size[2] + dz]
    image = np.expand_dims(image, [0, -1])
    return image


def open_nii_gz(path):
    data = ants.image_read(path)
    data = ants.reorient_image2(data, 'RAI')
    return np.expand_dims(data.numpy().astype('float32'), [0, -1]), data


def from_numpy(elements):
    data = tf.data.Dataset.from_tensor_slices((elements, elements))
    data = data.batch(1)
    return data


sizes_images = [32, 48, 64, 96, 128, 256]
list_snr = [256, 512, 1024, 2048, 4096]

for fov_size in sizes_images:
    for snr in list_snr:
        fov = [fov_size, ] * 3
        number_figures = 1
        centers = [[fov[0] // 2, ] * 3]

        sus = 1e-6
        sus_ext = 0
        sus_img, pha_img = create_testing_sphere(fov, centers[0], sus, sus_ext)
        magnitud = np.where(sus_img > 0, 1, 0)
        magnitud, pha_img = add_complex_noise(magnitud, pha_img, snr)

        image = np.concatenate([np.expand_dims(pha_img, [0, -1]),
                               np.expand_dims(magnitud, [0, -1])], -1)
        version = 'noise_48_5'
        size = (*[96, ] * 3 + [1],)
        lr = 1e-3
        model = Estimator(lr, size, version)

        out = np.squeeze(np.array(list(model._estimator.predict(lambda: from_numpy(image)))))

        print(fov_size, snr, np.sum(np.sqrt(np.square(out-sus_img))))

        ants.image_write(ants.from_numpy(out),  f'results_testing/fov_{fov_size}_snr_{snr}.nii')





















