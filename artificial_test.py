import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import ants
from model.estimator import Estimator
import numpy as np
from data_generator.figures.K_space import calculate_k
from data_generator.figures.spherestf import sphere
from model.imagenes import ImageShower
import matplotlib.pyplot as plt

def from_numpy(elements):
    data = tf.data.Dataset.from_tensor_slices((elements, elements))
    data = data.batch(1)
    return data



#######################################################################################################################

fov = [[48, ]*3, [96, ]*3, [128, ]*3, [256, ]*3]
number_figures = 1


#######################################################################################################################


def create_testing_sphere(fov, inner_sus, out_sus):
    radio = fov[0]//8
    centers = [fov[0] // 2, ] * 3
    k = (*calculate_k(fov, fov, center=centers),)
    x, y, _ = sphere(k, radio, inner_sus, out_sus)
    return x, y

version = 'normal_48_last'
size = (*[96, ] * 3 + [1],)
lr = 1e-3
model = Estimator(lr, size, version)
sus = 1e-1
sus_ext = 0

for fov_size in fov:
    image_show = ImageShower(fov_size)

    sus_img, pha_img = create_testing_sphere(fov_size, sus, sus_ext)

    image = np.concatenate([np.expand_dims(pha_img, [0, -1]),
                            np.expand_dims(np.where(sus_img > 0, 1, 0), [0, -1])], -1)

    # version = '2_48'
    sus_img = np.expand_dims(sus_img, [0, -1])

    out = np.squeeze(np.array(list(model._estimator.predict(lambda: from_numpy(image)))))

    diff = sus_img - np.expand_dims(out, [0, -1])
    new_images = image_show.show_image_cuts(np.expand_dims(out, [0, -1]))
    new_diff = image_show.show_image_cuts(diff)
    new_labels = image_show.show_image_cuts(sus_img)
    result = tf.concat([new_labels[:, :, :, :], new_images[:, :, :, :],
                        new_diff[:, :, :, :]], 2)
    fig, _axs = plt.subplots(nrows=1, ncols=1)
    _axs.imshow(np.squeeze(result), cmap='gray')
    _axs.axis('off')
    pcm = _axs.pcolormesh(np.squeeze(result), cmap='gray')
    fig.colorbar(pcm, ax=_axs)
    plt.title(f'imagen sintetica de {fov_size}')
    plt.show()
    sus_img = np.squeeze(sus_img)
    ants.image_write(ants.from_numpy(out), f'out_dipole{fov_size}.nii')
    ants.image_write(ants.from_numpy(sus_img), f'gt{fov_size}.nii')
    ants.image_write(ants.from_numpy(sus_img-out), f'diff{fov_size}.nii')




