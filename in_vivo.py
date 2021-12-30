import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import ants
import numpy as np
from model.estimator import Estimator
import time
from scipy.ndimage import zoom


def central_crop(image, size):
    def delta(size1, size2):
        diff = np.abs(size1 - size2)
        _delta = diff // 2
        return _delta

    image = np.squeeze(image)
    shape = image.shape
    dx = delta(shape[0], size[0])
    dy = delta(shape[1], size[1])
    dz = delta(shape[2], size[2])
    image = image[dx:size[0] + dx, dy:size[1] + dy, dz:size[2] + dz]
    image = np.expand_dims(image, [0, -1])
    return image


def open_nii_gz(path):
    data = ants.image_read(path).numpy().astype('float32')
    data = data[32:-32, 40:312, 8:-8]
    # data = ants.reorient_image2(data, 'RAI')
    if np.max(data) == 1:
        data = zoom(data, (.9, .9, 1/0.5966*.9), order=0)
    else:
        data = zoom(data, (.9, .9, 1/0.5966*.9), order=2)
    x, y, z = data.shape
    dx = int((1-(x/16-x//16))*16) if (x/16-x//16) else 0
    dy = int((1-(y/16-y//16))*16) if (y/16-y//16) else 0
    dz = int((1-(z/16-z//16))*16) if (z/16-z//16) else 0
    data = np.pad(data, [[dx, 0], [dy, 0], [dz, 0]])

    print(data.shape)
    return np.expand_dims(data, [0, -1]), data


def from_numpy(elements):
    data = tf.data.Dataset.from_tensor_slices((elements, elements))
    data = data.map(lambda x, y: (tf.cast(x, tf.float32), y))
    data = data.batch(1)
    return data


model_version = 'res_noise_norm_4_backupd_2'
# model_version = '48_res_paper_1'
size = (*[96, ] * 3 + [1],)
lr = 1e-3
model = Estimator(lr, size, model_version)

for directory in os.listdir('invivo'):

    print(os.path.join('invivo', directory))

    image_pha, _ = open_nii_gz(os.path.join('invivo', directory, 'invivo_phase.nii'))
    image_mag, _ = open_nii_gz(os.path.join('invivo', directory, 'invivo_mask.nii'))

    image_mag = np.where(image_mag > 0, 1., 0.)

    image_pha /= (42.58 * 7)
    TE = np.mean([0.00492,	0.00984,	0.01476,	0.01968,	0.0246,	0.02952])
    # image_pha /= 802.6141
    image_pha /= TE
    image_pha /= 3
    image_pha *= image_mag

    image = np.concatenate([image_pha, image_mag], -1)
    t = time.time()
    out = np.squeeze(np.array(list(model._estimator.predict(lambda: from_numpy(image)))))
    out *= np.squeeze(image_mag)
    print(time.time() -t)

    ants.image_write(ants.from_numpy(out), os.path.join('invivo', directory, f'{model_version}_reconstruction.nii'))
