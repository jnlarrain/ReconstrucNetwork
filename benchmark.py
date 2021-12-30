import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import ants
import numpy as np
from model.estimator import Estimator
import time


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
    data = ants.image_read(path)
    data = ants.reorient_image2(data, 'RAI')
    return np.expand_dims(data.numpy().astype('float32'), [0, -1]), data


def from_numpy(elements):
    data = tf.data.Dataset.from_tensor_slices((elements, elements))
    data = data.map(lambda x, y: (tf.cast(x, tf.float32), y))
    data = data.batch(1)
    return data


# model_version = 'res_noise_norm_4_backupd_2'
model_version = '48_res_paper_1'
experiment = '2'
size = (*[96, ] * 3 + [1],)
lr = 1e-3
model = Estimator(lr, size, model_version)
for i in range(1, 3):
    for j in range(1, 3):
        
        input_path = f'C:\\Users\\jlarr\\Desktop\\ReconstrucNetwork\\challenge\\Sim{i}Snr{j}'
        image_pha, original_data = open_nii_gz(os.path.join(input_path, 'Frequency.nii.gz'))
        image_mag, _ = open_nii_gz(os.path.join(input_path, 'MaskBrainExtracted.nii.gz'))
        ground, _ = open_nii_gz(os.path.join(input_path, 'GT', 'Chi.nii.gz'))

        image_mag = np.where(image_mag > 0, 1., 0.)

        image_pha /= (42.58 * 7)
        image_pha = central_crop(image_pha, (160, 192, 192))
        image_mag = central_crop(image_mag, (160, 192, 192))
        image_pha *= image_mag


        image = np.concatenate([image_pha, image_mag], -1)
        t = time.time()
        out = np.squeeze(np.array(list(model._estimator.predict(lambda: from_numpy(image)))))
        out *= np.squeeze(image_mag)
        out = np.pad(out, [[2, 2], [6, 7], [6, 7]])
        print(time.time() -t)

        print( os.path.join(input_path, f'output_noise{model_version}-{experiment}.nii'))
        ants.image_write(ants.reorient_image2(original_data.new_image_like(out), 'LPI'),
                         os.path.join(input_path, str(experiment),f'output_noise_{model_version}-{experiment}.nii'))





















