import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import ants
import numpy as np
from model.estimator import Estimator
from testing_tools import *


image_name = 'rimg4\\e1'
input_path = 'D:\\files\\tesis\invivo_uc\\{}'.format(image_name)
output_path = ''

image_pha, sitk_pha = open_nii_gz(os.path.join(input_path, 'unwrapped.nii'))
image_mag, sitk_mag = open_nii_gz(os.path.join(input_path, 'mag.nii.gz'))

image_mag = np.where(image_mag > 0, 1., 0.)
image_pha *= image_mag

gamma = 267.522e6
TE = 3.6e-3
B0 = 3

image_pha /= TE
image_pha /= B0
image_pha /= gamma / 1e6
print(np.max(image_pha))
new = sitk_pha.new_image_like(np.squeeze(image_pha))
ants.image_write(new, 'phase_in.nii.gz')

image = np.concatenate([image_pha, image_mag], -1)


version = 'back_noise_10'
size = (*[96, ] * 3 + [1],)
lr = 1e-3
model = Estimator(lr, size, version)

out = np.squeeze(np.array(list(model._estimator.predict(lambda: from_numpy(image))))[0])
# mask, _, _ = open_nii_gz('MaskBrainExtracted.nii.gz')
# mask = central_crop(mask, (160, 192, 192))
# out = np.squeeze(out)*np.squeeze(mask)
print(out.shape)
new = sitk_pha.new_image_like(np.squeeze(out))
ants.image_write(new, 'out_{}_{}_{}.nii.gz'.format(version, *image_name.split('\\')))
