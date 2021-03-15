import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

import tensorflow as tf
import nibabel as nib
import numpy as np
from model.estimator import Estimator

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
    image = np.expand_dims(image,[0, -1])
    return image


def open_nii_gz(path):
    data = nib.load(path)
    return data.get_fdata().astype('float32'), data.header, data.affine


def from_numpy(elements):
    data = tf.data.Dataset.from_tensor_slices((elements, elements))
    data = data.batch(1)
    return data

image, header, affine = open_nii_gz('Frequency.nii.gz')
image = central_crop(image, (160, 192, 192))
#image /= (7*42.7747892)
image *= 1e-4
header['dim'] = [1, 160, 192, 192, 1, 1, 1, 1]
new = nib.Nifti1Image(np.squeeze(image), affine, header)
nib.save(new, 'input.nii.gz')
print(image.shape)
print(header['dim'])

version = 3
size = (*[96, ]*3+[1],)
lr = 1e-3
model = Estimator(lr, size, version)

out = np.array(list(model._estimator.predict(lambda: from_numpy(image))))[0]
#mask, _, _ = open_nii_gz('MaskBrainExtracted.nii.gz')
#mask = central_crop(mask, (160, 192, 192))
#out = np.squeeze(out)*np.squeeze(mask) 
print(out.shape)
new = nib.Nifti1Image(out, affine, header)
nib.save(new, 'out.nii.gz')
