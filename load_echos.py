import os
import tensorflow as tf
import SimpleITK as sitk
import numpy as np

input_path = 'D:\\files\ReconstrucNetwork\challenge\challenge\Sim1Snr2'
image_pha = os.path.join(input_path, 'Phase.nii.gz')
image_mag = os.path.join(input_path, 'Magnitude.nii.gz')
image_pha = sitk.ReadImage(image_pha)
image_mag = sitk.ReadImage(image_mag)

mag = sitk.GetArrayFromImage(image_mag)
phase = sitk.GetArrayFromImage(image_pha)


echos = [4e-3, 12e-3, 20e-3, 28e-3]

print(mag.shape)
num = np.zeros(mag.shape[1:])
den = np.zeros(mag.shape[1:])

for i in range(4):
    num += phase[i] * mag[i]
    den += echos[i] * mag[i]


eps = np.finfo(num.dtype).eps

new_phase1 = num / (den+eps)

gamma = 267.522
B0 = 7     # 1.5

phs_scale = gamma * B0

new_phase1 /= phs_scale

sitk.WriteImage(sitk.GetImageFromArray(new_phase1),  os.path.join(input_path, 'phase_compuesta.nii.gz'))
