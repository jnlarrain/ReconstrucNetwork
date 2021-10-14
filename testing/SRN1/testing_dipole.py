import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from model.estimator import Estimator
from tqdm import tqdm
from data_generator.dipole import dipole_kernel
from testing.testing_tools.tools import *


# output_path = 'D:\\files\\ReconstrucNetwork\\outputs\\SNR'
input_path = 'D:/files/ReconstrucNetwork/SNR1/Sim1Snr1'
# image_pha, _ = open_nii_gz(os.path.join(input_path, 'Frequency.nii.gz'))
image_mag, _ = open_nii_gz(os.path.join(input_path, 'MaskBrainExtracted.nii.gz'))
ground, _ = open_nii_gz(os.path.join(input_path, 'GT', 'Chi.nii.gz'))


image = np.pad(np.squeeze(ground), [[48, 48], [48, 48], [48, 48]])
dipole = dipole_kernel(image.shape, [9e-6, 9e-6, 9e-6])
image_pha = np.expand_dims(central_crop(ifft(fft(image)*dipole), (160, 192, 192)), -1)
image_mag = np.expand_dims(central_crop(image_mag, (160, 192, 192)), -1)
image_mag = np.where(image_mag > 0, 1., 0.)

ants.image_write(ants.from_numpy(image_pha),  'input.nii.gz')

gamma = 267.522
TE = 3.6e-3
B0 = 1.5

image_pha /= TE
image_pha /= B0
image_pha /= gamma

image = np.expand_dims(np.concatenate([image_pha, image_mag], -1), 0)
# image -= np.mean(image)

version = '2_48'
size = (*[96, ] * 3 + [1],)
lr = 1e-3
model = Estimator(lr, size, version)


out = np.squeeze(np.array(list(model._estimator.predict(lambda: from_numpy(image)))))

ants.image_write(ants.from_numpy(out),  'out_dipole2.nii.gz')

