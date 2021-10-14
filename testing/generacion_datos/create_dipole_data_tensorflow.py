import numpy as np
import os
import ants
from data_generator.dipole import dipole_kernel
from tqdm import tqdm
from testing.testing_tools.tools import ifft, fft
import tensorflow as tf

size = 96

# tfrecords path


def open_nii_gz(path):
    data = ants.image_read(path)
    data = ants.reorient_image2(data, 'RAI')
    return data.numpy().astype('float32'), data



# image = ants.image_read(i)
# image = ants.reorient_image2(image, 'RAI').numpy()


# image = np.pad(image, [[48, 48], [48, 48], [48, 48]])

input_path = f'D:\\files\\ReconstrucNetwork\\SNR1\\Sim1Snr1'
image_pha, _ = open_nii_gz(os.path.join(input_path, 'ground_pad.nii.gz'))
dipole = dipole_kernel(image_pha.shape, [9e-6, 9e-6, 9e-6])
# dipole = np.pad(dipole, [[48, 48], [48, 48], [48, 48]])

# ants.image_write(ants.from_numpy(np.fft.fftshift(dipole)), 'dipole.nii.gz')
# exit()
print(image_pha.shape, dipole.shape)
final = tf.nn.convolution(tf.cast(image_pha, tf.float32), tf.cast(np.fft.ifft(np.fft.fftshift(dipole)),
                                                                  tf.float32)).numpy()
# final = central_crop(Fourier_inverse(Fourier_shift(image)*dipole), [96, 96, 96])
ants.image_write(ants.from_numpy(final), 'image_tf_dipole.nii.gz')
