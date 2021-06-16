import numpy as np
import os
import ants
from data_generator.dipole import dipole_kernel
from tqdm import tqdm
from testing.testing_tools.tools import ifft, fft

size = 96

# tfrecords path
path = 'F:/' + str(size) + 'data/train/'

datos_train = [os.path.join(path, x) for x in os.listdir(path) if '.nii' in x]
sus = [x for x in datos_train if 'sus' in x]



for i in tqdm(sus):
    # image = ants.image_read(i)
    # image = ants.reorient_image2(image, 'RAI').numpy()
    image = ants.image_read(i).numpy()

    # image = np.pad(image, [[48, 48], [48, 48], [48, 48]])
    dipole = dipole_kernel(image.shape, [9e-6, 9e-6, 9e-6])
    # dipole = np.pad(dipole, [[48, 48], [48, 48], [48, 48]])

    # ants.image_write(ants.from_numpy(np.fft.fftshift(dipole)), 'dipole.nii.gz')
    # exit()
    final = ifft(fft(image)*dipole)
    # final = central_crop(Fourier_inverse(Fourier_shift(image)*dipole), [96, 96, 96])
    ants.image_write(ants.from_numpy(final), i[:-10]+'dipole2.nii.gz')





