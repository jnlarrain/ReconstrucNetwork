import os
from data_generator import add_complex_noise
from data_generator.figures.spherestf import sphere
from data_generator.dipole import dipole_kernel
from data_generator.figures.K_space import calculate_k
from testing.testing_tools.tools import *


def foco_externo(number, fov, _susceptibilidad, _radio=False):
    if not _radio:
        _radio = [1, fov[0] // 20]
    background = np.zeros((160, 160, 160))
    posible_range = [x for x in range(fov[0]) if x < int(fov[0] // 8) or x > int(fov[0] // 8 * 7)]
    for _ in range(number):
        k, radio, suscep = kspace(fov, _radio, posible_range, _susceptibilidad, True)
        suscep = _susceptibilidad
        susceptibilidad, campo, _ = sphere(k, radio, suscep, 0)
        background += campo[80:240, 80:240, 80:240].numpy()
    return background


def kspace( original_fov, radio, range_center, chi, center_choice=(), _cylinder=False):
    radio = np.random.randint(*radio)
    if center_choice:
        centers = [np.random.choice(range_center), np.random.choice(range_center), np.random.choice(range_center)]
    else:
        centers = [np.random.randint(*range_center), np.random.randint(*range_center), np.random.randint(*range_center)]

    k = (*calculate_k(original_fov, original_fov, center=centers),)
    return k, radio, chi


output_path = 'D:\\files\\ReconstrucNetwork\\outputs\\SNR'
input_path = 'D:/files/ReconstrucNetwork/SNR1/Sim1Snr1'
image_pha, _ = open_nii_gz(os.path.join(input_path, 'phase.nii'))
image_mag, _ = open_nii_gz(os.path.join(input_path, 'MaskBrainExtracted.nii.gz'))
ground, _ = open_nii_gz(os.path.join(input_path, 'GT', 'Chi.nii.gz'))

print(image_pha.shape)

image = np.pad(np.squeeze(ground), [[80, 80], [80, 80], [80, 80]])
dipole = dipole_kernel(image.shape, [9e-6, 9e-6, 9e-6])


image_pha_dipole = central_crop(ifft(fft(image)*dipole), (160, 160, 160))
image_pha = central_crop(image_pha, (160, 160, 160))
image_mag = central_crop(image_mag, (160, 160, 160))
image_mag = np.where(image_mag > 0, 1., 0.)
image_pha *= image_mag

gamma = 267.522
TE = 3.6e-3
B0 = 3

# ants.image_write(ants.from_numpy(image_pha), 'field_without_dipole.nii')
# ants.image_write(ants.from_numpy(image_pha_dipole), 'phase_with_dipole.nii')

image_pha /= TE
image_pha /= B0
image_pha /= gamma

# ants.image_write(ants.from_numpy(image_pha), 'phase_without_dipole.nii')


foco = np.squeeze(foco_externo(8, (320, 320, 320), 1e-2, (30, 40)))*1e6
print(np.mean(foco), np.max(foco), np.min(foco), foco.std())

image_pha_dipole += foco
# ants.image_write(ants.from_numpy(np.squeeze(image_pha)), 'backgourd_phase.nii.gz')
# magnitud2, phase = add_complex_noise(image_mag, image_pha)
# ants.image_write(ants.from_numpy(np.squeeze(image_pha)), 'phase_final.nii.gz')
image_pha_dipole *= image_mag
print(np.round(np.mean(image_pha), 2),
      np.round(np.max(image_pha), 2),
      np.round(np.min(image_pha), 2),
      np.round(image_pha.std(), 2))
print(np.round(np.mean(image_pha_dipole), 2),
      np.round(np.max(image_pha_dipole), 2),
      np.round(np.min(image_pha_dipole), 2),
      np.round(image_pha_dipole.std(), 2))
ants.image_write(ants.from_numpy(image_pha_dipole), 'phase_with_dipole_background.nii')
