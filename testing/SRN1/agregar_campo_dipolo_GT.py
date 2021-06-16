import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from model.estimator import Estimator
from tqdm import tqdm
from data_generator.dipole import dipole_kernel
from testing.testing_tools.tools import *


output_path = 'D:\\files\\ReconstrucNetwork\\outputs\\SNR'
input_path = '/SNR1/Sim1Snr1'
image_pha, _ = open_nii_gz(os.path.join(input_path, 'Frequency.nii.gz'))
image_mag, _ = open_nii_gz(os.path.join(input_path, 'MaskBrainExtracted.nii.gz'))
ground, _ = open_nii_gz(os.path.join(input_path, 'GT', 'Chi.nii.gz'))


image = np.pad(np.squeeze(ground), [[48, 48], [48, 48], [48, 48]])
dipole = dipole_kernel(image.shape, [9e-6, 9e-6, 9e-6])
image_pha = central_crop(image_pha, (160, 192, 192))
image_mag = central_crop(image_mag, (160, 192, 192))
image_mag = np.where(image_mag > 0, 1., 0.)


gamma = 267.522e6
TE = 3.6e-3
B0 = 1.5

image_pha /= TE
image_pha /= B0
image_pha /= gamma / 1e6

image = np.concatenate([image_pha, image_mag], -1)
# image -= np.mean(image)

version = 'back_48'
size = (*[96, ] * 3 + [1],)
lr = 1e-3
model = Estimator(lr, size, version)

general_space = np.linspace(0, 3, 100)


out = np.squeeze(np.array(list(model._estimator.predict(lambda: from_numpy(image)))))

ants.image_write(ants.from_numpy(out),  'out_dipole2.nii.gz')

exit()

image_pha = central_crop(image_pha, (160, 192, 192))
image_mag = central_crop(image_mag, (160, 192, 192))
ground = central_crop(ground, (160, 192, 192))
image_mag = np.where(image_mag > 0, 1., 0.)
image_pha *= image_mag

gamma = 267.522e6
TE = 3.6e-3
B0 = 1.5

image_pha /= TE
image_pha /= B0
image_pha /= gamma / 1e6

image = np.concatenate([image_pha, image_mag], -1)

version = 'back_noise_3'
size = (*[96, ] * 3 + [1],)
lr = 1e-3
model = Estimator(lr, size, version)

general_space = np.linspace(0, 3, 100)
for general_value in general_space:
    space = np.linspace(general_value-1, general_value, 20)
    loss = np.inf
    status = []
    best = None
    inputs = np.zeros((len(space), 160, 192, 192, 2)).astype('float32')
    for iteracion in tqdm(range(len(space))):
        inputs[iteracion] = image*space[iteracion]


    out = np.squeeze(np.array(list(model._estimator.predict(lambda: from_numpy(inputs)))))
    # mask, _, _ = open_nii_gz('MaskBrainExtracted.nii.gz')
    # mask = central_crop(mask, (160, 192, 192))
    # out = np.squeeze(out)*np.squeeze(mask)
    for position in tqdm(range(len(space))):
        _loss = np.sum(np.sqrt(np.square(np.squeeze(ground)-out[position]/space[position]*np.squeeze(image_mag))))

        if _loss < loss:
            loss = _loss
            status.append(space[position])
            new = ants.from_numpy(np.squeeze(out[position]))
            best = new
    ants.image_write(best, os.path.join(output_path, 'out_{}.nii.gz'.format(status[-1])))
    print(status)
