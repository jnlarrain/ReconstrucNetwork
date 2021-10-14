import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow as tf
import ants
import numpy as np
from model.estimator import Estimator
from data_generator.dipole import dipole_kernel


def central_crop(image, size):
    def delta(size1, size2):
        diff = np.abs(size1 - size2)
        _delta = diff // 2
        return _delta

    image = np.squeeze(image)
    shape = image.shape
    print(shape)
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


for i in range(1, 3):
    for j in range(1, 3):
        input_path = f'D:\\files\\ReconstrucNetwork\\SNR1\\Sim{i}Snr{j}'
        image_pha, _ = open_nii_gz(os.path.join(input_path, 'phase_compuesta.nii.gz'))
        image_mag, _ = open_nii_gz(os.path.join(input_path, 'MaskBrainExtracted.nii.gz'))
        ground, _ = open_nii_gz(os.path.join(input_path, 'GT', 'Chi.nii.gz'))


        ants.image_write(ants.from_numpy(np.squeeze(central_crop(ground, (160, 192, 192)))),
                         os.path.join(input_path, 'ground.nii.gz'))

        ants.image_write(ants.from_numpy(np.squeeze(ground)),
                         os.path.join(input_path, 'ground_pad.nii.gz'))

        def Fourier_shift(image):
            img = np.fft.fftn(image)
            return img


        def Fourier_inverse(fourier):
            img = np.real(np.fft.ifftn(fourier))
            return img

        TE = 16e-3     # 3.6e-3
        B0 = 7     # 1.5

        image = np.pad(np.squeeze(ground), [[48, 48], [48, 48], [48, 48]])
        dipole = dipole_kernel(image.shape, [9e-6, 9e-6, 9e-6])
        # dipole = np.pad(dipole, [[48, 48], [48, 48], [48, 48]])



        #
        # exit()
        final = Fourier_inverse(Fourier_shift(image)*dipole)
        ants.image_write(ants.from_numpy(final), 'dipole_data.nii.gz')

        # image_pha = np.squeeze(image_pha)
        # image_pha = central_crop(image_pha, (160, 192, 192))
        image_pha = central_crop(final, (160, 192, 192))
        image_mag = central_crop(image_mag, (160, 192, 192))
        image_mag = np.where(image_mag > 0, 1., 0.)


        # gamma = 267.522


        # image_pha *= TE
        # image_pha /= B0
        # image_pha /= gamma

        # ants.image_write(ants.from_numpy(np.squeeze(image_pha)), 'input_phase.nii.gz')

        image = np.concatenate([image_pha, image_mag], -1)

        # version = '2_48'
        version = 'res_normal_1'
        size = (*[96, ] * 3 + [1],)
        lr = 1e-3
        model = Estimator(lr, size, version)


        out = np.squeeze(np.array(list(model._estimator.predict(lambda: from_numpy(image)))))
        out *= np.squeeze(image_mag)

        out = np.pad(out, [[2, 2], [6, 7], [6, 7]])

        image_pha, _ = open_nii_gz(os.path.join(input_path, 'Frequency.nii.gz'))

        image_pha *= TE
        image_pha /= B0

        image_pha = central_crop(image_pha, (160, 192, 192))
        image_pha *= image_mag

        image = np.concatenate([image_pha, image_mag], -1)

        out = np.squeeze(np.array(list(model._estimator.predict(lambda: from_numpy(image)))))
        out *= np.squeeze(image_mag)

        out = np.pad(out, [[2, 2], [6, 7], [6, 7]])

        ants.image_write(ants.from_numpy(out), os.path.join(input_path, 'output.nii'))

        ants.image_write(ants.from_numpy(out),  os.path.join(input_path, 'output2.nii'))

        image_pha, _ = open_nii_gz(os.path.join(input_path, 'phase_compuesta.nii.gz'))

        # image_pha *= TE
        # image_pha /= B0

        image_pha = central_crop(image_pha, (160, 192, 192))
        image_pha *= image_mag


        image = np.concatenate([image_pha, image_mag], -1)

        out = np.squeeze(np.array(list(model._estimator.predict(lambda: from_numpy(image)))))
        out *= np.squeeze(image_mag)

        out = np.pad(out, [[2, 2], [6, 7], [6, 7]])

        ants.image_write(ants.from_numpy(out),  os.path.join(input_path, 'output3.nii'))