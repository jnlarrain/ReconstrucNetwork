import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import ants
import numpy as np
from model.estimator import Estimator
import time

from figure_tools.difference import max_side_padding
from figure_tools.scale_figure import save_figure


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
    image = image[dx : size[0] + dx, dy : size[1] + dy, dz : size[2] + dz]
    image = np.expand_dims(image, [0, -1])
    return image


def open_nii_gz(path):
    data = ants.image_read(path)
    return np.expand_dims(data.numpy().astype("float32"), [0, -1]), data


def from_numpy(elements):
    data = tf.data.Dataset.from_tensor_slices((elements, elements))
    data = data.map(lambda x, y: (tf.cast(x, tf.float32), y))
    data = data.batch(1)
    return data


# model_version = "res_noise_norm_4_backupd_2"
model_version = "48_res_paper_41"
experiment = "2"
size = (
    *[
        96,
    ]
    * 3
    + [1],
)
lr = 1e-3
model = Estimator(lr, size, model_version)

for i in range(1, 3):
    input_path = f"data_phase"
    image_pha, original_data = open_nii_gz(
        os.path.join(input_path, str(i), "phase.nii")
    )
    image_mag, _ = open_nii_gz(os.path.join(input_path, str(i), "mask.nii"))

    image_mag = np.where(image_mag > 0, 1.0, 0.0)

    # image_pha /= 42.58 * 7
    # image_pha = central_crop(image_pha, (160, 192, 192))
    # image_mag = central_crop(image_mag, (160, 192, 192))
    image_pha *= image_mag

    image = np.concatenate([image_pha, image_mag], -1)
    t = time.time()
    out = np.squeeze(
        np.array(list(model._estimator.predict(lambda: from_numpy(image))))
    )
    out *= np.squeeze(image_mag)
    # out = np.pad(out, [[2, 2], [6, 7], [6, 7]])
    print(time.time() - t)

    challenge_data = ants.image_read("challenge\\Sim1Snr1\\GT\\chi.nii.gz")
    GT = ants.image_read("challenge\\Sim1Snr1\\GT\\chi.nii.gz")
    out = out[:164, :205, :205]

    ants.image_write(
        challenge_data.new_image_like(out),
        os.path.join(
            input_path,
            str(i),
            f"output_noise_{model_version}-{experiment}.nii",
        ),
    )
    delta = GT - challenge_data
    new = max_side_padding(challenge_data)
    GT = max_side_padding(GT)
    delta = max_side_padding(delta)

    x, y, z = new.shape
    figure = np.concatenate(
        [
            np.concatenate([new[x // 2, :, :], new[:, y // 2, :], new[:, :, z // 2]]),
            np.concatenate([GT[x // 2, :, :], GT[:, y // 2, :], GT[:, :, z // 2]]),
            np.concatenate(
                [delta[x // 2, :, :], delta[:, y // 2, :], delta[:, :, z // 2]]
            ),
        ],
        1,
    )
    figure -= np.min(figure)
    figure /= np.max(figure)
    figure *= 0.5
    figure -= 0.25
    save_figure(
        figure,
        os.path.join(
            input_path,
            str(i),
            f"output_noise_{model_version}-{experiment}.png",
        ),
    )