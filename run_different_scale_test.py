import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from tqdm import tqdm
from model.estimator import Estimator
import tensorflow as tf
import ants

physical_devices = tf.config.list_physical_devices("GPU")
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)


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
    return np.expand_dims(data.numpy(), [0, -1]), data


def from_numpy(elements):
    data = tf.data.Dataset.from_tensor_slices((elements, elements))
    data = data.map(lambda x, y: (tf.cast(x, tf.float32), y))
    data = data.batch(1)
    return data


def make_inference(model, size, snr, radio):
    input_path = f"different_scale//{size}_{radio}"
    image_pha, original_data = open_nii_gz(os.path.join(input_path, "phase.nii.gz"))
    image_mag, _ = open_nii_gz(os.path.join(input_path, "sus.nii.gz"))
    sus, _ = open_nii_gz(os.path.join(input_path, "sus.nii.gz"))

    image_mag = np.where(image_mag > 0, 1.0, 0.0)
    image_pha += np.random.normal(
        scale=np.max(np.abs(sus)) / snr, size=(1, size, size, size, 1)
    )

    # image_pha *= image_mag
    image = np.concatenate([image_pha, image_mag], -1)
    out = np.squeeze(
        np.array(list(model._estimator.predict(lambda: from_numpy(image))))
    )
    out *= np.squeeze(image_mag)
    ants.image_write(
        original_data.new_image_like(out),
        os.path.join(input_path, f"output_{snr}_{model_version}.nii"),
    )


sizes = [32, 48, 64, 96, 128, 256]
radios = [True, False]
SNR = [128, 256, 512, 1024]

num_cilindros = 0
num_esferas = 1  # 128
susceptibilidad_interna = 1e-7  # 1e-8
susceptibilidad_externa = 0  # 9 * 1e-6
susceptibilidad_foco_externo = 1e-1

model_version = "48_res_paper_41"
# model_version = "res_noise_norm_4_backupd_2"
size = (
    *[
        96,
    ]
    * 3
    + [1],
)
lr = 1e-3
model = Estimator(lr, size, model_version)
bias = 0
threads = []
spheres_number = 1
cylinder_number = 0
for size in tqdm(sizes):
    for snr in SNR:
        for scale_radio in radios:
            if scale_radio:
                radio = size // 4
            else:
                radio = 10

            if not os.path.exists(f"{size}_{radio}"):
                os.mkdir(f"{size}_{radio}")

            make_inference(model, size, snr, radio)
