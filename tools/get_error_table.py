import os

import numpy as np
from tqdm import tqdm
from figure_tools.scale_figure import save_figure
import ants

sizes = [32, 48, 64, 96, 128, 256]
radios = [True, False]
SNR = [128, 256, 512, 1024]


model_version = "48_res_paper_41"


def open_nii_gz(path):
    data = ants.image_read(path)
    return np.squeeze(data.numpy()), data


def get_error(size, snr, radio):
    input_path = f"different_scale//{size}_{radio}"
    new, original_data = open_nii_gz(
        os.path.join(input_path, f"output_{snr}_{model_version}.nii")
    )
    GT, _ = open_nii_gz(os.path.join(input_path, "sus.nii.gz"))
    delta = GT - new
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
    save_figure(figure, f"output_{size}_{radio}_{snr}_{model_version}.png")
    return np.sum((new - GT) ** 2) / np.size(new)


errors = {}

for size in sizes:
    errors[size] = {}

    for scale_radio in radios:
        if scale_radio:
            radio = size // 4
        else:
            radio = 10
        errors[size][radio] = {}
        for snr in SNR:
            errors[size][radio][snr] = get_error(size, snr, radio)


for size in tqdm(sizes):
    for scale_radio in radios:
        if scale_radio:
            radio = size // 4
        else:
            radio = 10
        for snr in SNR:
            print(size, snr, radio, errors[size][radio][snr])
