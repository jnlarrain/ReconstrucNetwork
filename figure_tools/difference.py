import numpy as np


def max_side_padding(img):
    x, y, z = img.shape
    max_size = max([x, y, z])
    dx = max([0, max_size - x])
    dy = max([0, max_size - y])
    dz = max([0, max_size - z])
    img = np.pad(
        img,
        [
            [dx // 2, dx // 2 + dx % 2],
            [dy // 2, dy // 2 + dy % 2],
            [dz // 2, dz // 2 + dz % 2],
        ],
    )
    return img
