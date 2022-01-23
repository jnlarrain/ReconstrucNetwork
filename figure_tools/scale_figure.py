import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_scale_figure(img):
    img = np.squeeze(img)
    fig, _axs = plt.subplots(nrows=1, ncols=1)
    _axs.imshow(img, cmap="gray")
    _axs.axis("off")
    divider = make_axes_locatable(_axs)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    pcm = _axs.pcolormesh(img, cmap="gray")
    fig.colorbar(pcm, cax=cax)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_arr = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img_arr


def save_figure(img, path):
    plt.imsave(path, get_scale_figure(img), cmap="gray")
