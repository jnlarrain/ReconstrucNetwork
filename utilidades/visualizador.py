import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


def cuatro_cortes(volumen):
    global humbral_s, humbral_i, pos_1, pos_2, pos_3
    volumen = np.squeeze(volumen)
    fig, _axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 8))
    axs = _axs.flatten()
    ax1, ax2, ax3 = axs

    plt.subplots_adjust(left=0.2, bottom=0.4)
    shape = volumen.shape
    ax1.imshow(volumen[shape[0] // 2, :, :])
    ax1.set_title('corte 1')
    # col1 = fig.colorbar(ax1.contourf(volumen[shape[0] // 2, :, :], 255), ax=ax1)

    ax2.imshow(volumen[:, :, shape[2] // 2])
    ax2.set_title('corte 2')
    # col2 = fig.colorbar(ax2.contourf(volumen[:, :, shape[2] // 2], 255), ax=ax2)

    ax3.imshow(volumen[:, shape[1] // 2, :])
    ax3.set_title('corte 3')
    # col3 = fig.colorbar(ax3.contourf(volumen[:, shape[1]//2, :], 255), ax=ax3)

    axcolor = 'lightgoldenrodyellow'
    axkp = plt.axes([0.15, 0.2, 0.20, 0.03], facecolor=axcolor)
    axkp2 = plt.axes([0.45, 0.2, 0.20, 0.03], facecolor=axcolor)
    axki = plt.axes([0.15, 0.25, 0.20, 0.03], facecolor=axcolor)
    axki2 = plt.axes([0.45, 0.25, 0.20, 0.03], facecolor=axcolor)
    axkd = plt.axes([0.15, 0.3, 0.20, 0.03], facecolor=axcolor)

    humbral_s = Slider(axkp, 'Arriba', 1e-10, 1, valinit=1)
    humbral_i = Slider(axkp2, 'Abajo', -1, 1e-10, valinit=-1)
    pos_1 = Slider(axki, 'pos_1', 0, shape[0]-1, valinit=shape[0] // 2)
    pos_2 = Slider(axki2, 'pos_2', 0, shape[2]-1, valinit=shape[2] // 2)
    pos_3 = Slider(axkd, 'pos_3', 0, shape[1]-1, valinit=shape[1] // 2)

    def update(val):
        global humbral_s, humbral_i, pos_1, pos_2, pos_3
        new_volmen = np.zeros(volumen.shape)
        new_volmen += volumen
        new_volmen[new_volmen < humbral_i.val] = 0
        new_volmen[new_volmen > humbral_s.val] = 0

        # ax1.imshow(new_volmen[int(pos_1.val), :, :], cmap='gray')
        ax1.contour(new_volmen[int(pos_1.val), :, :], 255)
        min1 = np.min(new_volmen[int(pos_1.val), :, :])
        max1 = np.max(new_volmen[int(pos_1.val), :, :])
        # col1.set_clim(vmin=min1, vmax=max1)
        # col1.set_ticks(np.linspace(min1, max1, 10))
        # col1.draw_all()

        # ax2.imshow(new_volmen[:, :, int(pos_2.val)], cmap='gray')
        ax2.contour(new_volmen[:, :, int(pos_2.val)], 255)
        min2 = np.min(new_volmen[:, :, int(pos_2.val)])
        max2 = np.max(new_volmen[:, :, int(pos_2.val)])
        # col2.set_clim(vmin=min2, vmax=max2)
        # col2.set_ticks(np.linspace(min2, max2, 10))
        # col2.draw_all()

        # ax3.imshow(new_volmen[:, int(pos_3.val), :], cmap='gray')
        ax3.contour(new_volmen[:, int(pos_3.val), :])
        min3 = np.min(new_volmen[:, int(pos_3.val), :])
        max3 = np.max(new_volmen[:, int(pos_3.val), :])
        # col3.set_clim(vmin=min3, vmax=max3)
        # col3.set_ticks(np.linspace(min3, max3, 10))
        # col3.draw_all()

        fig.canvas.draw()

    humbral_i.on_changed(update)
    humbral_s.on_changed(update)
    pos_1.on_changed(update)
    pos_2.on_changed(update)
    pos_3.on_changed(update)
    plt.show()
