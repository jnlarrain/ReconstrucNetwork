import numpy as np
from random import randint


def sphere(FOV, N, center, radius, xin, xout):
    kx = np.linspace(1, N[0], N[0]) - center[0]
    ky = np.linspace(1, N[1], N[1]) - center[1]
    kz = np.linspace(1, N[2], N[2]) - center[2]

    delta_kx = FOV[0] / N[0]
    delta_ky = FOV[1] / N[1]
    delta_kz = FOV[2] / N[2]

    kx *= delta_kx
    ky *= delta_ky
    kz *= delta_kz

    kx = np.reshape(kx, [len(kx), 1, 1])
    ky = np.reshape(ky, [1, len(ky), 1])
    kz = np.reshape(kz, [1, 1, len(kz)])

    kx = np.tile(kx, [1, N[1], N[2]])
    ky = np.tile(ky, [N[0], 1, N[2]])
    kz = np.tile(kz, [N[0], N[1], 1])

    k2 = kx ** 2 + ky ** 2 + kz ** 2
    chi = np.zeros(N)
    chi[k2 > radius ** 2] = xout
    chi[k2 <= radius ** 2] = xin

    dX = xin - xout
    Bnuc = 1 - chi * 2 / 3
    aux = ((3 + dX) * k2 ** (5 / 2))
    aux[aux == 0] = 1e-25
    Bmac = (radius ** 3) * dX * (2 * kz ** 2 - kx ** 2 - ky ** 2) / aux
    Bmac[k2 <= radius ** 2] = 0
    Bnuc *= Bmac

    q = radius * np.sqrt((kx ** 2) / N[0] ** 2 + (ky ** 2) / N[1] ** 2 + kz ** 2 / N[2] ** 2)
    p = np.pi * 2 * q
    Xs = -p * np.cos(p) + np.sin(p)
    aux = (2 * np.pi ** 2 * q ** 3)
    aux[aux == 0] = 1e-25
    Xs = xin * Xs / aux
    Xs = radius ** 3 * Xs / np.sqrt(N[0] * N[2] * N[1])

    return chi, Bnuc, Xs


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    a = sphere([128, ] * 3, [128, ] * 3, [randint(40, 70) for _ in range(3)], 28, 4.5786e-06, 0)[2]
    plt.imshow(a[:, 50, :], cmap='gray')
    plt.show()




