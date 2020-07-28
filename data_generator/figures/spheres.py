import numpy as np
from random import randint
from time import  time

def sphere(FOV, N, center, radius, xin, xout):
    '''
    
    
    '''
    kx = np.linspace(1, N[0], N[0], dtype='float32') - center[0]
    ky = np.linspace(1, N[1], N[1], dtype='float32') - center[1]
    kz = np.linspace(1, N[2], N[2], dtype='float32') - center[2]

    delta_kx = FOV[0] / N[0]
    delta_ky = FOV[1] / N[1]
    delta_kz = FOV[2] / N[2]

    kx *= delta_kx
    ky *= delta_ky
    kz *= delta_kz

    kx = np.reshape(kx, [N[0], 1, 1])
    ky = np.reshape(ky, [1, N[1], 1])
    kz = np.reshape(kz, [1, 1, N[2]])

    kx = np.tile(kx, [1, N[1], N[2]])
    ky = np.tile(ky, [N[0], 1, N[2]])
    kz = np.tile(kz, [N[0], N[1], 1])

    k2 = np.square(kx) + np.square(ky) + np.square(kz)
    chi = np.zeros(N, dtype='float32')
    chi[k2 > radius ** 2] = xout
    chi[k2 <= radius ** 2] = xin

    dX = xin - xout
    Bnuc = 1 - chi * 2 / 3
    aux = ((3 + dX) * k2 ** (5 / 2))
    aux[aux == 0] = 1e-25
    Bmac = (radius ** 3) * dX * (2 * kz - np.square(kx) - np.square(ky)) / aux
    Bmac[k2 <= radius ** 2] = 0
    Bnuc *= Bmac

    # q = radius * np.sqrt(np.square(kx) / N[0] ** 2 + np.square(ky) / N[1] ** 2 + np.square(kz) / N[2] ** 2,
    # dtype='float32')
    # p = np.pi * 2 * q
    # Xs = -p * np.cos(p, dtype='float32') + np.sin(p, dtype='float32')
    # aux = (2 * np.pi ** 2 * q ** 3)
    # aux[aux == 0] = 1e-25
    # Xs = xin * Xs / aux
    # Xs = radius ** 3 * Xs / np.sqrt(N[0] * N[2] * N[1], dtype='float32')

    return chi, Bnuc, None  # Xs


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    t1 = time()
    for _ in tqdm(range(1)):
        a = sphere([208, ] * 3, [208, ] * 3, [randint(40, 70) for _ in range(3)], 28, 4.5786e-06, 0)[2]
    print(time() - t1)
