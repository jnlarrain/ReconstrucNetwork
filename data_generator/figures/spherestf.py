import numpy as np
from random import randint
import tensorflow as tf

assert tf.executing_eagerly(), 'no se esta ejecutando en eager mode'


def sphere(FOV, N, center, radius, xin, xout):
    '''
  device='CPU'; T in [DT_FLOAT]; Tidx in [DT_INT32]
  device='CPU'; T in [DT_FLOAT]; Tidx in [DT_INT64]
  device='CPU'; T in [DT_DOUBLE]; Tidx in [DT_INT32]
  device='CPU'; T in [DT_DOUBLE]; Tidx in [DT_INT64]
  device='GPU'; T in [DT_FLOAT]; Tidx in [DT_INT32]
  device='GPU'; T in [DT_FLOAT]; Tidx in [DT_INT64]
  device='GPU'; T in [DT_DOUBLE]; Tidx in [DT_INT32]
  device='GPU'; T in [DT_DOUBLE]; Tidx in [DT_INT64]
    
    '''
    FOV = tf.cast(FOV, tf.float32)
    N = tf.cast(N, tf.float32)
    center = tf.cast(center, tf.float32)
    radius = tf.cast(radius, tf.float32)
    xin = tf.cast(xin, tf.float32)
    xout = tf.cast(xout, tf.float32)

    kx = tf.linspace(1., N[0], tf.cast(N[0], tf.int32)) - center[0]
    ky = tf.linspace(1., N[1], tf.cast(N[1], tf.int32)) - center[1]
    kz = tf.linspace(1., N[2], tf.cast(N[2], tf.int32)) - center[2]

    delta_kx = FOV[0] / N[0]
    delta_ky = FOV[1] / N[1]
    delta_kz = FOV[2] / N[2]

    kx *= delta_kx
    ky *= delta_ky
    kz *= delta_kz

    kx = tf.reshape(kx, [len(kx), 1, 1])
    ky = tf.reshape(ky, [1, len(ky), 1])
    kz = tf.reshape(kz, [1, 1, len(kz)])

    kx = tf.tile(kx, [1, N[1], N[2]])
    ky = tf.tile(ky, [N[0], 1, N[2]])
    kz = tf.tile(kz, [N[0], N[1], 1])

    k2 = kx ** 2 + ky ** 2 + kz ** 2
    chi = np.zeros(tf.cast(N, tf.int32), dtype='float32')

    chi[k2 > radius ** 2] = xout
    chi[k2 <= radius ** 2] = xin

    dX = xin - xout
    Bnuc = 1 - chi * 2 / 3
    aux = ((3 + dX) * k2 ** (5 / 2))
    aux = np.cast['float32'](aux)
    aux[aux == 0] = 1e-25
    Bmac = (radius ** 3) * dX * (2 * kz ** 2 - kx ** 2 - ky ** 2) / aux
    Bmac = np.cast['float32'](Bmac)
    Bmac[k2 <= radius ** 2] = 0
    Bnuc *= Bmac

    # q = radius * tf.sqrt((kx ** 2) / N[0] ** 2 + (ky ** 2) / N[1] ** 2 + kz ** 2 / N[2] ** 2)
    # p = np.pi * 2 * q
    # Xs = -p * tf.cos(p) + tf.sin(p)
    # aux = (2 * np.pi ** 2 * q ** 3)
    # aux = np.cast['float32'](aux)
    # aux[aux == 0] = 1e-25
    # Xs = xin * Xs / aux
    # Xs = radius ** 3 * Xs / tf.sqrt(N[0] * N[2] * N[1])

    return chi, Bnuc, None  # Xs


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # a = sphere([384, ] * 3, [384, ] * 3, [randint(40, 70) for _ in range(3)], 28, 4.5786e-06, 0)[2]
    # plt.imshow(a[:, 50, :], cmap='gray')
    # plt.show()
    from time import time
    from tqdm import tqdm
    t = time()
    for _ in tqdm(range(2)):
        a = sphere([208*3, ] * 3, [208*3, ] * 3, [randint(40, 70) for _ in range(3)], 28, 4.5786e-06, 0)[1]
    print(time()-t)




