import numpy as np
import tensorflow as tf

assert tf.executing_eagerly(), 'no se esta ejecutando en eager mode'


def cart2sph(x, y, z):
    x2 = x ** 2
    y2 = y ** 2
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x2 + y2))
    r = np.sqrt(x2 + y2 + z ** 2)
    return azimuth, elevation, r


def sph2cart(azimuth, elevation, r):
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z


def cylinder(FOV, N, p1, p2, radius, xin, xout):

    FOV = tf.cast(FOV, tf.float32)
    N = tf.cast(N, tf.float32)
    p1 = tf.cast(p1, tf.float32)
    p2 = tf.cast(p2, tf.float32)
    radius = tf.cast(radius, tf.float32)
    xin = tf.cast(xin, tf.float32)
    xout = tf.cast(xout, tf.float32)

    safe_number = 1e-16
    dp = p2 - p1
    alpha, theta, na = cart2sph(dp[0], dp[1], dp[2])
    st2 = tf.sin(np.pi / 2 - theta) ** 2
    ct2 = tf.cos(np.pi / 2 - theta) ** 2
    r2 = radius ** 2



    zu = dp / (na+safe_number)
    yu = np.array([zu[1], -zu[0], 0], dtype='float32')
    yu = yu / (np.linalg.norm(yu)+safe_number)
    xu = np.array([-zu[0] * zu[2], -zu[1] * zu[2], zu[1] * zu[1] + zu[0] * zu[0]], dtype='float32')
    xu = xu / (np.linalg.norm(xu)+safe_number)

    kx = tf.linspace(1., N[0], tf.cast(N[0], tf.int32))
    ky = tf.linspace(1., N[1], tf.cast(N[1], tf.int32))
    kz = tf.linspace(1., N[2], tf.cast(N[2], tf.int32))

    kx = kx - N[0] / 2
    ky = ky - N[1] / 2
    kz = kz - N[2] / 2

    delta_kx = FOV[0] / N[0]
    delta_ky = FOV[1] / N[1]
    delta_kz = FOV[2] / N[2]

    kx = kx * delta_kx - p1[0]
    ky = ky * delta_ky - p1[1]
    kz = kz * delta_kz - p1[2]

    kx = tf.reshape(kx, [len(kx), 1, 1])
    ky = tf.reshape(ky, [1, len(ky), 1])
    kz = tf.reshape(kz, [1, 1, len(kz)])

    kx = tf.tile(kx, [1, N[1], N[2]])
    ky = tf.tile(ky, [N[0], 1, N[2]])
    kz = tf.tile(kz, [N[0], N[1], 1])

    k2 = kx ** 2 + ky ** 2 + kz ** 2

    chi = np.zeros(tf.cast(N, tf.int32), dtype='float32')

    D2 = (k2 - (zu[0] * kx + zu[1] * ky + zu[2] * kz) ** 2)
    PX = (xu[0] * kx + xu[1] * ky + xu[2] * kz)
    PY = (yu[0] * kx + yu[1] * ky + yu[2] * kz)


    C2phi = np.cos(2 * (np.arctan2(PY, PX)))
    C2phi[C2phi is None] = 0.0

    chi[D2 > r2] = xout

    chi[D2 <= r2] = xin

    safe_number = tf.cast(safe_number, tf.float32)
    C2phi = tf.cast(C2phi, tf.float32)
    st2 = tf.cast(st2, tf.float32)
    r2 = tf.cast(r2, tf.float32)
    D2 = tf.cast(D2, tf.float32)
    ct2 = tf.cast(ct2, tf.float32)

    dX = xin-xout
    Bnuc = (1-chi*2/3)

    Bmac = 0.5*r2*dX*st2*C2phi/(D2+safe_number)
    Bmac = np.cast['float32'](Bmac)
    Bmac[D2 <= r2] = dX*(3*ct2 - 1)/6

    Bnuc = Bmac*Bnuc
    return chi, Bnuc

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from time import time
    t = time()
    p1 = np.array([4, 1, 4])
    p2 = np.array([1, 2, 4])
    a = cylinder([208, ] * 3, [208, ] * 3, p1, p2, 10, 4.5786e-06, 0)[1]
    # for i in a:
    #     plt.figure()
    #     plt.imshow(i, cmap='gray')
    # plt.show()
    print(time()-t)