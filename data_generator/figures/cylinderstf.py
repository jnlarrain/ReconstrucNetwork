import numpy as np
import tensorflow as tf


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


def cylinder(k, N, p1, p2, radius, xin, xout):
    k2, kx, ky, kz = k
    safe_number = tf.cast(1e-16, tf.float32)
    dp = tf.cast(p2 - p1, tf.float32)
    alpha, theta, na = tf.cast(cart2sph(dp[0], dp[1], dp[2]), tf.float32)
    st2 = tf.cast(tf.sin(np.pi / 2 - theta) ** 2, tf.float32)
    ct2 = tf.cast(tf.cos(np.pi / 2 - theta) ** 2, tf.float32)
    r2 = tf.cast(radius ** 2, tf.float32)

    zu = tf.cast(dp / (na+safe_number), tf.float32)
    yu = np.array([zu[1], -zu[0], 0], dtype='float32')
    yu = tf.cast(yu / (np.linalg.norm(yu)+safe_number), tf.float32)
    xu = np.array([-zu[0] * zu[2], -zu[1] * zu[2], zu[1] * zu[1] + zu[0] * zu[0]], dtype='float32')
    xu = tf.cast(xu / (np.linalg.norm(xu)+safe_number), tf.float32)

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

    Bmac = tf.cast(0.5*r2*dX*st2*C2phi/(D2+safe_number), tf.float32)
    Bmac = tf.where(D2 <= r2,  tf.cast(dX*(3*ct2 - 1)/6, tf.float32), Bmac)

    Bnuc = Bmac*Bnuc
    return tf.cast(chi, tf.float32), tf.cast(Bnuc, tf.float32)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from K_space import calculate_k
    from time import time
    p1 = np.array([4, 1, 4])
    p2 = np.array([30, 50, 40])
    k = (*calculate_k([256, ]*3, [256, ]*3, points=p1), )
    a = cylinder(k, [256, ]*3, p1, p2, 10, 4.5786e-06, 0)[1]
    print(np.max(np.abs(a)))
    # plt.hist(np.abs*np.ravel(a), bins=1000)
    # plt.show()
    for i in a[::8]:
        plt.figure()
        plt.imshow(i, cmap='gray')
    plt.show()
