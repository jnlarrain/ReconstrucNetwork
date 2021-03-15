import numpy as np
from random import randint


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
    safe_number = 1e-16
    dp = p2 - p1
    alpha, theta, na = cart2sph(dp[0], dp[1], dp[2])
    st2 = np.sin(np.pi / 2 - theta) ** 2
    ct2 = np.cos(np.pi / 2 - theta) ** 2
    r2 = radius ** 2

    zu = dp / (na+safe_number)
    yu = np.array([zu[1], -zu[0], 0], dtype='float32')
    yu = yu / (np.linalg.norm(yu)+safe_number)
    xu = np.array([-zu[0] * zu[2], -zu[1] * zu[2], zu[1] * zu[1] + zu[0] * zu[0]], dtype='float32')
    xu = xu / (np.linalg.norm(xu)+safe_number)

    chi = np.zeros(N)

    D2 = (k2 - (zu[0] * kx + zu[1] * ky + zu[2] * kz) ** 2)
    PX = (xu[0] * kx + xu[1] * ky + xu[2] * kz)
    PY = (yu[0] * kx + yu[1] * ky + yu[2] * kz)
    C2phi = np.cos(2 * (np.arctan2(PY, PX)))
    C2phi[C2phi is None] = 0.0

    chi[D2 > r2] = xout

    chi[D2 <= r2] = xin

    dX = xin-xout
    Bnuc = (1-chi*2/3)
    Bmac = 0.5*r2*dX*st2*C2phi/(D2+safe_number)
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