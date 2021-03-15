import tensorflow as tf


def sphere(k, radius, xin, xout):
    k2, kx, ky, kz = k
    radius = tf.cast(radius, tf.float32)
    xin = tf.cast(xin, tf.float32)
    xout = tf.cast(xout, tf.float32)

    # calcula la susceptibildad
    chi = tf.cast(tf.where(k2 > radius ** 2, xout, xin), tf.float32)

    # calculo del la phase o campo
    dX = tf.cast(xin - xout, tf.float32)
    Bnuc = tf.cast(1 - chi * 2 / 3, tf.float32)
    aux = tf.cast(((3 + dX) * k2 ** (5 / 2)), tf.float32)
    aux = tf.cast(tf.where(aux == 0, tf.cast(1e-20, tf.float32), aux), tf.float32)
    Bmac = tf.cast((radius ** 3) * dX * (2 * kz ** 2 - kx ** 2 - ky ** 2) / aux, tf.float32)
    Bmac = tf.cast(tf.where(k2 <= radius ** 2, tf.cast(0, tf.float32), Bmac), tf.float32)
    Bnuc *= tf.cast(Bmac, tf.float32)
    return tf.cast(chi, tf.float32), tf.cast(Bnuc, tf.float32), None  # Xs







