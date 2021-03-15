import tensorflow as tf


def calculate_k(FOV, N, center=(0, 0, 0), points=(0, 0, 0)):

    kx = tf.cast(tf.linspace(1., N[0], N[0]) - center[0], tf.float32)
    ky = tf.cast(tf.linspace(1., N[1], N[1]) - center[1], tf.float32)
    kz = tf.cast(tf.linspace(1., N[2], N[2]) - center[2], tf.float32)

    delta_kx = tf.cast(FOV[0] / N[0], tf.float32)
    delta_ky = tf.cast(FOV[1] / N[1], tf.float32)
    delta_kz = tf.cast(FOV[2] / N[2], tf.float32)

    kx = delta_kx*kx - points[0]
    ky = delta_ky*ky - points[1]
    kz = delta_kz*kz - points[2]

    kx = tf.cast(tf.reshape(kx, [kx.shape[0], 1, 1]), tf.float32)
    ky = tf.cast(tf.reshape(ky, [1, ky.shape[0], 1]), tf.float32)
    kz = tf.cast(tf.reshape(kz, [1, 1, kz.shape[0]]), tf.float32)

    kx = tf.cast(tf.tile(kx, [1, N[1], N[2]]), tf.float32)
    ky = tf.cast(tf.tile(ky, [N[0], 1, N[2]]), tf.float32)
    kz = tf.cast(tf.tile(kz, [N[0], N[1], 1]), tf.float32)

    k2 = tf.cast(kx ** 2 + ky ** 2 + kz ** 2, tf.float32)
    return k2, kx, ky, kz




