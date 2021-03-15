import tensorflow as tf


class PerceptualLoss:
    def __init__(self, model='model/loss_model/'):
        self.model = tf.saved_model.load(model)
        self.model.trainable = False

    @tf.function
    def loss(self, labels, preds):
        # [loss] = tf.py_function(self.perceptual, [self, labels, preds], [tf.float32])
        loss = self.perceptual(labels, preds)
        return loss + self.loss_function(labels, preds) + self.regulator(preds, 2e-5)

    @tf.function
    def perceptual(self, labels, preds):
        lp = tf.cast(self.calculate(preds), tf.float32)
        ll = tf.cast(self.calculate(labels), tf.float32)
        loss = tf.reduce_sum(tf.pow(lp-ll, 2.))
        return loss/tf.cast(tf.size(lp), tf.float32)

    @tf.function
    def calculate(self, item):
        def inner(x):
            def _inner(y):
                y = tf.concat([y, y, y], -1)
                return self.model.signatures["serving_default"](tf.expand_dims(y, 0))['conv2_block3_out']
            return tf.compat.v1.map_fn(_inner, x)
        return tf.compat.v1.map_fn(inner, item)

    @staticmethod
    @tf.function
    def loss_function(labels, preds):
        l2 = tf.subtract(labels, preds)
        l2 = tf.square(l2)
        l2 = tf.reduce_sum(l2)
        return l2/tf.cast(tf.size(labels), tf.float32)

    @staticmethod
    @tf.function
    def regulator(volumen, alpha=1e-7):
        pixel_dif1 = volumen[:, 1:, :, :, :] - volumen[:, :-1, :, :, :]
        pixel_dif2 = volumen[:, :, 1:, :, :] - volumen[:, :, :-1, :, :]
        pixel_dif3 = volumen[:, :, :, 1:, :] - volumen[:, :, :, :-1, :]
        total_var = tf.reduce_sum(
            tf.reduce_sum(tf.abs(pixel_dif1)) +
            tf.reduce_sum(tf.abs(pixel_dif2)) +
            tf.reduce_sum(tf.abs(pixel_dif3))
        )
        return total_var * alpha
