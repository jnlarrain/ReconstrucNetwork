import tensorflow as tf


class Loss:
    @tf.function
    def loss(self, labels, preds):
        with tf.device("cpu"):
            loss = self.l2(labels, preds) + self.l1(labels, preds) / 10 + self.regulator(preds, 2e-9)
            # loss = self.l1(labels, preds) + self.regulator(preds, 1e-9)
        return loss

    @staticmethod
    @tf.function
    def l1(labels, preds):
        with tf.device("cpu"):
            l1 = tf.abs(labels - preds)
            l1 = tf.reduce_sum(l1)
            return l1 / tf.cast(tf.size(preds), tf.float32)

    @staticmethod
    @tf.function
    def l2(labels, preds):
        with tf.device("cpu"):
            l2 = tf.subtract(labels, preds)
            l2 = tf.square(l2)
            l2 = tf.reduce_sum(l2)
            return l2 / tf.cast(tf.size(preds), tf.float32)

    @staticmethod
    @tf.function
    def gradient_variability(volume):
        with tf.device("cpu"):
            pixel_dif1 = volume[:, 1:, :, :, :] - volume[:, :-1, :, :, :]
            pixel_dif2 = volume[:, :, 1:, :, :] - volume[:, :, :-1, :, :]
            pixel_dif3 = volume[:, :, :, 1:, :] - volume[:, :, :, :-1, :]
            return pixel_dif1, pixel_dif2, pixel_dif3

    def gradient_dif(self, labels, preds):
        with tf.device("cpu"):
            pixel_dif1, pixel_dif2, pixel_dif3 = self.gradient_variability(labels)
            pixel_dif4, pixel_dif5, pixel_dif6 = self.gradient_variability(preds)
            return tf.reduce_sum(
                tf.reduce_sum(tf.abs(pixel_dif1 - pixel_dif4))
                + tf.reduce_sum(tf.abs(pixel_dif2 - pixel_dif5))
                + tf.reduce_sum(tf.abs(pixel_dif3 - pixel_dif6))
            ) / tf.cast(tf.size(preds), tf.float32)

    @tf.function
    def regulator(self, volume, alpha):
        pixel_dif1, pixel_dif2, pixel_dif3 = self.gradient_variability(volume)
        total_var = tf.reduce_sum(
            tf.reduce_sum(tf.abs(pixel_dif1)) + tf.reduce_sum(tf.abs(pixel_dif2)) + tf.reduce_sum(tf.abs(pixel_dif3))
        )
        return total_var * alpha
