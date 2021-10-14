import tensorflow as tf

class Loss:
    @tf.function
    def loss(self, labels, preds):
        loss = self.l2(labels, preds) + self.regulator(preds, 2e-9) + self.l1(labels, preds)/1e4
        return loss

    @staticmethod
    @tf.function
    def l1(labels, preds):
        return tf.reduce_sum(tf.abs(labels-preds)) / tf.cast(tf.size(preds), tf.float32)

    @staticmethod
    @tf.function
    def l2(labels, preds):
        l2 = tf.subtract(labels, preds)
        l2 = tf.square(l2)
        l2 = tf.reduce_sum(l2)
        return l2 / tf.cast(tf.size(preds), tf.float32)

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
