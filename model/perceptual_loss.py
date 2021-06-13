import tensorflow as tf

class PerceptualLoss:
    # def __init__(self, model='model/loss_model/'):
    #     self.model = tf.saved_model.load(model)
    #     self.model.trainable = False

    @staticmethod
    def norm_image(labels, preds):
        minimo = tf.reduce_min(labels)
        _labels = labels - minimo
        _preds = preds - minimo
        maximo = tf.reduce_max(_labels)
        _labels = _labels/maximo
        _preds = _preds/maximo
        return _labels, _preds

    # @tf.function
    # def dipole_reg(self, preds, dipole, entrance):
    #     def inner_reg(_pred, _entrance):
    #         def fourier_shift(image):
    #             image = tf.cast(image, tf.complex64)
    #             img = tf.signal.fft3d(image)
    #             return img
    #
    #         def fourier_inverse(fourier):
    #             fourier = tf.cast(fourier, tf.complex64)
    #             img = tf.math.real(tf.signal.ifft3d(fourier))
    #             return img
    #         _pred = tf.squeeze(_pred)
    #         _entrance = tf.squeeze(_entrance)
    #         final_prediction = fourier_inverse(fourier_shift(_pred) * tf.cast(tf.squeeze(dipole), tf.complex64))
    #         return self.l2(tf.cast(final_prediction, tf.float32), _entrance)
    #     loss = tf.reduce_sum(tf.map_fn(lambda x: inner_reg(preds[int(x)], entrance[int(x), 0]),
    #                                    tf.cast(tf.linspace(0, 3, 4), tf.int8)))/4
    #     return loss



    @tf.function
    def loss(self, labels, preds, entrance):
        # labels, preds = self.norm_image(labels, preds)
        # [loss] = tf.py_function(self.perceptual, [self, labels, preds], [tf.float32])
        loss = self.l2(labels, preds) #+ self.l1(labels, preds)/100  # + self.regulator(preds, 2e-7)
        # loss += self.dipole_reg(preds, labels[..., 1], entrance[..., 0])
        # loss += self.perceptual(labels, preds)
        return loss

    @staticmethod
    @tf.function
    def l1(labels, preds):
        return tf.reduce_sum(tf.abs(labels-preds)) / tf.cast(tf.size(preds), tf.float32)

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
    def l2(labels, preds):
        l2 = tf.subtract(labels, preds)
        l2 = tf.square(l2)
        # l2 = tf.sqrt(l2)
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
