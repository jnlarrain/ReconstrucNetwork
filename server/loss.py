import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from .imagenes import ImageShower


class PerceptualLoss:
    def __init__(self, shape, chn):
        self.model = ResNet50V2(weights='imagenet', include_top=False, input_shape=shape[1:-1]+[chn])
        self.model.trainable = False
        self.images = ImageShower(shape)
        c11 = self.model.get_layer('block1_conv1')
        c12 = self.model.get_layer('block1_conv2')

        c21 = self.model.get_layer('block2_conv1')
        c22 = self.model.get_layer('block2_conv2')

        c31 = self.model.get_layer('block3_conv1')
        c32 = self.model.get_layer('block3_conv2')
        c33 = self.model.get_layer('block3_conv3')

        self.c11 = Conv2D.from_config(c11.get_config())
        self.c12 = Conv2D.from_config(c12.get_config())
        self.m1 = MaxPooling2D((2, 2), strides=(2, 2))

        self.c21 = Conv2D.from_config(c21.get_config())
        self.c22 = Conv2D.from_config(c22.get_config())
        self.m2 = MaxPooling2D((2, 2), strides=(2, 2))

        self.c31 = Conv2D.from_config(c31.get_config())
        self.c32 = Conv2D.from_config(c32.get_config())
        self.c33 = Conv2D.from_config(c33.get_config())
        self.m3 = MaxPooling2D((2, 2), strides=(2, 2))

    def loss(self, labels, preds):
        lp = self.calculate(preds)
        ll = self.calculate(labels)
        loss = tf.reduce_sum(tf.pow(lp-ll, 2))
        #+ tf.reduce_sum(tf.pow(lp[1]-ll[1], 2)) + tf.reduce_sum(tf.pow(lp[2]-ll[2], 2))
        tf.summary.image('Train', self.images.show_summary(preds, labels), max_outputs=8)
        return loss + self.loss_function(labels, preds) + self.regulator(preds, 2e-5)

    def calculate(self, item):
        def inner(x):
            def _inner(y):
                y = tf.concat([y, y, y], -1)
                return self.model(tf.expand_dims(y, 0))
            return tf.map_fn(_inner, x)
        return tf.map_fn(inner, item)

    @staticmethod
    def loss_function(labels, preds):
        l2 = tf.subtract(labels, preds)
        l2 = tf.square(l2)
        l2 = tf.reduce_sum(l2)
        return l2

    @staticmethod
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

    def model(self, item):
        x1 = self.c11(item)
        x1 = self.c12(x1)
        x1 = self.m1(x1)
        x2 = self.c21(x1)
        x2 = self.c22(x2)
        x2 = self.m2(x2)
        x3 = self.c31(x2)
        x3 = self.c32(x3)
        x3 = self.c33(x3)
        x3 = self.m3(x3)
        return x3


# shape = (128, 128, 3)
# l = PerceptualLoss(shape)
# a = tf.random.normal((1, 128, 128, 128, 3))
# b = tf.random.normal((1, 128, 128, 128, 3))
# print(l(a, b))







