import matplotlib.pyplot as plt
import tensorflow as tf
import io


class Image_shower:
    def __init__(self, shape):
        self.shape = shape

    def show_image_cuts(self, image):
        cut_1 = tf.image.resize(image[:, self.shape[0] // 2, :, :, :], self.shape[:2])
        cut_2 = tf.image.resize(image[:, :, self.shape[1] // 2, :, :], self.shape[:2])
        cut_3 = tf.image.resize(image[:, :, :, self.shape[2] // 2, :], self.shape[:2])
        result = tf.concat([cut_1, cut_2, cut_3], 1)
        return result

    def show_summary(self, images, labels):
        diff = labels - images
        new_images = self.show_image_cuts(images)
        new_diff = self.show_image_cuts(diff)
        new_labels = self.show_image_cuts(labels)
        result = tf.concat([new_labels[:, :, :, :], new_images[:, :, :, :],
                            new_diff[:, :, :, :]], 2)
        return result

    @staticmethod
    def to_int(image):
        return tf.cast((image-tf.reduce_min(image))/tf.reduce_max(image)*(2**16-1), tf.uint16)

    def show_difference(self, images, labels):
        diff = labels - images
        cuts = self.show_image_cuts(diff)
        def get_figure(cuts):
            with tf.train.SessionRunHook:
                cuts = cuts.eval()
                figure = plt.figure()
                img = plt.imshow(tf.image.encode_png(self.to_int(cuts)))
                # plt.imshow(img, cmap='gray')
                plt.colorbar()
                plt.tight_layout()
                buf2 = io.BytesIO()
                plt.savefig(buf2, format='png')
                plt.close(figure)
                buf2.seek(0)
                image = tf.image.decode_png(buf2.getvalue(), channels=4)
                image = tf.expand_dims(image, 0)
            return image
        return tf.map_fn(get_figure, cuts)

