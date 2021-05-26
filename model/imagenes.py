import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class ImageShower:
    def __init__(self, shape):
        self.shape = shape

    def show_image_cuts(self, image):
        with tf.device('cpu:0'):
            cut_1 = tf.image.resize(image[:, self.shape[0] // 2, :, :, :], self.shape[:2])
            cut_2 = tf.image.resize(image[:, :, self.shape[1] // 2, :, :], self.shape[:2])
            cut_3 = tf.image.resize(image[:, :, :, self.shape[2] // 2, :], self.shape[:2])
            result = tf.concat([cut_1, cut_2, cut_3], 1)
            return result

    def show_summary(self, images, labels):
        with tf.device('cpu:0'):
            diff = labels - images
            new_images = self.show_image_cuts(images)
            new_diff = self.show_image_cuts(diff)
            new_labels = self.show_image_cuts(labels)
            result = tf.concat([new_labels[:, :, :, :], new_images[:, :, :, :],
                                new_diff[:, :, :, :]], 2)
            return tf.map_fn(self.plot_image, result)
        # return  result

    @staticmethod
    def to_int(image):
        return tf.cast((image-tf.reduce_min(image))/tf.reduce_max(image)*(2**16-1), tf.uint16)

    def plot_image(self, image):
        def _plot_buffer(img):
            with tf.device('cpu:0'):
                if not all(img.shape):
                    img = tf.random.uniform([self.shape[0]*3, self.shape[1]*3])
                img = np.squeeze(img)
                fig, _axs = plt.subplots(nrows=1, ncols=1)
                fig.set_size_inches(8, 8)
                _axs.imshow(img, cmap='gray')
                _axs.axis('off')
                pcm = _axs.pcolormesh(img, cmap='gray')
                fig.colorbar(pcm, ax=_axs)
                fig.canvas.draw()
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img_arr = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close(fig)
                return img_arr

        [image_numpy] = tf.py_function(_plot_buffer, [image, ], [tf.float32])
        return image_numpy
