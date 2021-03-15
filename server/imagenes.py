import tensorflow as tf


class ImageShower:
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
