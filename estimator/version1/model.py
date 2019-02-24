from tensorflow.layers import conv3d, max_pooling3d, batch_normalization, dropout, conv3d_transpose
from tensorflow.nn import relu
import tensorflow as tf


def network(inputs, image_size, training=False):
    # Input seatting
    inputs = tf.reshape(inputs, (-1, image_size, image_size, image_size, 1))

    # ----------------------------------------------------------------------------------------------------------------------

    #			CONSTRACTING PART OF THE DEEPQMS

    # ----------------------------------------------------------------------------------------------------------------------

    # Layer 1
    # the default strides of tensorflow is (1,1,1) so for a more clean code it is not write it
    # the papper does not tell if they work with normalize data but as a good practise I will do it
    # I add an 1,1,1 conv3d to use less compute resources
    l1 = conv3d(inputs, 16, kernel_size=(5, 5, 5), padding='same')  # /1
    l1 = batch_normalization(l1)
    l1 = relu(l1)
    l1 = conv3d(l1, 16, kernel_size=(5, 5, 5), padding='same')  # /1
    l1 = batch_normalization(l1)
    l1 = relu(l1)

    # Layer 2
    l2 = max_pooling3d(l1, 2, 2)  # /2
    l2 = conv3d(l2, 32, kernel_size=(5, 5, 5), padding='same')  # /2
    l2 = batch_normalization(l2)
    l2 = relu(l2)
    l2 = conv3d(l2, 16, kernel_size=(1, 1, 1), padding='same')
    l2 = relu(l2)
    l2 = conv3d(l2, 32, kernel_size=(5, 5, 5), padding='same')  # /2
    l2 = batch_normalization(l2)
    l2 = relu(l2)

    # Layer 3
    l3 = max_pooling3d(l2, 2, 2)  # /4
    l3 = conv3d(l3, 64, kernel_size=(1, 1, 1), padding='same')  # /4
    l3 = relu(l3)
    l3 = conv3d(l3, 32, kernel_size=(5, 5, 5), padding='same')  # /4
    l3 = batch_normalization(l3)
    l3 = relu(l3)
    l3 = conv3d(l3, 64, kernel_size=(1, 1, 1), padding='same')  # /4
    l3 = relu(l3)
    l3 = conv3d(l3, 32, kernel_size=(5, 5, 5), padding='same')  # /4
    l3 = batch_normalization(l3)
    l3 = relu(l3)

    # Layer 4
    l4 = max_pooling3d(l3, 2, 2)
    l4 = conv3d(l4, 64, kernel_size=(5, 5, 5), padding='same')  # /8
    l4 = batch_normalization(l4)
    l4 = relu(l4)
    l4 = conv3d(l4, 32, kernel_size=(1, 1, 1), padding='same')  # /8
    l4 = batch_normalization(l4)
    l4 = relu(l4)
    l4 = conv3d(l4, 64, kernel_size=(5, 5, 5), padding='same')  # /8
    l4 = batch_normalization(l4)
    l4 = relu(l4)

    # Layer 5
    l5 = max_pooling3d(l4, 2, 2)
    l5 = conv3d(l5, 128, kernel_size=(5, 5, 5), padding='same')  # /16
    l5 = batch_normalization(l5)
    l5 = relu(l5)
    l5 = conv3d(l5, 64, kernel_size=(1, 1, 1), padding='same')  # /16
    l5 = relu(l5)
    l5 = conv3d(l5, 128, kernel_size=(5, 5, 5), padding='same')  # /16
    l5 = batch_normalization(l5)
    l5 = relu(l5)

    l5 = dropout(l5, .3, training)

    # Layer 6
    l6 = conv3d_transpose(l5, 256, [5, 5, 5], [2, 2, 2], padding='same')  # /8
    l6 = tf.concat([l4, l6], axis=-1)
    l6 = conv3d(l6, 128, kernel_size=(1, 1, 1), padding='same')
    l6 = relu(l6)
    l6 = conv3d(l6, 256, kernel_size=(5, 5, 5), padding='same')  # /8
    l6 = batch_normalization(l6)
    l6 = relu(l6)
    l6 = conv3d(l6, 128, kernel_size=(1, 1, 1), padding='same')
    l6 = relu(l6)
    l6 = conv3d(l6, 128, kernel_size=(5, 5, 5), padding='same')  # /8
    l6 = batch_normalization(l6)
    l6 = relu(l6)

    # Layer 7
    l7 = conv3d_transpose(l6, 128, [5, 5, 5], [2, 2, 2], padding='same')  # /4
    l7 = tf.concat([l3, l7], axis=-1)
    l7 = conv3d(l7, 64, kernel_size=(1, 1, 1), padding='same')
    l7 = relu(l7)
    l7 = conv3d(l7, 128, kernel_size=(5, 5, 5), padding='same')  # /4
    l7 = batch_normalization(l7)
    l7 = relu(l7)
    l7 = conv3d(l7, 64, kernel_size=(1, 1, 1), padding='same')
    l7 = relu(l7)
    l7 = conv3d(l7, 64, kernel_size=(5, 5, 5), padding='same')  # /4
    l7 = batch_normalization(l7)
    l7 = relu(l7)

    # Layer 8
    l8 = conv3d_transpose(l7, 64, [5, 5, 5], [2, 2, 2], padding='same')  # /2
    l8 = tf.concat([l2, l8], axis=-1)
    l8 = conv3d(l8, 32, kernel_size=(1, 1, 1), padding='same')
    l8 = relu(l8)
    l8 = conv3d(l8, 64, kernel_size=(5, 5, 5), padding='same')  # /2
    l8 = batch_normalization(l8)
    l8 = relu(l8)
    l8 = conv3d(l8, 32, kernel_size=(1, 1, 1), padding='same')
    l8 = relu(l8)
    l8 = conv3d(l8, 32, kernel_size=(5, 5, 5), padding='same')  # /2
    l8 = batch_normalization(l8)
    l8 = relu(l8)

    # Layer 9
    l9 = conv3d_transpose(l8, 32, [5, 5, 5], [2, 2, 2], padding='same')  # /1
    l9 = tf.concat([l1, l9], axis=-1)
    l9 = conv3d(l9, 16, kernel_size=(5, 5, 5), padding='same')  # /1
    l9 = batch_normalization(l9)
    l9 = relu(l9)
    l9 = conv3d(l9, 16, kernel_size=(5, 5, 5), padding='same')  # /1
    l9 = batch_normalization(l9)
    l9 = relu(l9)

    output = conv3d(l9, 1, (1, 1, 1), padding='same')  # /1
    return output
