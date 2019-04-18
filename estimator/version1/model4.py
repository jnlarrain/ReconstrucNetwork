from tensorflow.layers import conv3d, max_pooling3d, dropout, conv3d_transpose, dense
from tensorflow.nn import leaky_relu
import tensorflow as tf


def network(inputs, training=False):
    # Input seatting

    # ----------------------------------------------------------------------------------------------------------------------

    #			CONSTRACTING PART OF THE DEEPQMS

    # ----------------------------------------------------------------------------------------------------------------------

    # Layer 1
    # the default strides of tensorflow is (1,1,1) so for a more clean code it is not write it
    # the papper does not tell if they work with normalize data but as a good practise I will do it
    # I add an 1,1,1 conv3d to use less compute resources
    # l1 = conv3d(inputs, 8, kernel_size=(3, 3, 3), padding='same')  # /1
    # l1 = leaky_relu(l1)
    l1 = dense(inputs, 128)
    l1 = leaky_relu(l1)

    # Layer 2
    # l2 = max_pooling3d(l1, 2, 2)  # /2
    # l2 = conv3d(l1, 16, kernel_size=(3, 3, 3), padding='same', strides=2)  # /2
    # l2 = leaky_relu(l2)
    l2 = dense(l1, 256)
    l2 = leaky_relu(l2)

    # l2 = max_pooling3d(l1, 2, 2)  # /2
    # l3 = conv3d(l2, 32, kernel_size=(3, 3, 3), padding='same', strides=2)  # /2
    # l3 = leaky_relu(l3)
    l3 = dense(l2, 512)
    l3 = leaky_relu(l3)

    # Layer 7
    # l4 = conv3d_transpose(l3, 16, [3, 3, 3], [2, 2, 2], padding='same')  # /4
    # l4 = leaky_relu(l4)
    l4 = dense(l3, 256)
    l4 = leaky_relu(l4)
    # l4 = tf.concat([l4, l2], -1)

    # Layer 7
    # l5 = conv3d_transpose(l4, 8, [3, 3, 3], [2, 2, 2], padding='same')  # /4
    # l5 = leaky_relu(l5)
    l5 = dense(l4, 128)
    l5 = leaky_relu(l5)
    # l5 = tf.concat([l5, l1], -1)

    output = conv3d(l5, 1, (1, 1, 1), padding='same')  # /1
    output = leaky_relu(output)
    return output
