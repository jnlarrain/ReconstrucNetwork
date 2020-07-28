import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from estimator.estimator import Estimator
import os


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


size = 128
disk = 'D:/'
input_shape = (size, size, size, 1)
learning_rate = 3e-5

path = disk + str(size) + 'data/'



def img_scale(img):
    num = 64
    fig = plt.figure()
    image = plt.imshow(img[:, :, num], cmap='gray')
    cbar = fig.colorbar(image)
    

def convert_tfrecords(images, labels):
    num_examples = len(labels)
    if images.shape[0] != num_examples:
        raise ValueError("Images size %d does not match label size %d." %
                         (images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join('D:/fantom.tfrecords')
    print('Writing', filename)

    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].astype('float32').tostring()
        label_raw = labels[index].astype('float32').tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int_feature(rows),
            'width': _int_feature(cols),
            'depth': _int_feature(depth),
            'label': _bytes_feature(label_raw),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())


def from_numpy(elements):
    elements = np.reshape(elements, [1, *elements.shape, 1])
    data = tf.data.Dataset.from_tensor_slices((elements, elements))
    data = data.batch(len(elements))
    return data
































