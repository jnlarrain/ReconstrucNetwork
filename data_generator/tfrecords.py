import tensorflow as tf
import os
import numpy as np
import tqdm


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_16_to_int(value):
    value = (((value - np.min(value)) / np.max(value)) * (2. ** 16 - 1))
    return value.astype('uint16')


# images and labels array as input
def convert_tfrecords(images, labels, name):
    num_examples = len(labels)
    if images.shape[0] != num_examples:
        raise ValueError("Images size %d does not match label size %d." %
                         (images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join('D:/', name + '.tfrecords')
    print('Writing', filename)

    writer = tf.python_io.TFRecordWriter(filename)
    for index in tqdm.tqdm(range(num_examples)):
        # image_raw = _float_16_to_int(images[index]).tostring()
        # label_raw = _float_16_to_int(labels[index]).tostring()
        image_raw = images[index].astype('float32').tostring()
        label_raw = labels[index].astype('float32').tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int_feature(rows),
            'width': _int_feature(cols),
            'depth': _int_feature(depth),
            'label': _bytes_feature(label_raw),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())


def read_and_decode(filename):
    def decode_data(coded_data, shape):
        data = tf.decode_raw(coded_data, tf.uint16)
        data = tf.cast(data, tf.float32)
        data = tf.reshape(data, shape + [1])
        return data

    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        features = tf.parse_single_example(example_proto, image_feature_description)
        height = features['height']
        width = features['width']
        depth = features['depth']
        shape = [height, width, depth]
        image_raw = features['image_raw']
        label_raw = features['label']
        image = decode_data(image_raw, shape)
        label = decode_data(label_raw, shape)

        return image, label

    dataset = tf.data.TFRecordDataset(filename)

    image_feature_description = {
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.string),
        'image_raw': tf.FixedLenFeature([], tf.string),
    }

    parsed_dataset = dataset.map(_parse_image_function)
    return parsed_dataset

