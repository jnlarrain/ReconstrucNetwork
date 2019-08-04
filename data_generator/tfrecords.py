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


def _int_to_float_16(value):
    return value / 2 ** 16, tf.float16


def dataset_generator(features, labels):
    inputs = tf.data.Dataset.from_tensor_slices((features, labels))
    return inputs


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
        image_raw = _float_16_to_int(images[index]).tostring()
        label_raw = _float_16_to_int(labels[index]).tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int_feature(rows),
            'width': _int_feature(cols),
            'depth': _int_feature(depth),
            'label': _bytes_feature(label_raw),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())


def read_and_decode(filename):
    def decode_data(coded_data, size):
        data = tf.decode_raw(coded_data, tf.uint16)
        data = tf.cast(data, tf.float32)
        data = tf.reshape(data, [size, ] * 3 + [1])
        data = tf.image.per_image_standardization(data)
        return data

    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        features = tf.parse_single_example(example_proto, image_feature_description)
        size = features['height']
        image_raw = features['image_raw']
        label_raw = features['label']
        image = decode_data(image_raw, size)
        label = decode_data(label_raw, size)

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


if __name__ == '__main__':
    tf.enable_eager_execution()
    disk = 'E:'
    size = 48

    from .load_data import Data

    number_of_data = 70  # 70400
    test_samples = 16

    data = Data(disk, size, number_of_data, test_samples)
    data_dict = data.main()
    train_labels = data_dict['train_input'][:16]
    train_data = data_dict['train_label'][:16]
    test_labels = data_dict['test_input'][:16]
    test_data = data_dict['test_label'][:16]
    convert_tfrecords(train_data, train_labels, str(size) + 'data/train_dataL')
    convert_tfrecords(test_data, test_labels, str(size) + 'data/test_dataL')

    # tfrecords path
    # train_tfrecord_path = disk + str(size) + 'data/train_data.tfrecords'
    # test_tfrecord_path = disk + str(size) + 'data/test_data.tfrecords'
    # read_and_decode(test_tfrecord_path)
