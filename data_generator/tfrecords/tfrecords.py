import tensorflow as tf


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# images and labels array as input
def convert_tfrecords(images, labels, name):
    num_examples = len(labels)
    if images.shape[0] != num_examples:
        raise ValueError("Images size %d does not match label size %d." %
                         (images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = name + '.tfrecords'
    print('Writing', filename)

    writer = tf.io.TFRecordWriter(filename)
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


def read_and_decode(filename):
    def decode_data(coded_data, shape):
        data = tf.io.decode_raw(coded_data, tf.float32)
        data = tf.reshape(data, shape + [1])
        data = tf.cast(data, tf.float32)
        return data

    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        features = tf.io.parse_single_example(example_proto, image_feature_description)
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
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }

    parsed_dataset = dataset.map(_parse_image_function)
    return parsed_dataset
