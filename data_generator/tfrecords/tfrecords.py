import tensorflow as tf


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# images and labels array as input
def convert_tfrecords(fase, magnitud, labels, name):
    num_examples = len(labels)

    rows = fase.shape[0]
    cols = fase.shape[1]
    depth = fase.shape[2]

    filename = name + '.tfrecords'
    print('Writing', filename)

    writer = tf.io.TFRecordWriter(filename)

    fase_raw = fase.astype('float32').tostring()
    mag_raw = magnitud.astype('float32').tostring()
    label_raw = labels.astype('float32').tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int_feature(rows),
        'width': _int_feature(cols),
        'depth': _int_feature(depth),
        'label': _bytes_feature(label_raw),
        'pha_raw': _bytes_feature(fase_raw),
        'mag_raw': _bytes_feature(mag_raw)}))
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
        fase_raw = features['pha_raw']
        mag_raw = features['mag_raw']
        label_raw = features['label']
        fase = decode_data(fase_raw, shape)
        magnitud = decode_data(mag_raw, shape)
        label = decode_data(label_raw, shape)
        image = tf.concat([fase, magnitud], -1)
        return image, label

    dataset = tf.data.TFRecordDataset(filename)

    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.string),
        'pha_raw': tf.io.FixedLenFeature([], tf.string),
        'mag_raw': tf.io.FixedLenFeature([], tf.string),
    }

    parsed_dataset = dataset.map(_parse_image_function)
    return parsed_dataset
