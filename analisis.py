import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle

tf.enable_eager_execution()
path = "D:/ReconstrucNetwork/resultados"
version = 2
size = 48
disk = 'D:/'
main_path = 'logs/mainOne' + str(size) + 'version' + str(version)
eval_path = 'logs/evaluationOne' + str(size) + 'version' + str(version)

# tfrecords path
train_tfrecord_path = disk + str(size) + 'data/train_16384.tfrecords'
test_tfrecord_path = disk + str(size) + 'data/test_1024.tfrecords'

with open(path + '/resultados.txt', 'rb') as data:
    out = pickle.load(data)

with open(path + '/ruido.txt', 'rb') as data:
    ruido = pickle.load(data)


def read_and_decode(filename):
    def decode_data(coded_data, size):
        data = tf.decode_raw(coded_data, tf.uint16)
        data = tf.cast(data, tf.float32)
        data = tf.reshape(data, [size, ] * 3 + [1])
        # data = tf.image.per_image_standardization(data)
        data = data /(2**16 - 1)
        return data

    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        features = tf.parse_single_example(example_proto, image_feature_description)
        size = features['height']
        image_raw = features['image_raw']
        label_raw = features['label']
        label = decode_data(image_raw, size)
        image = decode_data(label_raw, size)

        return label, image

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


data = read_and_decode(test_tfrecord_path)
dataset = data.take(len(out))

datos = []

for i in dataset.take(1024):
    datos.append(i[0][:, :, :, 0])

diff = []
for j in range(1024):
    # plt.figure()
    # plt.imshow(out[j][:, 24, :], cmap='gray')
    # plt.figure()
    # plt.imshow(datos[j][:, 24, :, 0], cmap='gray')
    # plt.show()
    diff.append(datos[j]-out[j])

error = []

for item in diff:
    print(np.max(item), np.min(item))

for item in diff:
    error.append(sum((len(x) for x in np.where(np.abs(item) > .2))))


plt.bar([x for x in range(1, 1025)], error)
plt.show()

test_tfrecord_path = disk + str(size) + 'data/ruido.tfrecords'
data = read_and_decode(test_tfrecord_path)

datos = []
for i in data.take(64):

    datos.append(i[0][:, :, :, 0])


for j in range(64):
    plt.figure()
    plt.imshow(ruido[j][:, 24, :], cmap='gray')

    plt.figure()
    plt.imshow(datos[j][:, 24, :], cmap='gray')

    plt.show()

diff = []
for j in range(64):
    # plt.figure()
    # plt.imshow(out[j][:, 24, :], cmap='gray')
    # plt.figure()
    # plt.imshow(datos[j][:, 24, :, 0], cmap='gray')
    # plt.show()
    diff.append(datos[j]-ruido[j])

for item in diff:
    print(np.max(item), np.min(item))

for item in diff:
    error.append(sum((len(x) for x in np.where(np.abs(item) > .2))))


plt.bar([x for x in range(1, 1025)], error)
plt.show()
