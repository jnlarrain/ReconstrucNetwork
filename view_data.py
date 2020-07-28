import tensorflow as tf
from data_generator.tfrecords import read_and_decode
import os
from visualizador import cuatro_cortes

size = 48
disk = 'F:/'
path = disk + str(size) + 'data_noise/'

train_path = list(os.walk(path + 'train/'))[0][-1]
test_path = list(os.walk(path + 'test/'))[0][-1]

train_path = [path + 'train/' + x for x in train_path]
test_path = [path + 'test/' + x for x in test_path]


dataset = read_and_decode(train_path).as_numpy_iterator()

for element in dataset:
    print(cuatro_cortes(element[0][:, :, :, 0]))
    exit()





