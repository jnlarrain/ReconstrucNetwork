from os import walk
import pickle
import numpy as np


class Data:
    def __init__(self, disk: str, size: int, number_of_data: int, batch: int):
        self.batch = batch
        self.number_of_data = number_of_data
        self.size = size
        self.disk = disk
        self.input_directory = self.disk + str(self.size) + 'data/new_data'
        self.label_directory = self.disk + str(self.size) + 'data/new_dataf'

    @staticmethod
    def abrir_data(path):
        with open(path, 'rb') as file:
            datos = pickle.load(file)
            return datos

    def main(self):
        train_data = list(walk(self.input_directory))[0][2][:self.number_of_data]
        train_data = [self.input_directory + path for path in train_data[:self.number_of_data - self.batch]]
        train_labels = list(walk(self.label_directory))[0][2][:self.number_of_data]
        train_labels = [self.label_directory + path for path in train_labels[:self.number_of_data - self.batch]]
        train_data = np.array(list(map(self.abrir_data, train_data)))
        train_labels = np.array(list(map(self.abrir_data, train_labels)))

        test_data = list(walk(self.input_directory))[0][2][:self.number_of_data]
        test_data = [self.input_directory + path for path in test_data[-self.batch:]]
        test_labels = list(walk(self.label_directory))[0][2][:self.number_of_data]
        test_labels = [self.label_directory + path for path in test_labels[-self.batch:]]
        test_data = np.array(list(map(self.abrir_data, test_data)))
        test_labels = np.array(list(map(self.abrir_data, test_labels)))

        print('Data already done there are {} test volumes and {}'
              ' training volumes'.format(len(test_data), len(train_data)))

        return {'train_input':train_data, 'train_label': train_labels,
                'test_input': test_data, 'test_label': test_labels}
