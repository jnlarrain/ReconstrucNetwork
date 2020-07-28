from tqdm import tqdm
from random import randint
import numpy as np
from tfrecords import convert_tfrecords
from spheres import sphere
from threading import Thread
from os import walk

size = 48
threads_number = 1
FOV = [size, size, size]
train_data = {'min': 0, 'max': 0, 'data': {}}
test_data = {'min': 0, 'max': 0, 'data': {}}
disk = 'F:/'
# tfrecords path
train_tfrecord_path = disk + str(size) + 'data_noise/train/'
test_tfrecord_path = disk + str(size) + 'data_noise/test/'
rand_esf = 1024  # 1024

datos_test = list(walk(test_tfrecord_path))[0][2]
datos_train = list(walk(train_tfrecord_path))[0][2]

N = 2 ** 16  # 14
print('generando esferas entrenamiento')

maxi = 0
mini = 1


def new_image(N_archivo, diccionary, file, range_radio, range_center, path):
    global maxi, mini
    img_x = np.zeros(FOV)
    img_y = np.zeros(FOV)
    for item in range(N_archivo):
        radio = randint(*range_radio)
        centers = [randint(*range_center), randint(*range_center), randint(*range_center)]
        suscep = np.random.uniform(-1e-3, 1e-3)
        x, y, mg = sphere(FOV, FOV, centers, radio, suscep, 0)
        # diccionary['data'][file][item] = {'r': radio, 'c': centers, 'n': N_archivo, 's': suscep}
        img_x += x
        img_y += y
    if np.max(img_x) > maxi:
        maxi = np.max(img_x)
    if np.min(img_x) < mini:
        mini = np.min(img_x)

    mag = np.abs(img_x) / np.max(np.abs(img_x))  # normalizamos la magnitud
    complex = mag * np.exp(img_y * 1j)
    percent = 1 / 1024 * np.max(np.abs(complex))
    noise = np.random.normal(size=FOV, scale=1e-6) + np.random.normal(size=FOV, scale=1e-6) * 1j
    noise = noise / (np.max(np.abs(noise)))
    noise *= percent
    complex += np.abs(noise) * np.exp(np.angle(noise) * 1j)
    img_y = np.angle(complex)
    img_x = np.reshape(img_x, [1] + FOV)
    img_y = np.reshape(img_y, [1] + FOV)

    convert_tfrecords(img_y, img_x, path + str(file))


def training_data(file):
    global maxi, mini
    N_archivo = randint(1, rand_esf)
    train_data['data'][file] = {}
    range_center = [size // 4, size // 4 * 3]
    range_radio = [4, size // 6]
    new_image(N_archivo, train_data, file, range_radio, range_center, train_tfrecord_path)


def testing_data(file):
    global maxi, mini
    N_archivo = randint(1, rand_esf)
    test_data['data'][file] = {}
    range_center = [size // 4, size // 4 * 3]
    range_radio = [4, size // 6]
    new_image(N_archivo, train_data, file, range_radio, range_center, test_tfrecord_path)


threads = []
for file in tqdm(range(N)):
    if str(file) + '.tfrecords' not in datos_train:
        threads.append(Thread(target=training_data, args=(file,)))
        threads[-1].start()
        if file % threads_number == 0:
            for worker in threads:
                worker.join()
            threads = []

for worker in threads:
    worker.join()
threads = []

print('Done')

# with open(train_tfrecord_path + "train.json", "w") as json_file:
#     train_data['max'] = maxi
#     train_data['min'] = mini
#     json_file.write(str(train_data))

N = 4096
print('generando esferas test')
maxi = 0
mini = 1

threads = []
for file in tqdm(range(N)):
    if str(file) + '.tfrecords' not in datos_test:
        threads.append(Thread(target=testing_data, args=(file,)))
        threads[-1].start()
        if file % threads_number == 0:
            for worker in threads:
                worker.join()
            threads = []

# with open(test_tfrecord_path + "test.json", "w") as json_file:
#     test_data['max'] = maxi
#     test_data['min'] = mini
#     json_file.write(str(test_data))
