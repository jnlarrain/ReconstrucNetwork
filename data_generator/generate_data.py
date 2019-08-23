from tqdm import tqdm
from random import randint
import numpy as np
from tfrecords import convert_tfrecords
from spheres import sphere
from threading import Thread

size = 256

FOV = [256, 256, 98]

train_data = {'min': 0, 'max': 0, 'data': {}}
test_data = {'min': 0, 'max': 0, 'data': {}}
disk = 'D:/'
# tfrecords path
train_tfrecord_path = disk + str(size) + 'data/train/'
test_tfrecord_path = disk + str(size) + 'data/test/'

N = 64  # 14
print('generando esferas entrenamiento')

maxi = 0
mini = 1


def new_image(N_archivo, diccionary, file, range_radio, range_center, path):
    global maxi, mini
    img_x = np.zeros(FOV)
    img_y = np.zeros(FOV)
    for item in range(N_archivo):
        radio = randint(*range_radio)
        centers = [randint(*range_center) * 4, randint(*range_center) * 4, randint(*range_center)]
        suscep = np.random.uniform(-1e-5, 1e-5)
        x, y, _ = sphere(FOV, FOV, centers, radio, suscep, 0)
        diccionary['data'][file][item] = {'r': radio, 'c': centers, 'n': N_archivo, 's': suscep}
        img_x += x
        img_y += y
    if np.max(img_x) > maxi:
        maxi = np.max(img_x)
    if np.min(img_x) < mini:
        mini = np.min(img_x)
    img_x = np.reshape(img_x, [1] + FOV)
    img_y = np.reshape(img_y, [1] + FOV)
    convert_tfrecords(img_y, img_x, path + str(file))


def training_data(file):
    global maxi, mini
    N_archivo = randint(1, 400)
    train_data['data'][file] = {}
    range_center = [2, size // 9]
    range_radio = [4, size // 8]
    new_image(N_archivo, train_data, file, range_radio, range_center, train_tfrecord_path)


def testing_data(file):
    global maxi, mini
    N_archivo = randint(1, 400)
    test_data['data'][file] = {}
    range_center = [2, size // 9]
    range_radio = [4, size // 8]
    new_image(N_archivo, train_data, file, range_radio, range_center, test_tfrecord_path)


threads = []
for file in tqdm(range(N)):
    threads.append(Thread(target=training_data, args=(file,)))
    threads[-1].start()
    if file % 64 == 0:
        for worker in threads:
            worker.join()
        threads = []

print('Done')

with open(train_tfrecord_path + "diccionario.json", "w") as json_file:
    train_data['max'] = maxi
    train_data['min'] = mini
    json_file.write(str(train_data))

N = 64
print('generando esferas test')
maxi = 0
mini = 1

threads = []
for file in tqdm(range(N)):
    threads.append(Thread(target=testing_data, args=(file,)))
    threads[-1].start()
    if file % 64 == 0:
        for worker in threads:
            worker.join()
        threads = []

with open(test_tfrecord_path + "diccionario.json", "w") as json_file:
    test_data['max'] = maxi
    test_data['min'] = mini
    json_file.write(str(test_data))
