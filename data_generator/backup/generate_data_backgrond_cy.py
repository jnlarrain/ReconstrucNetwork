from tqdm import tqdm
from random import randint, choice
import numpy as np
from tfrecords import convert_tfrecords
from figures.spheres import sphere
from figures.spherestf import sphere as tfsphere
from figures.cylinders import cylinder
from threading import Thread, Lock
from os import walk


size = 384
threads_number = 256
FOV = [size, size, size]
train_data = {'min': 0, 'max': 0, 'data': {}}
test_data = {'min': 0, 'max': 0, 'data': {}}
disk = 'F:/'

# tfrecords path
train_tfrecord_path = disk + str(size) + 'data_background_cy/train/'
test_tfrecord_path = disk + str(size) + 'data_background_cy/test/'


datos_test = list(walk(test_tfrecord_path))[0][2]
datos_train = list(walk(train_tfrecord_path))[0][2]

N = 2**16  # 14
print('generando esferas entrenamiento')

maxi = 0
mini = 1
lock = Lock()

def new_image(N_archivo, diccionary, file, range_radio, range_center, path):
    def out_spheres(number):
        new_FOV = [dimention*3 for dimention in FOV]
        background = np.zeros(FOV)
        posible_range = [x for x in range(new_FOV[0]) if x < size or x > size * 2]
        for _ in range(number):
            radio = randint(*range_radio)
            centers = [choice(posible_range), choice(posible_range), choice(posible_range)]
            suscep = np.random.uniform(-1e2, 1e2)*1e-6
            with lock:
                x, y, _ = tfsphere(new_FOV, new_FOV, centers, radio, suscep, 0)
            x /= 1e-6
            y /= 1e-6
            constant = FOV[0]
            background += y[constant:constant*2, constant:constant*2, constant:constant*2]
        return background

    global maxi, mini
    img_x = np.zeros(FOV)
    img_y = np.zeros(FOV)
    for item in range(N_archivo):
        radio = randint(*range_radio)
        centers = [randint(*range_center), randint(*range_center), randint(*range_center)]
        suscep = np.random.uniform(-1e-3, 1e-3)*1e-6
        x, y, _ = tfsphere(FOV, FOV, centers, radio, suscep, 0)
        img_x += x / 1e-6
        img_y += y / 1e-6
    for item in range(N_archivo - randint(0, N_archivo) + randint(0, N_archivo)):
        radio = randint(*range_radio)
        p1 = np.array([randint(*range_center), randint(*range_center), randint(*range_center)], dtype='float32')
        p2 = np.array([randint(*range_center), randint(*range_center), randint(*range_center)], dtype='float32')
        suscep = np.random.uniform(-1e-3, 1e-3)*1e-6
        x, y = cylinder(FOV, FOV, p1, p2, radio, suscep, 0)
        img_x += x / 1e-6
        img_y += y / 1e-6
    back_spheres = randint(1, 7)
    img_y += out_spheres(back_spheres)
    img_x = np.reshape(img_x, [1] + FOV)
    img_y = np.reshape(img_y, [1] + FOV)
    convert_tfrecords(img_y, img_x, path + str(file))


def training_data(file):
    global maxi, mini
    N_archivo = randint(1, 1024)
    train_data['data'][file] = {}
    range_center = [size//4, size // 4 * 3]
    range_radio = [4, size // 6]
    new_image(N_archivo, train_data, file, range_radio, range_center, train_tfrecord_path)


def testing_data(file):
    global maxi, mini
    N_archivo = randint(1, 1024)
    test_data['data'][file] = {}
    range_center = [size//4, size // 4 * 3]
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

N = 4096
print('generando esferas test')
maxi = 0
mini = 1

threads = []
for file in tqdm(range(N)):
    if str(file)+'.tfrecords' not in datos_test:
        threads.append(Thread(target=testing_data, args=(file,)))
        threads[-1].start()
        if file % threads_number == 0:
            for worker in threads:
                worker.join()
            threads = []

