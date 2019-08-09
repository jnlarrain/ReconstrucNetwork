from tqdm import tqdm
from random import randint
import numpy as np
from tfrecords import convert_tfrecords
from spheres import sphere
from threading import Thread
from time import time, sleep



size = 48
FOV = [size, ] * 3

train_data = {'min': 0, 'max': 0, 'data': {}}
test_data = {'min': 0, 'max': 0, 'data': {}}
disk = 'D:/'
# tfrecords path
train_tfrecord_path = disk + str(size) + 'data/train_OneSphereAll'
test_tfrecord_path = disk + str(size) + 'data/ruido'

N = 2 ** 14
print('generando esferas entrenamiento')
labels = []
inputs = []
maxi = 0
mini = 1

# def new_image(N_archivo, diccionary, file, range_radio, range_center):
#     global maxi, mini, labels, inputs
#     img_x = np.zeros(FOV)
#     img_y = np.zeros(FOV)
#     for item in range(N_archivo):
#         radio = randint(*range_radio)
#         centers = [randint(*range_center) for _ in range(3)]
#         suscep = np.random.uniform(1e-7, 1e-5)
#         x, y, _ = sphere(FOV, FOV, centers, radio, suscep, 0)
#         diccionary['data'][file][item] = {'r': radio, 'c': centers, 'n': N_archivo, 's': suscep}
#         img_x += x
#         img_y += y
#     if np.max(img_x) > maxi:
#         maxi = np.max(img_x)
#     if np.min(img_x) < mini:
#         mini = np.min(img_x)
#     labels.append(img_y)
#     inputs.append(img_x)


# def training_data(file):
#     global labels, inputs, maxi, mini
#     N_archivo = randint(30, 70)
#     train_data['data'][file] = {}
#     range_center = [12, 30]
#     range_radio = [4, size // 4]
#     new_image(N_archivo, train_data, file, range_radio, range_center)
#     if file%1000==0:
#         print(file)
#
# # for file in tqdm(range(N)):
# #     training_data(file)
#
# # with Pool(2) as p:
# #     print(p.map(training_data, [x for x in range(N)]))
#
# threads = []
# for file in tqdm(range(N)):
#     threads.append(Thread(target=training_data, args=(file,)))
#     threads[-1].start()
#     # if file%1000==0:
#
# print('esperando threads')
# for worker in threads:
#     worker.join()
# print('threads listos')
#
# while any([thread.isAlive() for thread in threads]):
#     len(list(filter(lambda x: x, [thread.isAlive() for thread in threads])))
#
# print('Done')
#
# with open(train_tfrecord_path + ".json", "w") as json_file:
#     train_data['max'] = maxi
#     train_data['min'] = mini
#     json_file.write(str(train_data))
#
# labels = np.array(labels)
# inputs = np.array(inputs) * 255
#
#
# convert_tfrecords(inputs, labels, train_tfrecord_path)

N = 64
print('generando esferas test')
# labels = []
# inputs = []
# maxi = 0
# mini = 1

for file in tqdm(range(N)):
    # N_archivo = randint(1, 10)
    # test_data['data'][file] = {}
    # range_center = [12, 30]
    # range_radio = [4, size // 4]
    # new_image(N_archivo, test_data, file, range_radio, range_center)
    labels.append(np.random.normal(0, 1, (48, 48, 48, 1)))
    inputs.append(np.random.normal(0, 1, (48, 48, 48, 1)))

# with open(test_tfrecord_path + ".json", "w") as json_file:
#     test_data['max'] = maxi
#     test_data['min'] = mini
#     json_file.write(str(test_data))

labels = np.array(labels)
inputs = np.array(inputs)

convert_tfrecords(inputs, labels, test_tfrecord_path)
