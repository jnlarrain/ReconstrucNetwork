import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tqdm import tqdm
from random import randint
from data_generator.tfrecords.tfrecords import convert_tfrecords
from data_generator.create_one_image import Data
from threading import Thread
from os import walk
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

size = 48
threads_number = 128
FOV = [size, size, size]


disk = "D:/"

# tfrecords path
train_tfrecord_path_00 = disk + str(size) + "data_normal4/train/"
test_tfrecord_path_00 = disk + str(size) + "data_normal4/test/"

if not os.path.exists(disk + str(size) + "data_normal4"):
    os.mkdir(disk + str(size) + "data_normal4")

if not os.path.exists(train_tfrecord_path_00):
    os.mkdir(train_tfrecord_path_00)

if not os.path.exists(test_tfrecord_path_00):
    os.mkdir(test_tfrecord_path_00)


datos_test_00 = list(walk(test_tfrecord_path_00))[0][2]
datos_train_00 = list(walk(train_tfrecord_path_00))[0][2]

num_cylinders = 32
num_spheres = 128  # 128
inner_sus = 1e-7  # 1e-8
ext_sus = 0  # 9 * 1e-6
sources_sus = 1e-1
fov_foco_ext = [
    size * 3,
] * 3
bias = 0

N1 = 4096 * 16  # number for training
N2 = 4096 * 4  # number for evaluation


def training_data(file, num_spheres, num_cylinders, path):
    data = Data(
        size,
        inner_sus,
        ext_sus,
        bias,
        num_spheres,
        num_cylinders,
    )
    data.range_radio = [4, size]
    sus, phase, mag = data.new_image
    sus /= 1e-6
    phase /= 1e-6
    if "train" in path:
        convert_tfrecords(phase, mag, sus, train_tfrecord_path_00 + str(file))
    else:
        convert_tfrecords(phase, mag, sus, test_tfrecord_path_00 + str(file))


def create(N, path, test_flag=False):
    threads = []
    for file in tqdm(range(N)):
        if str(file) + ".tfrecords" not in datos_train_00 or test_flag:
            spheres_number = (
                num_spheres - randint(0, num_spheres) + randint(0, num_spheres)
            )
            cylinder_number = (
                num_cylinders - randint(0, num_cylinders) + randint(0, num_cylinders)
            )
            threads.append(
                Thread(
                    target=training_data,
                    args=(file, spheres_number, cylinder_number, path),
                )
            )
            threads[-1].start()
            if file % threads_number == 0:
                for worker in threads:
                    worker.join()
                threads = []
    for worker in threads:
        worker.join()


create(N1, train_tfrecord_path_00)
create(N2, test_tfrecord_path_00, True)
