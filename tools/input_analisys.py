import os
import tensorflow as tf
from data_generator.tfrecords.tfrecords import read_and_decode

physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
tf.random.set_seed(1024)

path ='96data/'

train_path = list(os.walk(path + 'train/'))[0][-1]
train_path = [path + 'train/' + x for x in train_path]


def train_inputs(batch_size=1, num_shuffles=1, para=24):
    dataset = read_and_decode(train_path)

    dataset = dataset.shuffle(num_shuffles)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=24)
    return dataset


data = train_inputs()
c = 0
for x, y in data.take(475):
    import nibabel as nib
    import numpy as np
    img = nib.Nifti1Image(np.squeeze(x.numpy()), np.identity(4))
    nib.save(img, 'nifti/test{}.nii.gz'.format(c))
    c += 1
