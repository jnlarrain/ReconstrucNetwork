import tensorflow as tf
import numpy as np
from model.estimator import Estimator


def load_image(path) -> np.array:
    # load your image as a numpy array, the z axes must by the 3rd dimention in ppm
    pass


def load_model(model_version="weigths", size=(48, 48, 48, 1), lr=1e-3) -> Estimator:
    return Estimator(lr, size, model_version)


def from_numpy(elements):
    data = tf.data.Dataset.from_tensor_slices((elements, elements))
    data = data.map(lambda x, y: (tf.cast(x, tf.float32), y))
    data = data.batch(1)
    return data


def infer(image):
    model = load_model()
    return np.squeeze(
        np.array(list(model._estimator.predict(lambda: from_numpy(image))))
    )
