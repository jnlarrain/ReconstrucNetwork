import os
import tensorflow as tf
from model.estimator import Estimator

physical_devices = tf.config.list_physical_devices("GPU")
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
tf.random.set_seed(1024)

version = "64_normal_10"

print(version)

size = (
    *[
        96,
    ]
    * 3
    + [1],
)
lr = 1e-3
model = Estimator(lr, size, version)


def serving_input_fn():
    features = {"x": tf.compat.v1.placeholder(shape=[1, None, None, None, 1], dtype=tf.float32)}
    return tf.estimator.export.ServingInputReceiver(features, features)


estimator_path = model._estimator.export_saved_model("saved_model", serving_input_fn)


# imported = tf.saved_model.load('saved_model/model')
#
#
# def predict(x):
#   return imported.signatures["serving_default"](x)
#
# print(predict(tf.zeros((1, *window, 3)))['output'].shape)

# converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/model')
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
#
# quantized_tflite_model = converter.convert()
