from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.util.tf_export import tf_export
from dipolo import dipole as d

# @tf_export('keras.metrics.mean_squared_error',
#            'keras.metrics.mse',
#            'keras.metrics.MSE',
#            'keras.losses.mean_squared_error',
#            'keras.losses.mse',
#            'keras.losses.MSE')
# def mean_squared_error(y_true, y_pred):
#   return K.mean(math_ops.square(y_pred - y_true), axis=-1)


# @tf_export('keras.metrics.mean_absolute_error',
#            'keras.metrics.mae',
#            'keras.metrics.MAE',
#            'keras.losses.mean_absolute_error',
#            'keras.losses.mae',
#            'keras.losses.MAE')
# def mean_absolute_error(y_true, y_pred):
#   return K.mean(math_ops.abs(y_pred - y_true), axis=-1)


# @tf_export('keras.metrics.mean_absolute_percentage_error',
#            'keras.metrics.mape',
#            'keras.metrics.MAPE',
#            'keras.losses.mean_absolute_percentage_error',
#            'keras.losses.mape',
#            'keras.losses.MAPE')
# def mean_absolute_percentage_error(y_true, y_pred):
#   diff = math_ops.abs(
#       (y_true - y_pred) / K.clip(math_ops.abs(y_true), K.epsilon(), None))
# return 100. * K.mean(diff, axis=-1)




@tf_export('keras.metrics.L1_error',
           'keras.losses.L1_error')
def L1_error(y_true, y_pred):
  return K.mean(math_ops.square(y_pred - y_true), axis=-1)


@tf_export('keras.metrics.Model_error'
           'keras.losses.Model_error')
def Model_error(y_true, y_pred):
  return K.mean(math_ops.square(K.conv3d(y_pred, d, padding='same') - K.conv3d(y_true, d, padding='same')), axis=-1)


@tf_export('keras.metrics.Gradient_error',
           'keras.losses.Gradient_error')
def Gradient_error(y_true, y_pred):
  return 0

@tf_export('keras.metrics.Conicit_error',
           'keras.losses.Conicit_error')
def Conicit_error(y_true, y_pred):
  return L1_error(y_true,y_pred)+0.1*Gradient_error(y_true,y_pred)#Model_error(y_true,y_pred)+



