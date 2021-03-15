from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import Model
import tensorflow as tf

filepath = 'saved_model'
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
base_model.trainable = False
base_model.summary()
model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv2_block3_out').output)
tf.keras.models.save_model(model, filepath, overwrite=True, include_optimizer=False)
