import keras 
from keras.models import load_model

modelo = load_model('KerasModel1.ckpt')



weights, biases = modelo.layers[2].get_weights()

print(weights, biases)