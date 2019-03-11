import tensorflow as tf
from tensorflow.data import Dataset, Iterator
from model4 import network
import pickle
from os import walk
import numpy as np

'''
main48 tiene 256 samples, batch 32
'''

version = 1
FOV = 64
main_path = 'logs/mainOne' + str(FOV) + 'version' + str(version)
eval_path = 'logs/evaluationOne' + str(FOV) + 'version' + str(version)

# set the path's were you want to storage the data(tensorboard and checkpoints)
batch = 32
epochs = 1000
number_of_data = 128
test_samples = number_of_data // 16 * 12
input_shape = (FOV, FOV, FOV, 1)
learning_rate = 1e-3
B1 = 0.9
B2 = 0.99


def abrir_data(path):
    with open(path, 'rb') as file:
        datos = pickle.load(file)
        return datos


train_data = list(walk('D:/' + str(FOV) + "data/new_data"))[0][2][:number_of_data]
train_data = ['D:/' + str(FOV) + "data/new_data/" + path for path in train_data[:number_of_data - batch]]
train_labels = list(walk('D:/' + str(FOV) + "data/new_dataf"))[0][2][:number_of_data]
train_labels = ['D:/' + str(FOV) + "data/new_dataf/" + path for path in train_labels[:number_of_data - batch]]
train_data = np.array(list(map(abrir_data, train_data)))
train_labels = np.array(list(map(abrir_data, train_labels)))

test_data = list(walk('D:/' + str(FOV) + "data/new_data"))[0][2][:number_of_data]
test_data = ['D:/' + str(FOV) + "data/new_data/" + path for path in test_data[-batch:]]
test_labels = list(walk('D:/' + str(FOV) + "data/new_dataf"))[0][2][:number_of_data]
test_labels = ['D:/' + str(FOV) + "data/new_dataf/" + path for path in test_labels[-batch:]]
test_data = np.array(list(map(abrir_data, test_data)))
test_labels = np.array(list(map(abrir_data, test_labels)))

print('Data already done there are {} test volumes and {} training volumes'.format(len(test_data), len(train_data)))


def train_inputs(features, labels, batch_size, num_shuffles=1000):
    ''' Function to create a training dataset

    Parameters
    ----------
    features: tensor
        Heart images used for training
    labels: tensor
        Labels used for training
    batch_size: int
        Number that specifies the batch size
    num_shuffles: int
        Number of times to shuffle the data, default is 1000

    Returns
    -------
    inputs: tensorflow Dataset
        dataset in tensorfloww format ready to be train
    '''
    inputs = Dataset.from_tensor_slices((features, labels))
    inputs = inputs.shuffle(num_shuffles).batch(batch_size)
    return inputs


def show_image(preds):
    ''' Function to draw the bounding boxes in the images

    Parameters
    ----------
    images: tensor
        tensor of images without bounding boxes
    labels: tensor
        tensor of labeled corners
    preds: tensor
        tensor of predicted corners

    Returns
    -------
    results:tensor
            tensor of images with bounding boxes. Blue: predicted / Yellow: labeled
    '''
    result = preds[:, FOV // 2, :, :, :]
    return result


def eval_inputs(features, labels, batch_size):
    ''' Function to create a evaluation dataset

    Parameters
    ----------
    features: tensor
        tensor of images features
    labels: tensor
        tensor of labels
    batch_size: int
        int of the batch size

    Returns
    -------
    inputs: tensorflow Dataset
        dataset in tensorflow format ready to eval the epoch
    '''
    inputs = Dataset.from_tensor_slices((features, labels))
    # we shuffle the data justo see differente images in tensorboard
    inputs = inputs.shuffle(100).batch(batch_size)
    return inputs

def loss_funtion(labels, preds):
    l1 = tf.losses.absolute_difference(labels, preds)
    l2 = tf.losses.mean_squared_error(labels, preds)
    return l1 + l2

def estimator_function(features, labels, mode, params):
    ''' Estimator function

    Parameters
    ----------
    inputs: tensor
        images and labels, image shape (?, field_of_view, field_of_view, 1)
        and labels of shape(?, 1, 4)
    training: boolean
        determinates if we used the dropout, True: yes / False: no

    Returns
    -------
    the estimates values from the tensorflow estimator
    '''
    if mode == tf.estimator.ModeKeys.PREDICT:
        y_pred = network(features)
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions=y_pred)
    # For training and testing
    else:
        training = mode == tf.estimator.ModeKeys.TRAIN
        y_pred = network(features, FOV, training)
        # summary the training image
        prediction_image = tf.summary.image("Prediction",
                                            show_image(y_pred),
                                            max_outputs=8)
        label_image = tf.summary.image("Label",
                                       show_image(labels),
                                       max_outputs=8)
        loss = loss_funtion(labels, y_pred)
        summary_loss = tf.summary.scalar('loss', loss)
        tf.summary.merge([prediction_image, label_image, summary_loss])
        if training:
            params["learning_rate"] = params["learning_rate"]*.99
            # tf.print(loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"], beta1=0.95, beta2=0.999)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                predictions=y_pred
            )
        else:
            # add summary evaluation image
            evaluation_hook = tf.train.SummarySaverHook(
                save_steps=1,
                output_dir=eval_path,
                summary_op=[tf.summary.scalar('loss', loss),
                            tf.summary.image("Test prediction",
                                             show_image(y_pred),
                                             max_outputs=8)]
            )

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                predictions=y_pred,
                evaluation_hooks=[evaluation_hook]
            )
    return spec


params = {"learning_rate": learning_rate}
# set up the configurations of the estimator
config = tf.estimator.RunConfig(
    save_summary_steps=100,
    save_checkpoints_steps=100,
    keep_checkpoint_max=1
)
# Inializate the estimator
model = tf.estimator.Estimator(
    model_fn=estimator_function,
    params=params,
    model_dir=main_path,
    config=config)

print('Starting training')

for epoch in range(epochs):
    print(epoch)
    model.train(input_fn=lambda: train_inputs(train_data, train_labels, batch, number_of_data))
    print(model.evaluate(input_fn=lambda: eval_inputs(test_data, test_labels, batch)))

print('Trainned finished')
