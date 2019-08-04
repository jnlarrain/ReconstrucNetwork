import tensorflow as tf
from model import network
from tfrecords import read_and_decode
import os


os.environ['TF_ENABLE_MIXED_PRECISION'] = '1'

'''
main48 tiene 256 samples, batch 32
'''

version = 53
size = 48
disk = 'D:/'
main_path = 'logs/mainOne' + str(size) + 'version' + str(version)
eval_path = 'logs/evaluationOne' + str(size) + 'version' + str(version)

# tfrecords path
train_tfrecord_path = disk+str(size)+'data/train_OneSphereAll.tfrecords'
test_tfrecord_path = disk+str(size)+'data/test_OneSphereAll2.tfrecords'


# set the path's were you want to storage the data(tensorboard and checkpoints)
batch = 16
epochs = 21000 #- ( 90 )
input_shape = (size, size, size, 1)
learning_rate = 3e-5
B1 = 0.9
B2 = 0.99


def train_inputs(batch_size, num_shuffles=100):
    dataset = read_and_decode(train_tfrecord_path)

    # shuffle and repeat examples for better randomness and allow training beyond one epoch
    dataset = dataset.repeat(epochs//batch_size)
    dataset = dataset.shuffle(num_shuffles)

    # map the parse  function to each example individually in threads*2 parallel calls

    # batch the examples
    dataset = dataset.batch(batch_size=batch_size)

    # prefetch batch

    dataset = dataset.prefetch(buffer_size=batch_size)
    return dataset


def show_image(preds):
    result = tf.concat([preds[:, size // 2, :, :, :],
                        preds[:, :, size // 2, :, :],
                        preds[:, :, :, size // 2, :]], 1)
    return result


def show_summary(images, labels):
    diff = labels-images
    new_images = tf.concat([images[:, size // 2, :, :, :],
                        images[:, :, size // 2, :, :],
                        images[:, :, :, size // 2, :]], 1)
    new_diff = tf.concat([diff[:, size // 2, :, :, :],
                        diff[:, :, size // 2, :, :],
                        diff[:, :, :, size // 2, :]], 1)
    new_labels = tf.concat([labels[:, size // 2, :, :, :],
                        labels[:, :, size // 2, :, :],
                        labels[:, :, :, size // 2, :]], 1)
    maximum = tf.ones(tf.shape(new_labels))*tf.reduce_max(diff)
    minimum = tf.ones(tf.shape(new_labels))*tf.reduce_min(diff)
    zeros = tf.zeros(tf.shape(new_labels))
    scale = tf.concat([maximum[:, :size, :2, :], zeros[:, :size, :2, :], minimum[:, :size, :2, :]], 1)
    result = tf.concat([new_labels[:, :, :, :],
                        new_images[:, :, :, :],
                        new_diff[:, :, :, :], scale], 2)


    return result

def eval_inputs(batch_size):
    dataset = read_and_decode(test_tfrecord_path)
    dataset = dataset.shuffle(100).batch(batch_size)
    return dataset


def loss_funtion(labels, preds):
    # l1 = tf.losses.absolute_difference(labels, preds)
    l2 = tf.losses.mean_squared_error(labels, preds)
    return l2


def estimator_function(features, labels, mode, params):
    global step
    if mode == tf.estimator.ModeKeys.PREDICT:
        y_pred = network(features)
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions=y_pred)
    # For training and testing
    else:
        training = mode == tf.estimator.ModeKeys.TRAIN
        y_pred = network(features)
        # summary the training image
        summary_images = tf.summary.image("Summary",
                                       show_summary(y_pred, labels),
                                       max_outputs=8)
        loss = loss_funtion(tf.cast(labels, tf.float32), tf.cast(y_pred, tf.float32))
        summary_loss = tf.summary.scalar('loss', loss)
        tf.summary.merge([summary_loss, summary_images])
        if training:
            params["learning_rate"] = params["learning_rate"] * .99
            optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"], beta1=B1, beta2=B2)
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
                            tf.summary.image("Summary_eval",
                                             show_summary(y_pred, labels),
                                             max_outputs=8)]
            )

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                predictions=y_pred,
                evaluation_hooks=[evaluation_hook]
            )
    return spec


config = tf.ConfigProto()
config.gpu_options.allow_growth = True


params = {"learning_rate": learning_rate}
# Inializate the estimator
model = tf.estimator.Estimator(
    model_fn=estimator_function,
    params=params,
    model_dir=main_path)

eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_inputs(batch))
print('Starting training')


salida = model.predict(input_fn=lambda: eval_inputs(batch))

import matplotlib.pyplot as plt
import numpy as np

out = np.array(list(salida))
out = out.reshape(out.shape[:-1])

for img in out[0]:
    plt.imshow(img, cmap='gray')
    plt.show()



