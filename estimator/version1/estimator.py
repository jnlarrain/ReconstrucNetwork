import numpy as np
from load_data import Data
import tensorflow as tf
from model4 import network
from tfrecords import read_and_decode

'''
main48 tiene 256 samples, batch 32
'''

version = 1
size = 48
disk = 'D:/'
main_path = 'logs/mainOne' + str(size) + 'version' + str(version)
eval_path = 'logs/evaluationOne' + str(size) + 'version' + str(version)

# tfrecords path
train_tfrecord_path = disk+str(size)+'data/train_data.tfrecords'
test_tfrecord_path = disk+str(size)+'data/test_data.tfrecords'


# set the path's were you want to storage the data(tensorboard and checkpoints)
batch = 2
epochs = 70400
number_of_data = 128
test_samples = 32
input_shape = (size, size, size, 1)
learning_rate = 1e-3
B1 = 0.9
B2 = 0.99


def train_inputs(batch_size, num_shuffles=100):
    dataset = read_and_decode(train_tfrecord_path)
    dataset = dataset.shuffle(num_shuffles).batch(batch_size)
    return dataset


def show_image(preds):
    result = preds[:, size // 2, :, :, :]
    return result


def eval_inputs(batch_size):
    dataset = read_and_decode(test_tfrecord_path)
    dataset = dataset.shuffle(100).batch(batch_size)
    return dataset


def loss_funtion(labels, preds):
    l1 = tf.losses.absolute_difference(labels, preds)
    l2 = tf.losses.mean_squared_error(labels, preds)
    return l1 + l2


def estimator_function(features, labels, mode, params):
    if mode == tf.estimator.ModeKeys.PREDICT:
        y_pred = network(features)
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions=y_pred)
    # For training and testing
    else:
        training = mode == tf.estimator.ModeKeys.TRAIN
        y_pred = network(features, size, training)
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
            params["learning_rate"] = params["learning_rate"] * .99
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
    model.train(input_fn=lambda: train_inputs(batch, number_of_data))
    print(model.evaluate(input_fn=lambda: eval_inputs(batch)))

print('Trainned finished')
