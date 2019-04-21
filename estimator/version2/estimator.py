import tqdm
import tensorflow as tf
from model import network
from tfrecords import read_and_decode
import os


os.environ['TF_ENABLE_MIXED_PRECISION'] = '1'

'''
main48 tiene 256 samples, batch 32
'''

version = 3
size = 48
disk = 'D:/'
main_path = 'logs/mainOne' + str(size) + 'version' + str(version)
eval_path = 'logs/evaluationOne' + str(size) + 'version' + str(version)

# tfrecords path
train_tfrecord_path = disk+str(size)+'data/train_dataL.tfrecords'
test_tfrecord_path = disk+str(size)+'data/test_dataL.tfrecords'


# set the path's were you want to storage the data(tensorboard and checkpoints)
batch = 4
epochs = 20500 #- ( 90 )
input_shape = (size, size, size, 1)
learning_rate = 3e-4
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
        prediction_image = tf.summary.image("Prediction",
                                            show_image(y_pred),
                                            max_outputs=8)
        label_image = tf.summary.image("Label",
                                       show_image(labels),
                                       max_outputs=8)
        input_image = tf.summary.image("Input",
                                       show_image(features),
                                       max_outputs=8)
        loss = loss_funtion(tf.cast(labels, tf.float32), tf.cast(y_pred, tf.float32))
        summary_loss = tf.summary.scalar('loss', loss)
        tf.summary.merge([prediction_image, label_image, input_image, summary_loss])
        if training:
            # params["learning_rate"] = params["learning_rate"] * .99
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


config = tf.ConfigProto()
config.gpu_options.allow_growth = True


params = {"learning_rate": learning_rate}
# set up the configurations of the estimator
config = tf.estimator.RunConfig(
    save_summary_steps=100,
    save_checkpoints_steps=100,
    keep_checkpoint_max=1,
    session_config=config
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
    model.train(input_fn=lambda: train_inputs(batch))
    print(model.evaluate(input_fn=lambda: eval_inputs(batch)))

print('Trainned finished')
