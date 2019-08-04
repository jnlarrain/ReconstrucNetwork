import tensorflow as tf
from model import network
from tfrecords import read_and_decode
import os

os.environ['TF_ENABLE_MIXED_PRECISION'] = '1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config = tf.estimator.RunConfig(
    save_summary_steps=100,
    save_checkpoints_steps=500,
    keep_checkpoint_max=1,
    session_config=config
)

class Estimator:
    def __init__(self, path, steps, batch, learning_rate, size, version, train=True):
        self.train_path = path + '_train.tfrecords'
        self.evaluation_path = path + '_test.tfrecords'
        self.epochs = steps // batch
        self.batch = batch
        self.steps = steps
        self.size = size
        self.B1 = 0.9
        self.B2 = 0.99
        self.disk = 'D:/'
        self.main_path = 'logs/mainOne' + str(size) + 'version' + str(version)
        self.eval_path = 'logs/evaluationOne' + str(size) + 'version' + str(version)
        self.train = train
        self.learning_rate = learning_rate

    def train_inputs(self, batch_size, num_shuffles=100):
        dataset = read_and_decode(self.train_path)

        # shuffle and repeat examples for better randomness and allow training beyond one epoch
        dataset = dataset.repeat(self.epochs)
        dataset = dataset.shuffle(num_shuffles)

        # map the parse  function to each example individually in threads*2 parallel calls

        # batch the examples
        dataset = dataset.batch(batch_size=batch_size)

        # prefetch batch

        dataset = dataset.prefetch(buffer_size=batch_size)
        return dataset

    def show_image(self, preds):
        result = tf.concat([preds[:, self.size // 2, :, :, :],
                            preds[:, :, self.size // 2, :, :],
                            preds[:, :, :, self.size // 2, :]], 1)
        return result

    def show_summary(self, images, labels):
        diff = labels - images
        new_images = tf.concat([images[:, self.size // 2, :, :, :],
                                images[:, :, self.size // 2, :, :],
                                images[:, :, :, self.size // 2, :]], 1)
        new_diff = tf.concat([diff[:, self.size // 2, :, :, :],
                              diff[:, :, self.size // 2, :, :],
                              diff[:, :, :, self.size // 2, :]], 1)
        new_labels = tf.concat([labels[:, self.size // 2, :, :, :],
                                labels[:, :, self.size // 2, :, :],
                                labels[:, :, :, self.size // 2, :]], 1)
        maximum = tf.ones(tf.shape(new_labels)) * tf.reduce_max(diff)
        minimum = tf.ones(tf.shape(new_labels)) * tf.reduce_min(diff)
        zeros = tf.zeros(tf.shape(new_labels))
        scale = tf.concat([maximum[:, :self.size, :2, :], zeros[:, :self.size, :2, :], minimum[:, :self.size, :2, :]],
                          1)
        result = tf.concat([new_labels[:, :, :, :],
                            new_images[:, :, :, :],
                            new_diff[:, :, :, :], scale], 2)
        return result

    def eval_inputs(self, batch_size):
        dataset = read_and_decode(self.evaluation_path)
        dataset = dataset.shuffle(100).batch(batch_size)
        return dataset

    def loss_funtion(self, labels, preds):
        # l1 = tf.losses.absolute_difference(labels, preds)
        l2 = tf.losses.mean_squared_error(labels, preds)
        return l2

    def estimator_function(self, features, labels, mode, params):
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
                                              self.show_summary(y_pred, labels),
                                              max_outputs=8)
            loss = self.loss_funtion(tf.cast(labels, tf.float32), tf.cast(y_pred, tf.float32))
            summary_loss = tf.summary.scalar('loss', loss)
            tf.summary.merge([summary_loss, summary_images])
            if training:
                params["learning_rate"] = params["learning_rate"] * .99
                optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"], beta1=self.B1, beta2=self.B2)
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
                    output_dir=self.eval_path,
                    summary_op=[tf.summary.scalar('loss', loss),
                                tf.summary.image("Summary_eval",
                                                 self.show_summary(y_pred, labels),
                                                 max_outputs=8)]
                )

                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    predictions=y_pred,
                    evaluation_hooks=[evaluation_hook]
                )
        return spec

   def training(self):

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
