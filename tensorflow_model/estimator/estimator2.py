import tensorflow as tf
from .model import Network
from .imagenes import Image_shower
import os
from tensorflow.core.protobuf import rewriter_config_pb2


class Estimator:
    def __init__(self, learning_rate, shape, version, train=True):
        os.environ['TF_ENABLE_MIXED_PRECISION'] = '1'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.graph_options.rewrite_options.dependency_optimization = rewriter_config_pb2.RewriterConfig.OFF
        config = tf.estimator.RunConfig(
            save_summary_steps=10,
            save_checkpoints_steps=250,
            keep_checkpoint_max=10,
            session_config=config
        )
        self.network = Network()
        self.shape = shape
        self.B1 = 0.9
        self.B2 = 0.99
        self.disk = 'D:/'
        self.main_path = 'logs/' + str(shape) + 'version' + str(version)
        self.eval_path = 'logs/' + str(shape) + 'version' + str(version) + 'evaluation'
        self.train = train
        self.learning_rate = learning_rate
        self._estimator = tf.estimator.Estimator(model_fn=self.estimator_function,
                                                 params={"learning_rate": learning_rate},
                                                 model_dir=self.main_path,
                                                 config=config)
        self.loss = 0
        self.images = Image_shower(shape)
        self.count = 0
        self.grads = 0
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                          beta1=self.B1, beta2=self.B2, epsilon=1e-8)

    def loss_funtion(self, labels, preds):
        l2 = tf.losses.mean_squared_error(labels, preds)
        return l2

    def update_grads(self, loss):
        self.count += 1
        if self.count != 1:
            with tf.GradientTape() as tape:
                self.grads += tape.gradient(loss, self._estimator.get_variable_names())

            if self.count % 16 == 0:
                train_op = self.opt.apply_gradients(zip(self.grads/16, self._estimator.get_variable_names()))
                self.count = 0
                self.grads = 0
                return train_op
        return tf.train.get_global_step()

    def estimator_function(self, features, labels, mode, params):
        global step
        if mode == tf.estimator.ModeKeys.PREDICT:
            y_pred = self.network.main(features)
            spec = tf.estimator.EstimatorSpec(mode=mode, predictions=y_pred)
        # For training and testing
        else:
            training = mode == tf.estimator.ModeKeys.TRAIN
            y_pred = self.network.main(features)
            # summary the training image
            tf.summary.image("Summary", self.images.show_summary(y_pred, labels), max_outputs=8)
            # tf.summary.image("Difference", self.images.show_difference(y_pred, labels), max_outputs=8)
            loss = self.loss_funtion(tf.cast(labels, tf.float32), tf.cast(y_pred, tf.float32))
            tf.summary.scalar('loss', loss)
            diference_max = tf.reduce_max(labels - y_pred)
            diference_min = tf.reduce_min(labels - y_pred)
            diference_mean = tf.reduce_mean(labels - y_pred)
            tf.summary.scalar('diference_max', diference_max)
            tf.summary.scalar('diference_min', diference_min)
            tf.summary.scalar('diference_mean', diference_mean)
            tf.summary.scalar('learning_rate', self.learning_rate)
            if training:
                tf.summary.merge_all()
                with tf.name_scope('adam_optimizer'):
                    # train_op = self.opt.minimize(loss=loss, global_step=tf.train.get_global_step())
                    train_op = self.update_grads(loss)
                spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, predictions=y_pred)
            else:
                tf.summary.image("Summary_eval", self.images.show_summary(y_pred, labels), max_outputs=8)
                tf.summary.scalar('eval_loss', loss)
                tf.summary.merge_all()
                spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, predictions=y_pred)
        return spec

    def entrenamiento(self, train_spec, eval_spec):
        def estimator_training(args):
            del args
            tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

        model = self._estimator
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.app.run(estimator_training)
