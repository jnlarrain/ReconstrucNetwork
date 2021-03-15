import tensorflow as tf
from .new_model import Network
from .imagenes import ImageShower
from .perceptual_loss import PerceptualLoss


class Estimator:
    def __init__(self, learning_rate, shape, version):

        session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config = tf.estimator.RunConfig(save_summary_steps=10, save_checkpoints_steps=250,
                                        keep_checkpoint_max=10, session_config=session_config)
        self.network = Network()
        self.main_path = 'logs/' + str(shape) + 'version' + str(version)
        self.eval_path = 'logs/' + str(shape) + 'version' + str(version) + 'evaluation'
        self.learning_rate = learning_rate
        self._estimator = tf.estimator.Estimator(model_fn=self.estimator_function, model_dir=self.main_path,
                                                 config=config)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        self.images = ImageShower(shape)
        self.loss_model = None

    def estimator_function(self, features, labels, mode):
        y_pred = self.network.main(features)
        if mode == tf.estimator.ModeKeys.PREDICT:

            return tf.estimator.EstimatorSpec(mode=mode, predictions=y_pred)
        else:
            self.loss_model = PerceptualLoss()
            training = mode == tf.estimator.ModeKeys.TRAIN
            loss = self.loss_model.loss(labels, y_pred)
            with tf.device('cpu:0'):
                sum_loss = tf.compat.v1.summary.scalar('loss', loss)
                sum_img = tf.compat.v1.summary.image("Training", self.images.show_summary(y_pred, labels), max_outputs=8)
                sum_input = tf.compat.v1.summary.image("Entrance", self.images.show_image_cuts(features), max_outputs=8)
            tf.compat.v1.summary.merge([sum_input, sum_img, sum_loss])
            if training:
                train_op = self.optimizer.minimize(loss, tf.compat.v1.train.get_global_step(),
                                                   colocate_gradients_with_ops=True)
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, predictions=y_pred)
            else:
                img = tf.compat.v1.summary.image("Evaluation", self.images.show_summary(y_pred, labels), max_outputs=8)
                ev = tf.estimator.SummarySaverHook(
                    save_steps=1, output_dir=self.main_path,
                    summary_op=[tf.compat.v1.summary.scalar('loss', loss),img])
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, predictions=y_pred, evaluation_hooks=[ev])

    def entrenamiento(self, train_spec, eval_spec):
        tf.estimator.train_and_evaluate(self._estimator, train_spec, eval_spec)
