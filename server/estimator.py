import tensorflow as tf
from .model import Network
from .imagenes import ImageShower


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

    @staticmethod
    def loss_function(labels, preds):
        l2 = tf.subtract(labels, preds)
        l2 = tf.square(l2)
        l2 = tf.reduce_sum(l2)
        return l2

    @staticmethod
    def regulator(volumen, alpha=1e-7):
        pixel_dif1 = volumen[:, 1:, :, :, :] - volumen[:, :-1, :, :, :]
        pixel_dif2 = volumen[:, :, 1:, :, :] - volumen[:, :, :-1, :, :]
        pixel_dif3 = volumen[:, :, :, 1:, :] - volumen[:, :, :, :-1, :]
        total_var = tf.reduce_sum(
            tf.reduce_sum(tf.abs(pixel_dif1)) +
            tf.reduce_sum(tf.abs(pixel_dif2)) +
            tf.reduce_sum(tf.abs(pixel_dif3))
        )
        return total_var * alpha

    def estimator_function(self, features, labels, mode):
        y_pred = self.network.main(features)
        if mode == tf.estimator.ModeKeys.PREDICT:

            return tf.estimator.EstimatorSpec(mode=mode, predictions=y_pred)
        else:
            training = mode == tf.estimator.ModeKeys.TRAIN
            loss = self.loss_function(labels, y_pred) + self.regulator(y_pred, 2e-6) #1.2e-5 + tf.reduce_sum(tf.abs(labels-y_pred))/250
            sum_loss = tf.compat.v1.summary.scalar('loss', loss)
            sum_img = tf.compat.v1.summary.image("Training", self.images.show_summary(y_pred, labels), max_outputs=8)
            sum_input = tf.compat.v1.summary.image("Entrance", self.images.show_image_cuts(features), max_outputs=8)
            tf.compat.v1.summary.merge([sum_input, sum_img, sum_loss])
            if training:
                train_op = self.optimizer.minimize(loss, tf.compat.v1.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, predictions=y_pred)
            else:
                img = tf.compat.v1.summary.image("Evaluation", self.images.show_summary(y_pred, labels), max_outputs=8)
                ev = tf.estimator.SummarySaverHook(
                    save_steps=1, output_dir=self.main_path,
                    summary_op=[tf.compat.v1.summary.scalar('loss', loss),img])
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, predictions=y_pred, evaluation_hooks=[ev])

    def entrenamiento(self, train_spec, eval_spec):
        tf.estimator.train_and_evaluate(self._estimator, train_spec, eval_spec)
