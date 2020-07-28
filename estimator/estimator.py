import tensorflow as tf
from .model import Network
from .imagenes import Image_shower


class Estimator:
    def __init__(self, learning_rate, shape, version, batch_size=32):
        config = tf.estimator.RunConfig(save_summary_steps=10, save_checkpoints_steps=500, keep_checkpoint_max=10)
        self.network = Network()
        self.shape = shape
        self.batch = batch_size
        self.B1 = 0.9
        self.B2 = 0.99
        self.disk = 'D:/'
        self.main_path = 'logs/' + str(shape) + 'version' + str(version)
        self.eval_path = 'logs/' + str(shape) + 'version' + str(version) + 'evaluation'
        self.learning_rate = learning_rate
        self._estimator = tf.estimator.Estimator(model_fn=self.estimator_function,
                                                 params={"learning_rate": learning_rate},
                                                 model_dir=self.main_path, config=config)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=self.B1, beta2=self.B2)
        self.loss = 0
        self.images = Image_shower(shape)
        self.count = 0
        self.grads = 0

    @staticmethod
    def loss_function(labels, preds):
        l2 = tf.subtract(labels, preds)
        l2 = tf.square(l2)
        l2 = tf.reduce_sum(l2)
        return l2

    # @staticmethod
    #     # def loss_function_regulator(labels, preds, alpha=1e-7):
    #     #     l2 = tf.losses.mean_squared_error(labels, preds)
    #     #     reg = tf.norm(preds, ord=2, axis=-1)
    #     #     return l2 + alpha * reg

    def estimator_function(self, features, labels, mode, params):
        y_pred = self.network.main(features)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=y_pred)
        else:
            training = mode == tf.estimator.ModeKeys.TRAIN
            loss = self.loss_function(labels, y_pred)
            sum_loss = tf.compat.v1.summary.scalar('loss', loss)
            sum_img = tf.compat.v1.summary.image("Summary", self.images.show_summary(y_pred, labels), max_outputs=8)
            sum_input = tf.compat.v1.summary.image("Entrance", self.images.show_image_cuts(features), max_outputs=8)
            tf.compat.v1.summary.merge([sum_input, sum_img, sum_loss])
            if training:
                train_op = self.optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    train_op=train_op,
                    predictions=y_pred
                )
            else:
                evaluation_hook = tf.estimator.SummarySaverHook(
                    save_steps=1,
                    output_dir=self.main_path,
                    summary_op=[tf.compat.v1.summary.scalar('loss', loss),
                                tf.compat.v1.summary.image("Summary_eval",
                                                           self.images.show_summary(y_pred, labels),
                                                           max_outputs=8)]
                )

                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    predictions=y_pred,
                    evaluation_hooks=[evaluation_hook]
                )

    def entrenamiento(self, train_spec, eval_spec):
        tf.estimator.train_and_evaluate(self._estimator, train_spec, eval_spec)
