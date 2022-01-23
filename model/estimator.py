import tensorflow as tf
from .model import Network
from .tensorboard_images import ImageShower
from .losses import Loss
from pathlib import Path


class Estimator:
    def __init__(self, learning_rate, shape, version):
        # central_storage_strategy = tf.distribute.experimental.CentralStorageStrategy()
        # session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        # config = tf.estimator.RunConfig(
        #     save_summary_steps=10,
        #     save_checkpoints_steps=250,
        #     keep_checkpoint_max=10,
        #     session_config=session_config,
        #     train_distribute=central_storage_strategy,
        #     eval_distribute=central_storage_strategy,
        # )
        session_config = tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=12,
            allow_soft_placement=True,
        )
        config = tf.estimator.RunConfig(
            save_summary_steps=10,
            save_checkpoints_steps=250,
            keep_checkpoint_max=10,
            session_config=session_config,
        )
        self.network = Network()
        self.main_path = str(
            Path().joinpath("logs", "version_" + str(version)).absolute()
        )
        self.eval_path = str(
            Path().joinpath("logs", "version_" + str(version) + "evaluation").absolute()
        )
        self.learning_rate = learning_rate
        self._estimator = tf.estimator.Estimator(
            model_fn=self.estimator_function, model_dir=self.main_path, config=config
        )
        # self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.99)
        self.images = ImageShower(shape)
        self.loss_model = Loss()

    def estimator_function(self, features, labels, mode):

        y_pred = self.network.main(features)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=y_pred)

        else:
            training = mode == tf.estimator.ModeKeys.TRAIN

            with tf.device("cpu:0"):
                loss = self.loss_model.loss(labels, y_pred)
                sum_loss = tf.compat.v1.summary.scalar("loss", loss)
                # sum_img = tf.compat.v1.summary.image("Training", self.images.show_summary(y_pred, labels), max_outputs=8)
                # sum_in = tf.compat.v1.summary.image("Entrance", self.images.show_image_cuts(tf.expand_dims(
                #           features[..., 0], -1)), max_outputs=8)
                tf.compat.v1.summary.merge([sum_loss])

            if training:
                train_op = tf.compat.v1.train.AdamOptimizer(
                    learning_rate=self.learning_rate, beta1=0.9, beta2=0.99
                ).minimize(loss, tf.compat.v1.train.get_global_step())
                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=loss, train_op=train_op, predictions=y_pred
                )

            else:
                # with tf.device('cpu:0'):
                # img = tf.compat.v1.summary.image("Evaluation", self.images.show_summary(y_pred, labels), max_outputs=8)
                ev = tf.estimator.SummarySaverHook(
                    save_steps=1,
                    output_dir=self.main_path,
                    summary_op=[
                        tf.compat.v1.summary.scalar("loss", loss),
                    ],
                )
                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=loss, predictions=y_pred, evaluation_hooks=[ev]
                )

    def train_loop(self, train_spec, eval_spec):
        tf.estimator.train_and_evaluate(self._estimator, train_spec, eval_spec)
