import tensorflow as tf

from base_TrainingManager import BaseTrainingManager


class BaseModelGraph:

    def __init__(self, base_training_manager: BaseTrainingManager):
        self.tf_graph = tf.Graph()
        self.trainingManager = base_training_manager

        with self.tf_graph.as_default():
            self.build_graph()

            if not base_training_manager.is_local_env:
                self.make_saver()

            self.variable_initializer()

    def make_saver(self):
        raise NotImplementedError

    def variable_initializer(self):
        raise NotImplementedError

    def build_graph(self):
        raise NotImplementedError
