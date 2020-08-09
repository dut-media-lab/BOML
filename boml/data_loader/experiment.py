"""
Simple container for useful quantities for a supervised learning experiment, where data is managed
with feed dictionary
"""
import tensorflow as tf


class BMLExperiment:

    def __init__(self, datasets, dtype=tf.float32):
        '''

        :param datasets: initialized instance of Dataloader
        :param dtype: default tf.float32

        Attributes:
          - x: input placeholder of input for your defined lower level problem
          - y: label placeholder of output for yourdefined lower level problem
          - x_:input placeholder of input for your defined upper level problem
          - y_:label placeholder of output for your defined upper level problem
          - model: used to restore the task-specific model
          - errors: dictionary to restore defined loss functions of different levels
          - scores: dictionary to restore defined accuracies functions
          - optimizers: dictonary to restore optimized chosen for inner and outer loop optimization
        '''
        self.datasets = datasets
        self.x = tf.placeholder(dtype, name='x', shape=self._compute_input_shape())
        self.y = tf.placeholder(dtype, name='y', shape=self._compute_output_shape())
        self.x_ = tf.placeholder(dtype, name='x_', shape=self._compute_input_shape())
        self.y_ = tf.placeholder(dtype, name='y_', shape=self._compute_output_shape())
        self.dtype = dtype
        self.model = None
        self.errors = {}
        self.scores = {}
        self.optimizers = {}

    # noinspection PyBroadException
    def _compute_input_shape(self):
        try:
            sh = self.datasets.train.dim_data
            return (None, sh) if isinstance(sh, int) else (None,) + sh
        except:
            print('Could not determine input dimension')
            return None

    # noinspection PyBroadException
    def _compute_output_shape(self):
        try:
            sh = self.datasets.train.dim_target
            return (None, sh) if isinstance(sh, int) else (None,) + sh
        except:
            print('Could not determine output dimension')
            return None
