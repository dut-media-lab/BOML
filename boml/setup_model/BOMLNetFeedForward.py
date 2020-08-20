from collections import OrderedDict

import tensorflow as tf

from boml import BOMLNet
from boml.extension import GraphKeys
from boml.utils import as_tuple_or_list, remove_from_collection


class BOMLNetFeedForward(BOMLNet):
    def __init__(self, _input, dims, task_parameter=None,name='BMLNetFeedForward', activation=tf.nn.relu,
                 var_collections=tf.GraphKeys.MODEL_VARIABLES,
                 output_weight_initializer=tf.contrib.layers.xavier_initializer(tf.float32), data_type=tf.float32,
                 deterministic_initialization=False, reuse=False,use_T=False):
        self.dims = as_tuple_or_list(dims)
        self.activation = activation
        self.data_type = data_type
        self.var_collections = var_collections
        self.output_weight_initializer = output_weight_initializer
        self.use_T = use_T
        super().__init__(_input=_input, name=name, var_collections=var_collections,task_parameter=task_parameter,
                         deterministic_initialization=deterministic_initialization, reuse=reuse)
        if not reuse:
            print(name, 'MODEL CREATED')

    def _forward(self):
        '''
        self + tcl.fully_connected(self.out, self.dims[-1], activation_fn=None,
                                   weights_initializer=self.output_weight_initializer,
                                   variables_collections=self.var_collections, trainable=False)
        '''
        if not isinstance(self.task_parameter, dict):
            self.create_initial_parameter()
        self + tf.add(tf.matmul(self.out, self.task_parameter['fc_weight']), self.task_parameter['fc_bias'])

        if self.use_T:
            conv_z = tf.get_variable(initializer=tf.eye(self.dims[-1]), dtype=tf.float32,
                                     collections=self.var_collections, trainable=False, name='conv_z')
            self + tf.matmul(self.out, conv_z)

    def create_initial_parameter(self):
        self.task_parameter = OrderedDict()
        self.task_parameter['fc_weight'] = tf.get_variable('fc_weight', shape=
        [self.layers[-1].shape.as_list()[-1], self.dims[-1]], initializer=self.output_weight_initializer,
                                                           dtype=self.data_type)
        self.task_parameter['fc_bias'] = tf.get_variable('fc_bias', [self.dims[-1]], initializer=tf.zeros_initializer,
                                                         dtype=self.data_type)
        [tf.add_to_collections(self.var_collections, initial_param) for initial_param in self.task_parameter.values()]
        remove_from_collection(GraphKeys.GLOBAL_VARIABLES, *self.task_parameter.values())

    def re_forward(self, new_input=None, task_parameter=OrderedDict()):
        return BOMLNetFeedForward(new_input if new_input is not None else self.layers[0], dims=self.dims,
                                  task_parameter=self.task_parameter if self.task_parameter is not None
                                  else task_parameter, name=self.name, activation=self.activation,
                                  data_type=self.data_type, var_collections=self.var_collections,
                                  output_weight_initializer=self.output_weight_initializer,
                                  deterministic_initialization=self.deterministic_initialization, reuse=True, use_T=self.use_T)