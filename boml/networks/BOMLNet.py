import sys
from collections import OrderedDict

import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.python.training import slot_creator

from boml.extension import GraphKeys
from boml.networks.network_utils import filter_vars
from boml.utils import remove_from_collection, as_tuple_or_list


class BOMLNet(object):
    # definitely deprecated!
    """
    Base object for models
    """

    def __init__(self, _input, outer_param_dict=OrderedDict(), model_param_dict=OrderedDict(),task_parameter=None, var_collections=None, name=None,
                 deterministic_initialization=False, reuse=False):
        """
        Creates an object that represent a network. Important attributes of a Network object are

        `var_list`: list of tf.Variables that constitute the parameters of the model

        `inp`: list, first element is `_input` and last should be output of the model. Other entries can be
        hidden layers activations.

        :param _input: tf.Tensor, input of this model.
        """
        super(BOMLNet, self).__init__()

        if not name:
            try:
                name = tf.get_variable_scope().name
            except IndexError:
                print('Warning: no name and no variable scope given', sys.stderr)
        self.outer_param_dict = outer_param_dict
        self.model_param_dict = model_param_dict
        self.task_parameter = task_parameter
        self.var_collections = var_collections
        self.name = name
        self.reuse = reuse
        self.deterministic_initialization = deterministic_initialization
        self._var_list_initial_values = []
        self._var_init_placeholder = None
        self._assign_int = []
        self._var_initializer_op = None

        self.layers = [_input]
        # self.s = None
        self._tf_saver = None

        with self._variable_scope(reuse):
            if (len(self.outer_param_dict) == 0) and (callable(getattr(self, 'create_outer_parameters', None))):
                self.create_outer_parameters()
            self._forward()

    def _variable_scope(self, reuse):
        """
        May override default variable scope context, but pay attention since it looks like
        initializer and default initializer form contrib.layers do not work well together (I think this is because
        functions in contrib.layers usually specify an initializer....

        :param reuse:
        :return:
        """
        return tf.variable_scope(self.name, reuse=reuse)

    def __getitem__(self, item):
        """
        Get's the `activation`

        :param item:
        :return:
        """
        return self.layers[item]

    def __add__(self, other):
        if self.name not in tf.get_variable_scope().name:
            print('Warning: adding layers outside model variable scope', file=sys.stderr)
        self.layers.append(other)
        return self

    @property
    def var_list(self):
        if len(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, self.name)) == 0:
            assert self.task_parameter is not None, 'No Model Variables to optimize, ' \
                                                 'please double check your computational graph'
            return list(self.task_parameter.values())
        else:
            return tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, self.name)

    @property
    def Ws(self):
        return self.filter_vars('weights')

    @property
    def bs(self):
        return self.filter_vars('biases')

    @property
    def out(self):
        return self[-1]

    def create_initial_parameter(self, primary_outerparameter=None):
        assert primary_outerparameter is not None, 'Primary hyperparameters ' \
                                                   'must be provided for initialization of slot variables'
        # task_weights_keys = [key for key in primary_hyperparameter.keys() if 'prob' not in key and 'z' not in key]
        initial_parameter = OrderedDict([(primary_key, slot_creator.create_slot(primary=self.out,
                                                                                val=primary_outerparameter[primary_key].initialized_value(),
                                                                                name=primary_key)) for primary_key in primary_outerparameter.keys()])
        [tf.add_to_collection(self.var_collections, initial_param) for initial_param in initial_parameter.values()]
        remove_from_collection(GraphKeys.GLOBAL_VARIABLES, *initial_parameter.values())
        return initial_parameter

    def filter_vars(self, var_name):
        return filter_vars(var_name, self.name)

    def initialize(self, session=None):
        """
        Initialize the model. If `deterministic_initialization` is set to true,
        saves the initial weight in numpy which will be used for subsequent initialization.
        This is because random seed management in tensorflow is rather obscure... and I could not
        find a way to set the same seed across different initialization without exiting the session.

        :param session:
        :return:
        """
        # TODO placeholder are not useful at all here... just change initializer of tf.Variables !
        ss = session or tf.get_default_session()
        assert ss, 'No default session'
        if not self._var_initializer_op:
            self._var_initializer_op = tf.variables_initializer(self.var_list)
        ss.run(self._var_initializer_op)
        if self.deterministic_initialization:
            if self._var_init_placeholder is None:
                self._var_init_placeholder = tf.placeholder(tf.float32)
                self._assign_int = [v.assign(self._var_init_placeholder) for v in self.var_list]

            if not self._var_list_initial_values:
                self._var_list_initial_values = ss.run(self.var_list)

            else:
                [ss.run(v_op, feed_dict={self._var_init_placeholder: val})
                 for v_op, val in zip(self._assign_int, self._var_list_initial_values)]

    def _variables_to_save(self):
        return self.var_list

    def _forward(self):
        '''
        _forward() uses defined convolutional neural networks with initial input
        :return:
        '''
        raise NotImplemented()

    def re_forward(self, new_input):
        '''
        reuses defined convolutional with new input and update the output results
        :param new_input: new input with same shape as the old one
        :return:
        '''
        raise NotImplemented()

    @property
    def tf_saver(self):
        if not self._tf_saver:
            self._tf_saver = tf.train.Saver(var_list=self._variables_to_save(), max_to_keep=1)
        return self._tf_saver

    def save(self, file_path, session=None, global_step=None):
        # TODO change this method! save the class (or at least the attributes you can save
        self.tf_saver.save(session or tf.get_default_session(), file_path, global_step=global_step)

    def restore(self, file_path, session=None, global_step=None):
        if global_step: file_path += '-' + str(global_step)
        self.tf_saver.restore(session or tf.get_default_session(), file_path)


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
