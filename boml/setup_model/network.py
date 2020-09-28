# MIT License

# Copyright (c) 2020 Yaohua Liu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
The base class in setup_model to encapsulate BOMLNet based on conventional gradient descent optimizers.  setup_model
defines network structure and initializes network parameters of meta-learner and base-learner on the basis of the data formats returned by load data.
"""
import sys
from collections import OrderedDict

import tensorflow as tf
from tensorflow.python.training import slot_creator

from boml.extension import GraphKeys
from boml.setup_model.network_utils import filter_vars
from boml.utils import remove_from_collection


class BOMLNet(object):
    # definitely deprecated!
    """
    Base object for building neural networks
    """

    def __init__(
        self,
        _input,
        outer_param_dict=OrderedDict(),
        model_param_dict=OrderedDict(),
        var_collections=None,
        name=None,
        reuse=False,
    ):
        """
        Creates an object that creates model parameters and defines the network structure.
        :param _input: the input shape for defined network
        :param outer_param_dict: type of OrderedDict to restore parameters to optimize in the upper level
        :param model_param_dict: type of OrderedDict to restore parameters of special model like Iteration Warping
        :param task_parameter: type of OrderedDict to restore parameters to optimize in the lower level
        """
        super(BOMLNet, self).__init__()

        if not name:
            try:
                name = tf.get_variable_scope().name
            except IndexError:
                print("Warning: no name and no variable scope given", sys.stderr)
        self.outer_param_dict = outer_param_dict
        self.model_param_dict = model_param_dict
        self.var_collections = var_collections
        self.name = name
        self.reuse = reuse
        self._var_initializer_op = None

        self.layers = [_input]
        # self.s = None
        self._tf_saver = None

        with self._variable_scope(reuse):
            if (len(self.outer_param_dict) == 0) and (
                callable(getattr(self, "create_outer_parameters", None))
            ):
                self.create_outer_parameters()
            self._forward()

    def _variable_scope(self, reuse):
        """
        May override default variable scope context.
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
            print(
                "Warning: adding layers outside model variable scope", file=sys.stderr
            )
        self.layers.append(other)
        return self

    @property
    def var_list(self):
        """
        :return: list that contains the variables created in the current scope
        """
        return tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, self.name)

    @property
    def out(self):
        """
        :return: the current output of the BOMLNet
        """
        return self[-1]

    def create_initial_parameter(self, primary_outerparameter=None):
        """
        :param primary_outerparameter: the primary outerparameters used to
        create the task-specific parameters
        :return: dictionary to keep the created task parameters
        """
        assert primary_outerparameter is not None, (
            "Primary hyperparameters "
            "must be provided for initialization of slot variables"
        )
        # task_weights_keys = [key for key in primary_hyperparameter.keys() if 'prob' not in key and 'z' not in key]
        initial_parameter = OrderedDict(
            [
                (
                    primary_key,
                    slot_creator.create_slot(
                        primary=self.out,
                        val=primary_outerparameter[primary_key].initialized_value(),
                        name=primary_key,
                    ),
                )
                for primary_key in primary_outerparameter.keys()
            ]
        )
        [
            tf.add_to_collection(self.var_collections, initial_param)
            for initial_param in initial_parameter.values()
        ]
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
        assert ss, "No default session"
        if not self._var_initializer_op:
            self._var_initializer_op = tf.variables_initializer(self.var_list)
        ss.run(self._var_initializer_op)

    def _forward(self):
        """
        _forward() uses defined convolutional neural networks with initial input
        :return:
        """
        raise NotImplemented()

    def re_forward(self, new_input):
        """
        reuses defined convolutional networks with new input and update the output results
        :param new_input: new input with same shape as the old one
        :return: updated instance of BOMLNet
        """
        raise NotImplemented()
