"""
The base class in setup_model to encapsulate the fully-connected layer.
"""
from collections import OrderedDict

import tensorflow as tf
from boml.extension import GraphKeys
from boml.setup_model.network import BOMLNet
from boml.utils import as_tuple_or_list, remove_from_collection


class BOMLNetFeedForward(BOMLNet):
    def __init__(
        self,
        _input,
        dims,
        task_parameter=OrderedDict(),
        name="BMLNetFeedForward",
        activation=tf.nn.relu,
        var_collections=tf.GraphKeys.MODEL_VARIABLES,
        output_weight_initializer=tf.contrib.layers.xavier_initializer(tf.float32),
        data_type=tf.float32,
        reuse=False,
    ):
        """

        :param _input: original input for the FeedForward network,
        which is also output of the convolutional neural networks
        :param dims: the dimension of the final output
        :param task_parameter: dictionary to store the task-specific parameters
        :param name: name for the task-specific network
        :param activation: activation function, default to be tf.nn.relu
        :param var_collections: collections to manage the parameters of
         base-learner in the computational graph
        :param output_weight_initializer: function to initialize the weights of FeedForward network
        :param data_type: default to be tf.float32
        :param reuse: whether to reuse the created parameters in the named scope
        """
        self.dims = as_tuple_or_list(dims)
        self.activation = activation
        self.data_type = data_type
        self.task_parameter = task_parameter
        self.var_collections = var_collections
        self.output_weight_initializer = output_weight_initializer
        super().__init__(
            _input=_input, name=name, var_collections=var_collections, reuse=reuse,
        )
        if not reuse:
            print(name, "MODEL CREATED")

    def _forward(self):

        if len(self.task_parameter) == 0:
            self.create_initial_parameter()
        self + tf.add(
            tf.matmul(self.out, self.task_parameter["fc_weight"]),
            self.task_parameter["fc_bias"],
        )

    def create_initial_parameter(self):
        self.task_parameter = OrderedDict()
        self.task_parameter["fc_weight"] = tf.get_variable(
            "fc_weight",
            shape=[self.layers[-1].shape.as_list()[-1], self.dims[-1]],
            initializer=self.output_weight_initializer,
            dtype=self.data_type,
        )
        self.task_parameter["fc_bias"] = tf.get_variable(
            "fc_bias",
            [self.dims[-1]],
            initializer=tf.zeros_initializer(tf.float32),
            dtype=self.data_type,
        )
        [
            tf.add_to_collections(self.var_collections, initial_param)
            for initial_param in self.task_parameter.values()
        ]
        remove_from_collection(
            GraphKeys.GLOBAL_VARIABLES, *self.task_parameter.values()
        )

    def re_forward(self, new_input=None, task_parameter=OrderedDict()):
        return BOMLNetFeedForward(
            new_input if new_input is not None else self.layers[0],
            dims=self.dims,
            task_parameter=self.task_parameter
            if len(task_parameter) == 0
            else task_parameter,
            name=self.name,
            activation=self.activation,
            data_type=self.data_type,
            var_collections=self.var_collections,
            output_weight_initializer=self.output_weight_initializer,
            reuse=True,
        )
