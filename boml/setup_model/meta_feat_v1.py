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
The base class in setup_model to encapsulate C4L neural network for meta-feature-based methods.
"""
from collections import OrderedDict
from functools import reduce

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python import layers

from boml import extension
from boml.setup_model import network_utils
from boml.setup_model.network import BOMLNet


class BOMLNetMetaFeatV1(BOMLNet):
    def __init__(
        self,
        _input,
        name="BMLNetC4LMetaFeat",
        outer_param_dict=OrderedDict(),
        model_param_dict=OrderedDict(),
        task_parameter=None,
        use_t=False,
        use_warp=False,
        outer_method="Reverse",
        dim_output=-1,
        activation=tf.nn.relu,
        var_collections=extension.METAPARAMETERS_COLLECTIONS,
        conv_initializer=tf.contrib.layers.xavier_initializer_conv2d(tf.float32),
        output_weight_initializer=tf.contrib.layers.xavier_initializer(tf.float32),
        norm=layers.batch_norm,
        data_type=tf.float32,
        channels=1,
        dim_hidden=[64, 64, 64, 64],
        kernel=3,
        max_pool=False,
        reuse=False,
    ):
        """
        :param _input: original input
        :param dim_output: dimension of output
        :param name: scope of meta-learner
        :param outer_param_dict: dictionary of outer parameters
        :param model_param_dict:dictonary of model parameters for specific algorithms such t-layer or warp-layer
        :param task_parameter: dictionary of task-specific parameters or temporary values of task-specific parameters
        :param use_t: Boolean, whether to use t-layer for neural network construction
        :param use_warp: Boolean, whether to use warp-layer for neural network construction
        :param outer_method: the name of outer method
        :param activation: form of activation function
        :param var_collections: collection to store variables
        :param conv_initializer: initializer for convolution blocks
        :param output_weight_initializer: initializer for the fully-connected layer
        :param norm: form of normalization function
        :param data_type: default to be tf.float32
        :param channels: number of channels
        :param dim_hidden: lists to specify the dimension of hidden layer
        :param kernel: size of the kernel
        :param max_pool: Boolean, whether to use max_pool
        :param reuse: Boolean, whether to reuse the parameters
        """
        self.task_parameter = task_parameter
        self.dim_output = dim_output
        self.kernel = kernel
        self.channels = channels
        self.dim_hidden = dim_hidden
        self.datatype = data_type
        self.batch_norm = norm
        self.max_pool = max_pool
        self.stride = [1, 2, 2, 1]
        self.no_stride = [1, 1, 1, 1]
        self.activation = activation
        self.bias_initializer = tf.zeros_initializer(tf.float32)
        self.conv_initializer = conv_initializer
        self.output_weight_initializer = output_weight_initializer
        self.use_t = use_t
        self.use_warp = use_warp
        self.outer_method = outer_method
        self.flatten = False if self.outer_method == "Implicit" else True

        super(BOMLNetMetaFeatV1, self).__init__(
            _input=_input,
            outer_param_dict=outer_param_dict,
            var_collections=var_collections,
            name=name,
            model_param_dict=model_param_dict,
            reuse=reuse,
        )

        self.betas = self.filter_vars("beta")

        self.moving_means = self.filter_vars("moving_mean")
        self.moving_variances = self.filter_vars("moving_variance")

        if not reuse:
            extension.remove_from_collection(
                extension.GraphKeys.MODEL_VARIABLES, *self.moving_means
            )
            extension.remove_from_collection(
                extension.GraphKeys.MODEL_VARIABLES, *self.moving_variances
            )
            print(name, "MODEL CREATED")
        extension.remove_from_collection(
            extension.GraphKeys.METAPARAMETERS, *self.moving_means
        )
        extension.remove_from_collection(
            extension.GraphKeys.METAPARAMETERS, *self.moving_variances
        )

    def create_outer_parameters(self):
        """
        :param var_collections: name of collections to store the created variables.
        :return: dictionary to index the created variables.
        """
        for i in range(len(self.dim_hidden)):
            self.outer_param_dict["conv" + str(i)] = network_utils.get_conv_weight(
                self, i=i, initializer=self.conv_initializer
            )

        [
            tf.add_to_collections(extension.GraphKeys.METAPARAMETERS, hyper)
            for hyper in self.outer_param_dict.values()
        ]

        if len(self.model_param_dict) == 0 and callable(
            getattr(self, "create_model_parameters", None)
        ):
            self.create_model_parameters()

        return self.outer_param_dict

    def create_model_parameters(
        self, var_collections=extension.GraphKeys.METAPARAMETERS
    ):
        """
        :param var_collections: name of collections to store the created variables.
        :return: dictionary to index the created variables.
        """
        if self.use_t:
            # hyper parameters of transformation layer
            for i in range(len(self.dim_hidden)):
                self.model_param_dict[
                    "conv" + str(i) + "_z"
                ] = network_utils.get_identity(
                    self.dim_hidden[0], name="conv" + str(i) + "_z", conv=True
                )
        elif self.use_warp:
            for i in range(len(self.dim_hidden)):
                self.model_param_dict[
                    "conv" + str(i) + "_z"
                ] = network_utils.get_warp_weight(self, i, self.conv_initializer)
                self.model_param_dict[
                    "bias" + str(i) + "_z"
                ] = network_utils.get_warp_bias(self, i, self.bias_initializer)
        [
            tf.add_to_collections(var_collections, model_param)
            for model_param in self.model_param_dict.values()
        ]

        return self.model_param_dict

    def _forward(self):
        """
        _forward() uses defined convolutional neural networks with initial input
        :return:
        """
        for i in range(len(self.dim_hidden)):
            if self.use_t:
                self + network_utils.conv_block_t(
                    self,
                    conv_weight=self.outer_param_dict["conv" + str(i)],
                    conv_bias=None,
                    zweight=self.model_param_dict["conv" + str(i) + "_z"],
                )
            elif self.use_warp:
                self + network_utils.conv_block_warp(
                    self,
                    cweight=self.outer_param_dict["conv" + str(i)],
                    bweight=None,
                    zweight=self.model_param_dict["conv" + str(i) + "_z"],
                    zbias=self.model_param_dict["bias" + str(i) + "_z"],
                )
            else:
                self + network_utils.conv_block(
                    self,
                    cweight=self.outer_param_dict["conv" + str(i)],
                    bweight=None,
                )
        if self.flatten:
            flattened_shape = reduce(
                lambda a, v: a * v, self.layers[-1].get_shape().as_list()[1:]
            )
            self + tf.reshape(
                self.out, shape=(-1, flattened_shape), name="representation"
            )
        else:
            if self.max_pool:
                self + tf.reshape(
                    self.out,
                    [-1, np.prod([int(dim) for dim in self.out.get_shape()[1:]])],
                )
            else:
                self + tf.reduce_mean(self.out, [1, 2])

    def re_forward(self, new_input):
        """
        reuses defined convolutional networks with new input and update the output results
        :param new_input: new input with same shape as the old one
        :return: updated instance of BOMLNet
        """
        return BOMLNetMetaFeatV1(
            _input=new_input if new_input is not None else self.layers[0],
            name=self.name,
            activation=self.activation,
            outer_param_dict=self.outer_param_dict,
            model_param_dict=self.model_param_dict,
            dim_output=self.dim_output,
            task_parameter=self.task_parameter,
            use_warp=self.use_warp,
            use_t=self.use_t,
            var_collections=self.var_collections,
            dim_hidden=self.dim_hidden,
            output_weight_initializer=self.output_weight_initializer,
            max_pool=self.max_pool,
            reuse=True,
            outer_method=self.outer_method,
        )


def BOMLNetOmniglotMetaFeatV1(
    _input,
    outer_param_dict=OrderedDict(),
    model_param_dict=OrderedDict(),
    batch_norm=layers.batch_norm,
    name="BMLNetC4LOmniglot",
    use_t=False,
    dim_output=-1,
    use_warp=False,
    outer_method="Reverse",
    **model_args
):

    return BOMLNetMetaFeatV1(
        _input=_input,
        name=name,
        model_param_dict=model_param_dict,
        dim_output=dim_output,
        outer_param_dict=outer_param_dict,
        norm=batch_norm,
        use_t=use_t,
        use_warp=use_warp,
        outer_method=outer_method,
        **model_args
    )


def BOMLNetMiniMetaFeatV1(
    _input,
    outer_param_dict=OrderedDict(),
    model_param_dict=OrderedDict(),
    dim_output=-1,
    batch_norm=layers.batch_norm,
    name="BOMLNetC4LMini",
    use_t=False,
    use_warp=False,
    outer_method="Reverse",
    **model_args
):
    return BOMLNetMetaFeatV1(
        _input=_input,
        name=name,
        use_t=use_t,
        use_warp=use_warp,
        dim_output=dim_output,
        outer_param_dict=outer_param_dict,
        model_param_dict=model_param_dict,
        norm=batch_norm,
        channels=3,
        dim_hidden=[32, 32, 32, 32],
        max_pool=True,
        outer_method=outer_method,
        **model_args
    )
