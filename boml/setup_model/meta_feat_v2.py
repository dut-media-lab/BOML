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
The base class in setup_model to encapsulate Residual Block for meta-feature-based methods.
"""
from collections import OrderedDict

import tensorflow as tf
from tensorflow.contrib import layers as tcl

import boml.extension
from boml.setup_model import network_utils
from boml.setup_model.network import BOMLNet


class BOMLNetMiniMetaFeatV2(BOMLNet):
    def __init__(
        self,
        _input,
        name="BOMLNetMiniMetaFeatV2",
        outer_param_dict=OrderedDict(),
        dim_output=-1,
        model_param_dict=OrderedDict(),
        task_parameter=OrderedDict(),
        use_t=False,
        use_warp=False,
        reuse=False,
        outer_method="Reverse",
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
        :param reuse: Boolean, whether to reuse the parameters
        """
        self.var_coll = boml.extension.METAPARAMETERS_COLLECTIONS
        self.task_paramter = task_parameter
        self.outer_method = outer_method
        self.dim_output = dim_output
        self.use_t = use_t
        self.use_warp = use_warp

        super().__init__(
            _input=_input,
            outer_param_dict=outer_param_dict,
            model_param_dict=model_param_dict,
            name=name,
            reuse=reuse,
        )

        self.betas = self.filter_vars("beta")
        self.moving_means = self.filter_vars("moving_mean")
        self.moving_variances = self.filter_vars("moving_variance")
        if not reuse:
            boml.extension.remove_from_collection(
                boml.extension.GraphKeys.MODEL_VARIABLES,
                *self.moving_means,
                *self.moving_variances
            )

        boml.extension.remove_from_collection(
            boml.extension.GraphKeys.METAPARAMETERS,
            *self.moving_means,
            *self.moving_variances
        )
        print(name, "MODEL CREATED")

    def _forward(self):
        """
        _forward() uses defined convolutional neural networks with initial input
        :return:
        """
        def residual_block(x, n_filters):
            skip_c = tcl.conv2d(x, n_filters, 1, activation_fn=None)

            def conv_block(xx):
                out = tcl.conv2d(
                    xx,
                    n_filters,
                    3,
                    activation_fn=None,
                    normalizer_fn=tcl.batch_norm,
                    variables_collections=self.var_coll,
                )
                return network_utils.leaky_relu(out, 0.1)

            out = x
            for _ in range(3):
                out = conv_block(out)

            add = tf.add(skip_c, out)

            return tf.nn.max_pool(add, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        self + residual_block(self.out, 64)
        self + residual_block(self.out, 96)
        self + residual_block(self.out, 128)
        self + residual_block(self.out, 256)
        self + tcl.conv2d(self.out, 2048, 1, variables_collections=self.var_coll)
        self + tf.nn.avg_pool(self.out, [1, 6, 6, 1], [1, 6, 6, 1], "VALID")
        self + tcl.conv2d(self.out, 512, 1, variables_collections=self.var_coll)
        self + tf.reshape(self.out, (-1, 512))

    def re_forward(self, new_input=None):
        """
        reuses defined convolutional networks with new input and update the output results
        :param new_input: new input with same shape as the old one
        :param task_parameter: the dictionary of task-specific
        :return: updated instance of BOMLNet
        """
        return BOMLNetMiniMetaFeatV2(
            _input=new_input if new_input is not None else self.layers[0],
            model_param_dict=self.model_param_dict,
            name=self.name,
            dim_output=self.dim_output,
            outer_param_dict=self.outer_param_dict,
            reuse=tf.AUTO_REUSE,
            outer_method=self.outer_method,
            use_t=self.use_t,
        )


class BOMLNetOmniglotMetaFeatV2(BOMLNet):
    def __init__(
        self,
        _input,
        name="BOMLNetOmniglotMetaFeatV2",
        outer_param_dict=OrderedDict(),
        dim_output=-1,
        model_param_dict=OrderedDict(),
        use_t=False,
        use_warp=False,
        reuse=False,
        outer_method="Reverse",
    ):
        """
        :param _input: original input
        :param dim_output: dimension of output
        :param name: scope of meta-learner
        :param outer_param_dict: dictionary of outer parameters
        :param model_param_dict:dictonary of model parameters for specific algorithms such t-layer or warp-layer
        :param use_t: Boolean, whether to use t-layer for neural network construction
        :param use_warp: Boolean, whether to use warp-layer for neural network construction
        :param outer_method: the name of outer method
        :param reuse: Boolean, whether to reuse the parameters
        """
        self.var_coll = boml.extension.METAPARAMETERS_COLLECTIONS
        self.outer_method = outer_method
        self.dim_output = dim_output
        self.use_t = use_t
        self.use_warp = use_warp
        super().__init__(
            _input=_input,
            outer_param_dict=outer_param_dict,
            model_param_dict=model_param_dict,
            name=name,
            reuse=reuse,
        )

        self.betas = self.filter_vars("beta")
        self.moving_means = self.filter_vars("moving_mean")
        self.moving_variances = self.filter_vars("moving_variance")

        if not reuse:
            boml.extension.remove_from_collection(
                boml.extension.GraphKeys.MODEL_VARIABLES,
                *self.moving_means,
                *self.moving_variances
            )

        boml.extension.remove_from_collection(
            boml.extension.GraphKeys.METAPARAMETERS,
            *self.moving_means,
            *self.moving_variances
        )
        print(name, "MODEL CREATED")

    def _forward(self):
        """
        _forward() uses defined convolutional neural networks with initial input
        :return:
        """
        def residual_block(x, n_filters):
            skip_c = tcl.conv2d(x, n_filters, 1, activation_fn=None)

            def conv_block(xx):
                out = tcl.conv2d(
                    xx,
                    n_filters,
                    3,
                    activation_fn=None,
                    normalizer_fn=tcl.batch_norm,
                    variables_collections=self.var_coll,
                )
                return network_utils.leaky_relu(out, 0.1)

            out = x
            for _ in range(3):
                out = conv_block(out)

            add = tf.add(skip_c, out)

            return tf.nn.max_pool(add, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        self + residual_block(self.out, 64)
        self + residual_block(self.out, 96)
        self + tcl.conv2d(self.out, 2048, 1, variables_collections=self.var_coll)
        self + tf.nn.avg_pool(self.out, [1, 6, 6, 1], [1, 6, 6, 1], "VALID")
        self + tcl.conv2d(self.out, 512, 1, variables_collections=self.var_coll)
        self + tf.reshape(self.out, (-1, 512))

    def re_forward(self, new_input=None, task_parameter=OrderedDict()):
        """
        reuses defined convolutional networks with new input and update the output results
        :param new_input: new input with same shape as the old one
        :param task_parameter: the dictionary of task-specific
        :return: updated instance of BOMLNet
        """
        return BOMLNetOmniglotMetaFeatV2(
            new_input if new_input is not None else self.layers[0],
            model_param_dict=self.model_param_dict,
            name=self.name,
            dim_output=self.dim_output,
            outer_param_dict=self.outer_param_dict,
            reuse=tf.AUTO_REUSE,
            use_t=self.use_t,
            outer_method=self.outer_method,
        )
