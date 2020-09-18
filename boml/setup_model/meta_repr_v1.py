from collections import OrderedDict
from functools import reduce

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python import layers

from boml import extension
from boml.setup_model import network_utils
from boml.setup_model.network import BOMLNet


class BOMLNetMetaReprV1(BOMLNet):
    def __init__(
        self,
        _input,
        name="BMLNetC4LMetaRepr",
        outer_param_dict=OrderedDict(),
        model_param_dict=OrderedDict(),
        task_parameter=None,
        use_T=False,
        use_Warp=False,
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
        self.use_T = use_T
        self.use_Warp = use_Warp
        self.outer_method = outer_method
        self.flatten = False if self.outer_method == "Implicit" else True

        super(BOMLNetMetaReprV1, self).__init__(
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

        for i in range(len(self.dim_hidden)):
            self.outer_param_dict["conv" + str(i)] = network_utils.get_conv_weight(
                self, layer=i, initializer=self.conv_initializer
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
        if self.use_T:
            # hyper parameters of transformation layer
            for i in range(len(self.dim_hidden)):
                self.model_param_dict[
                    "conv" + str(i) + "_z"
                ] = network_utils.get_identity(
                    self.dim_hidden[0], name="conv" + str(i) + "_z", conv=True
                )
        elif self.use_Warp:
            for i in range(len(self.dim_hidden)):
                self.model_param_dict[
                    "conv" + str(i) + "_z"
                ] = network_utils.get_warp_weight(
                    self, layer=i, initializer=self.conv_initializer
                )
                self.model_param_dict[
                    "bias" + str(i) + "_z"
                ] = network_utils.get_warp_bias(
                    self, layer=i, initializer=self.bias_initializer
                )
        [
            tf.add_to_collections(var_collections, model_param)
            for model_param in self.model_param_dict.values()
        ]

        return self.model_param_dict

    def _forward(self):
        """
        for i in range(4):
            self.conv_layer(filters=self.dim_hidden[i],stride=self.stride, max_pool=self.max_pool)
        flattened_shape = reduce(lambda a, v: a * v, self.layers[-1].get_shape().as_list()[1:])
        self + tf.reshape(self.out, shape=(-1, flattened_shape), name='representation')
        """

        for i in range(len(self.dim_hidden)):
            if self.use_T:
                self + network_utils.conv_block_t(
                    self,
                    self.outer_param_dict["conv" + str(i)],
                    self.outer_param_dict["bias" + str(i)],
                    self.model_param_dict["conv" + str(i) + "_z"],
                )
            elif self.use_Warp:
                self + network_utils.conv_block_warp(
                    self,
                    self.outer_param_dict["conv" + str(i)],
                    self.outer_param_dict["bias" + str(i)],
                    self.model_param_dict["conv" + str(i) + "_z"],
                    self.model_param_dict["bias" + str(i) + "_z"],
                )
            else:
                self + network_utils.conv_block(
                    self, self.outer_param_dict["conv" + str(i)],
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
        return BOMLNetMetaReprV1(
            _input=new_input if new_input is not None else self.layers[0],
            name=self.name,
            activation=self.activation,
            outer_param_dict=self.outer_param_dict,
            model_param_dict=self.model_param_dict,
            dim_output=self.dim_output,
            task_parameter=self.task_parameter,
            use_Warp=self.use_Warp,
            use_T=self.use_T,
            var_collections=self.var_collections,
            dim_hidden=self.dim_hidden,
            output_weight_initializer=self.output_weight_initializer,
            max_pool=self.max_pool,
            reuse=True,
            outer_method=self.outer_method,
        )


def BOMLNetOmniglotMetaReprV1(
    _input,
    outer_param_dict=OrderedDict(),
    model_param_dict=OrderedDict(),
    batch_norm=layers.batch_norm,
    name="BMLNetC4LOmniglot",
    use_T=False,
    dim_output=-1,
    use_Warp=False,
    outer_method="Reverse",
    **model_args
):

    return BOMLNetMetaReprV1(
        _input=_input,
        name=name,
        model_param_dict=model_param_dict,
        dim_output=dim_output,
        outer_param_dict=outer_param_dict,
        norm=batch_norm,
        use_T=use_T,
        use_Warp=use_Warp,
        outer_method=outer_method,
        **model_args
    )


def BOMLNetMiniMetaReprV1(
    _input,
    outer_param_dict=OrderedDict(),
    model_param_dict=OrderedDict(),
    dim_output=-1,
    batch_norm=layers.batch_norm,
    name="BOMLNetC4LMini",
    use_T=False,
    use_Warp=False,
    outer_method="Reverse",
    **model_args
):
    return BOMLNetMetaReprV1(
        _input=_input,
        name=name,
        use_T=use_T,
        use_Warp=use_Warp,
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
