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
The base class in setup_model to encapsulate Residual Block for meta-initialization-based methods.
"""
from boml.extension import remove_from_collection
from boml.setup_model import network_utils
from boml.setup_model.network import *
from boml.setup_model.network_utils import as_tuple_or_list


class BOMLNetMiniMetaInitV2(BOMLNet):
    def __init__(
        self,
        _input,
        dim_output,
        name="BOMLNetMiniMetaInitV2",
        outer_param_dict=OrderedDict(),
        model_param_dict=OrderedDict(),
        task_parameter=OrderedDict(),
        activation=tf.nn.relu,
        var_collections=tf.GraphKeys.MODEL_VARIABLES,
        outer_method="Simple",
        conv_initializer=tf.contrib.layers.xavier_initializer_conv2d(tf.float32),
        output_weight_initializer=tf.contrib.layers.xavier_initializer(tf.float32),
        batch_norm=True,
        data_type=tf.float32,
        channels=3,
        dim_resnet=[64, 96, 128, 256],
        kernel=3,
        reuse=False,
        use_t=False,
        use_warp=False,
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
        :param data_type: default to be tf.float32
        :param channels: number of channels
        :param dim_resnet: lists to specify the dimension of residual block
        :param kernel: size of the kernel
        :param reuse: Boolean, whether to reuse the parameters
        """
        self.task_parameter = task_parameter
        self.dims = as_tuple_or_list(dim_output)
        self.kernel = kernel
        self.dim_resnet = dim_resnet
        self.channels = channels
        self.datatype = data_type
        self.activation = activation
        self.batch_norm = batch_norm
        self.stride = [1, 2, 2, 1]
        self.no_stride = [1, 1, 1, 1]
        self.conv_initializer = conv_initializer
        self.output_weight_initializer = output_weight_initializer
        self.use_t = use_t
        self.outer_method = outer_method
        self.use_warp = use_warp

        super().__init__(
            _input=_input,
            outer_param_dict=outer_param_dict,
            model_param_dict=model_param_dict,
            var_collections=var_collections,
            name=name,
            reuse=reuse,
        )
        # variables from batch normalization
        self.betas = self.filter_vars("beta")
        # moving mean and variance (these variables should be used at inference time... so must save them)
        self.moving_means = self.filter_vars("moving_mean")
        self.moving_variances = self.filter_vars("moving_variance")

        if not reuse:  # these calls might print a warning... it's not a problem..
            remove_from_collection(GraphKeys.MODEL_VARIABLES, *self.betas)
            remove_from_collection(GraphKeys.MODEL_VARIABLES, *self.moving_means)
            remove_from_collection(GraphKeys.MODEL_VARIABLES, *self.moving_variances)
        print(name, "MiniImagenet_MODEL CREATED")

    def create_outer_parameters(self, var_collections=GraphKeys.METAPARAMETERS):
        """
        :param var_collections: name of collections to store the created variables.
        :return: dictionary to index the created variables.
        """
        for i in range(len(self.dim_resnet)):
            self.outer_param_dict["res" + str(i + 1) + "id"] = tf.get_variable(
                name="res" + str(i + 1) + "id",
                shape=[
                    1,
                    1,
                    self.channels if i == 0 else self.dim_resnet[i - 1],
                    self.dim_resnet[i],
                ],
                dtype=self.datatype,
                initializer=self.conv_initializer,
            )
            for j in range(3):
                if i == 0:
                    self.outer_param_dict[
                        "res" + str(i + 1) + "conv_w" + str(j + 1)
                    ] = tf.get_variable(
                        name="res" + str(i + 1) + "conv_w" + str(j + 1),
                        shape=[
                            3,
                            3,
                            (self.channels if j == 0 else self.dim_resnet[i]),
                            self.dim_resnet[i],
                        ],
                        dtype=self.datatype,
                        initializer=self.conv_initializer,
                    )
                else:
                    self.outer_param_dict[
                        "res" + str(i + 1) + "conv_w" + str(j + 1)
                    ] = tf.get_variable(
                        name="res" + str(i + 1) + "conv_w" + str(j + 1),
                        shape=[
                            3,
                            3,
                            (self.dim_resnet[i - 1] if j == 0 else self.dim_resnet[i]),
                            self.dim_resnet[i],
                        ],
                        dtype=self.datatype,
                        initializer=self.conv_initializer,
                    )

        self.outer_param_dict["res_conv1"] = tf.get_variable(
            name="conv1",
            shape=[1, 1, self.dim_resnet[-1], 2048],
            initializer=self.conv_initializer,
            dtype=self.datatype,
        )
        self.outer_param_dict["res_bias1"] = tf.Variable(
            tf.zeros([2048]), name="res_bias1"
        )
        self.outer_param_dict["res_conv2"] = tf.get_variable(
            name="res_conv2",
            dtype=self.datatype,
            shape=[1, 1, 2048, 512],
            initializer=self.conv_initializer,
        )
        self.outer_param_dict["res_bias2"] = tf.Variable(
            tf.zeros([512]), name="res_bias2"
        )
        self.outer_param_dict["fully_connected_weights"] = tf.get_variable(
            name="fully_connected_weights",
            dtype=self.datatype,
            shape=[512, self.dims[-1]],
            initializer=self.output_weight_initializer,
        )
        self.outer_param_dict["fully_connected_bias"] = tf.Variable(
            name="fully_connected_bias", initial_value=tf.zeros([self.dims[-1]])
        )
        [
            tf.add_to_collection(var_collections, hyper)
            for hyper in self.outer_param_dict.values()
        ]
        return self.outer_param_dict

    def _forward(self):
        """
        _forward() uses defined convolutional neural networks with initial input
        :return:
        """
        if len(self.task_parameter) == 0:
            self.task_parameter = self.create_initial_parameter(
                primary_outerparameter=self.outer_param_dict
            )

        def residual_block(x, i):
            skipc = tf.nn.conv2d(
                input=x,
                filter=self.task_parameter["res" + str(i + 1) + "id"],
                strides=self.no_stride,
                padding="SAME",
            )

            def conv_block(xx, i, j):
                out = tf.nn.conv2d(
                    xx,
                    self.task_parameter["res" + str(i + 1) + "conv_w" + str(j + 1)],
                    self.no_stride,
                    "SAME",
                )
                out = tf.contrib.layers.batch_norm(out, activation_fn=None)
                return network_utils.leaky_relu(out, 0.1)

            out = x
            for j in range(3):
                out = conv_block(out, i, j)

            add = tf.add(skipc, out)
            return tf.nn.max_pool(add, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        for i in range(len(self.dim_resnet)):
            self + residual_block(self.out, i)
        self + tf.add(
            tf.nn.conv2d(
                self.out, self.task_parameter["res_conv1"], self.no_stride, "SAME"
            ),
            self.task_parameter["res_bias1"],
        )
        self + tf.nn.avg_pool(self.out, [1, 6, 6, 1], [1, 6, 6, 1], "VALID")
        self + tf.add(
            tf.nn.conv2d(
                self.out, self.task_parameter["res_conv2"], self.no_stride, "SAME"
            ),
            self.task_parameter["res_bias2"],
        )
        self + tf.reshape(self.out, (-1, 512))
        self + tf.add(
            tf.matmul(self.out, self.task_parameter["fully_connected_weights"]),
            self.task_parameter["fully_connected_bias"],
        )

    def re_forward(self, new_input=None, task_parameter=OrderedDict()):
        return BOMLNetMiniMetaInitV2(
            _input=new_input if (new_input is not None) else self.layers[0],
            dim_output=self.dims[-1],
            name=self.name,
            activation=self.activation,
            outer_method=self.outer_method,
            outer_param_dict=self.outer_param_dict,
            model_param_dict=self.model_param_dict,
            task_parameter=self.task_parameter
            if len(task_parameter.keys()) == 0
            else task_parameter,
            var_collections=self.var_collections,
            output_weight_initializer=self.output_weight_initializer,
            reuse=True,
            use_t=self.use_t,
            use_warp=self.use_warp,
        )


class BOMLNetOmniglotMetaInitV2(BOMLNet):
    def __init__(
        self,
        _input,
        dim_output,
        name="Omniglot_ResNet",
        outer_param_dict=OrderedDict(),
        model_param_dict=OrderedDict(),
        task_parameter=OrderedDict(),
        activation=tf.nn.relu,
        var_collections=tf.GraphKeys.MODEL_VARIABLES,
        conv_initializer=tf.contrib.layers.xavier_initializer_conv2d(tf.float32),
        outer_method="Simple",
        output_weight_initializer=tf.zeros_initializer,
        batch_norm=True,
        data_type=tf.float32,
        channels=1,
        dim_resnet=[64, 96],
        dim_hidden=64,
        kernel=3,
        max_pool=False,
        reuse=False,
        use_t=False,
        use_warp=False,
    ):
        self.task_parameter = task_parameter
        self.kernel = kernel
        self.channels = channels
        self.dim_resnet = dim_resnet
        self.dims = as_tuple_or_list(dim_output)
        self.dim_hidden = dim_hidden
        self.datatype = data_type
        self.batch_norm = batch_norm
        self.max_pool = max_pool
        self.stride = [1, 2, 2, 1]
        self.no_stride = [1, 1, 1, 1]
        self.activation = activation
        self.conv_initializer = conv_initializer
        self.output_weight_initializer = output_weight_initializer
        self.use_t = use_t
        self.outer_method = (outer_method,)
        self.use_warp = use_warp
        super().__init__(
            _input=_input,
            outer_param_dict=outer_param_dict,
            model_param_dict=model_param_dict,
            var_collections=var_collections,
            name=name,
            reuse=reuse,
        )

        # variables from batch normalization
        self.betas = self.filter_vars("beta")
        # moving mean and variance (these variables should be used at inference time... so must save them)
        self.moving_means = self.filter_vars("moving_mean")
        self.moving_variances = self.filter_vars("moving_variance")

        if not reuse:  # these calls might print a warning... it's not a problem..
            remove_from_collection(GraphKeys.MODEL_VARIABLES, *self.betas)
            remove_from_collection(GraphKeys.MODEL_VARIABLES, *self.moving_means)
            remove_from_collection(GraphKeys.MODEL_VARIABLES, *self.moving_variances)
            print(name, "MODEL CREATED")

    def create_outer_parameters(self, var_collections=GraphKeys.METAPARAMETERS):
        """
        :param var_collections: name of collections to store the created variables.
        :return: dictionary to index the created variables.
        """
        for i in range(len(self.dim_resnet)):
            self.outer_param_dict["res" + str(i + 1) + "id"] = tf.get_variable(
                name="res" + str(i + 1) + "id",
                shape=[
                    1,
                    1,
                    self.channels if i == 0 else self.dim_resnet[i - 1],
                    self.dim_resnet[i],
                ],
                dtype=self.datatype,
                initializer=self.conv_initializer,
            )
            for j in range(3):
                if i == 0:
                    self.outer_param_dict[
                        "res" + str(i + 1) + "conv_w" + str(j + 1)
                    ] = tf.get_variable(
                        name="res" + str(i + 1) + "conv_w" + str(j + 1),
                        shape=[
                            3,
                            3,
                            (self.channels if j == 0 else self.dim_resnet[i]),
                            self.dim_resnet[i],
                        ],
                        dtype=self.datatype,
                        initializer=self.conv_initializer,
                    )
                else:
                    self.outer_param_dict[
                        "res" + str(i + 1) + "conv_w" + str(j + 1)
                    ] = tf.get_variable(
                        name="res" + str(i + 1) + "conv_w" + str(j + 1),
                        shape=[
                            3,
                            3,
                            (self.dim_resnet[i - 1] if j == 0 else self.dim_resnet[i]),
                            self.dim_resnet[i],
                        ],
                        dtype=self.datatype,
                        initializer=self.conv_initializer,
                    )
                self.outer_param_dict[
                    "res" + str(i + 1) + "conv_bias" + str(j + 1)
                ] = tf.get_variable(
                    name="res" + str(i + 1) + "conv_bias" + str(j + 1),
                    shape=[self.dim_resnet[i]],
                    initializer=tf.zeros_initializer(tf.float32),
                    dtype=self.datatype,
                )
        self.outer_param_dict["res_conv1"] = tf.get_variable(
            name="conv1",
            shape=[1, 1, self.dim_resnet[-1], 2048],
            initializer=self.conv_initializer,
            dtype=self.datatype,
        )
        self.outer_param_dict["res_bias1"] = tf.Variable(
            tf.zeros([2048]), name="res_bias1"
        )
        self.outer_param_dict["res_conv2"] = tf.get_variable(
            name="res_conv2",
            dtype=self.datatype,
            shape=[1, 1, 2048, 512],
            initializer=self.conv_initializer,
        )
        self.outer_param_dict["res_bias2"] = tf.Variable(
            tf.zeros([512]), name="res_bias2"
        )
        self.outer_param_dict["fully_connected_weights"] = tf.get_variable(
            name="fully_connected_weights",
            dtype=self.datatype,
            shape=[512, self.dims[-1]],
            initializer=self.output_weight_initializer,
        )
        self.outer_param_dict["fully_connected_bias"] = tf.Variable(
            name="fully_connected_bias", initial_value=tf.zeros([self.dims[-1]])
        )
        [
            tf.add_to_collection(var_collections, hyper)
            for hyper in self.outer_param_dict.values()
        ]
        return self.outer_param_dict

    def _forward(self):
        """
        _forward() uses defined convolutional neural networks with initial input
        :return:
        """
        if len(self.task_parameter) == 0:
            self.task_parameter = self.create_initial_parameter(
                primary_outerparameter=self.outer_param_dict
            )

        for i in range(len(self.dim_resnet)):
            self + self.residual_block(self.out, i)

        self + tf.add(
            tf.nn.conv2d(
                self.out, self.task_parameter["res_conv1"], self.no_stride, "SAME"
            ),
            self.task_parameter["res_bias1"],
        )
        self + tf.nn.avg_pool(self.out, [1, 6, 6, 1], [1, 6, 6, 1], "VALID")
        self + tf.add(
            tf.nn.conv2d(
                self.out, self.task_parameter["res_conv2"], self.no_stride, "SAME"
            ),
            self.task_parameter["res_bias2"],
        )
        self + tf.reshape(self.out, (-1, 512))
        self + tf.add(
            tf.matmul(self.out, self.task_parameter["fully_connected_weights"]),
            self.task_parameter["fully_connected_bias"],
        )

    def residual_block(self, x, i):
        """
        :param x: input for the i-th block
        :param i: i-th block
        :return: the output of the j-th conv block in the i-th residual block
        """
        skipc = tf.nn.conv2d(
            x, self.task_parameter["res" + str(i + 1) + "id"], self.no_stride, "SAME"
        )
        out = x
        for j in range(3):
            out = self.conv_block(out, i, j)

        add = tf.add(skipc, out)
        return tf.nn.max_pool(add, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

    def conv_block(self, xx, i, j):
        """
        :param xx:  input for the i-th block
        :param i: i-th block
        :param j: the j-th convolution block
        :return: the ouput of the i-th conv block in the
        """
        out = tf.add(
            tf.nn.conv2d(
                xx,
                self.task_parameter["res" + str(i + 1) + "conv_w" + str(j + 1)],
                self.no_stride,
                "SAME",
            ),
            self.task_parameter["res" + str(i + 1) + "conv_bias" + str(j + 1)],
        )
        out = tf.contrib.layers.batch_norm(
            out,
            activation_fn=None,
            variables_collections=self.var_collections,
            scope="scope" + str(i + 1) + str(j + 1),
            reuse=self.reuse,
        )
        return network_utils.leaky_relu(out, 0.1)

    def re_forward(self, new_input=None, task_parameter=OrderedDict()):
        """
        reuses defined convolutional networks with new input and update the output results
        :param new_input: new input with same shape as the old one
        :param task_parameter: the dictionary of task-specific
        :return: updated instance of BOMLNet
        """
        return BOMLNetOmniglotMetaInitV2(
            _input=new_input if new_input is not None else self.layers[0],
            dim_output=self.dims[-1],
            name=self.name,
            activation=self.activation,
            outer_param_dict=self.outer_param_dict,
            model_param_dict=self.model_param_dict,
            task_parameter=task_parameter
            if len(task_parameter) == 0
            else self.task_parameter,
            var_collections=self.var_collections,
            outer_method=self.outer_method,
            output_weight_initializer=self.output_weight_initializer,
            reuse=True,
            use_t=self.use_t,
            use_warp=self.use_warp,
        )
