from collections import OrderedDict

import tensorflow as tf
from tensorflow.contrib import layers as tcl

import boml.extension
from boml.setup_model import network_utils
from boml.setup_model.BOMLNet import BOMLNet


class BOMLNetMiniMetaReprV2(BOMLNet):

    def __init__(self, _input, name='BMLNetMetaReprMini', deterministic_initialization=False, outer_param_dict=OrderedDict(), model_param_dict=OrderedDict(),
                 use_T=False, reuse=False, outer_method='Reverse'):
        self.var_coll = boml.extension.METAPARAMETERS_COLLECTIONS
        super().__init__(_input=_input, name=name, outer_param_dict=outer_param_dict, model_param_dict=model_param_dict,
                         deterministic_initialization=deterministic_initialization, reuse=reuse)
        self.outer_method = outer_method
        self.use_T = use_T
        self.betas = self.filter_vars('beta')
        self.moving_means = self.filter_vars('moving_mean')
        self.moving_variances = self.filter_vars('moving_variance')

        if not reuse:
            boml.extension.remove_from_collection(boml.extension.GraphKeys.MODEL_VARIABLES, *self.moving_means,
                                                  *self.moving_variances)

        boml.extension.remove_from_collection(boml.extension.GraphKeys.METAPARAMETERS, *self.moving_means,
                                              *self.moving_variances)
        print(name, 'MODEL CREATED')

    def _forward(self):

        def residual_block(x, n_filters):
            skip_c = tcl.conv2d(x, n_filters, 1, activation_fn=None)

            def conv_block(xx):
                out = tcl.conv2d(xx, n_filters, 3, activation_fn=None, normalizer_fn=tcl.batch_norm,
                                 variables_collections=self.var_coll)
                return network_utils.leaky_relu(out, 0.1)

            out = x
            for _ in range(3):
                out = conv_block(out)

            add = tf.add(skip_c, out)

            return tf.nn.max_pool(add, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        self + residual_block(self.out, 64)
        self + residual_block(self.out, 96)
        self + residual_block(self.out, 128)
        self + residual_block(self.out, 256)
        self + tcl.conv2d(self.out, 2048, 1, variables_collections=self.var_coll)
        self + tf.nn.avg_pool(self.out, [1, 6, 6, 1], [1, 6, 6, 1], 'VALID')
        self + tcl.conv2d(self.out, 512, 1, variables_collections=self.var_coll)
        self + tf.reshape(self.out, (-1, 512))

    def re_forward(self, new_input=None):
        return BOMLNetMiniMetaReprV2(_input=new_input if new_input is not None else self.layers[0], model_param_dict=self.model_param_dict,
                                      name=self.name, outer_param_dict=self.outer_param_dict, deterministic_initialization=self.deterministic_initialization,
                                      reuse=True, outer_method=self.outer_method, use_T=self.use_T)


class BOMLNetOmniglotMetaReprV2(BOMLNet):
    def __init__(self, _input, name='BMLNetMetaReprOmniglot', deterministic_initialization=False, outer_param_dict=OrderedDict(),
                 model_param_dict=OrderedDict(),use_T=False, reuse=False, outer_method='Reverse'):
        self.var_coll = boml.extension.METAPARAMETERS_COLLECTIONS
        super().__init__(_input=_input, outer_param_dict=outer_param_dict,model_param_dict=model_param_dict,
                         deterministic_initialization=deterministic_initialization, name=name,
                         reuse=reuse)
        self.outer_method = outer_method
        self.use_T = use_T
        self.betas = self.filter_vars('beta')
        self.moving_means = self.filter_vars('moving_mean')
        self.moving_variances = self.filter_vars('moving_variance')

        if not reuse:
            boml.extension.remove_from_collection(boml.extension.GraphKeys.MODEL_VARIABLES, *self.moving_means,
                                                  *self.moving_variances)

        boml.extension.remove_from_collection(boml.extension.GraphKeys.METAPARAMETERS, *self.moving_means,
                                              *self.moving_variances)
        print(name, 'MODEL CREATED')

    def _forward(self):

        def residual_block(x, n_filters):
            skip_c = tcl.conv2d(x, n_filters, 1, activation_fn=None)

            def conv_block(xx):
                out = tcl.conv2d(xx, n_filters, 3, activation_fn=None, normalizer_fn=tcl.batch_norm,
                                 variables_collections=self.var_coll)
                return network_utils.leaky_relu(out, 0.1)

            out = x
            for _ in range(3):
                out = conv_block(out)

            add = tf.add(skip_c, out)

            return tf.nn.max_pool(add, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        self + residual_block(self.out, 64)
        self + residual_block(self.out, 96)
        self + tcl.conv2d(self.out, 2048, 1, variables_collections=self.var_coll)
        self + tf.nn.avg_pool(self.out, [1, 6, 6, 1], [1, 6, 6, 1], 'VALID')
        self + tcl.conv2d(self.out, 512, 1, variables_collections=self.var_coll)
        self + tf.reshape(self.out, (-1, 512))

    def re_forward(self, new_input=None):
        return BOMLNetOmniglotMetaReprV2(new_input if new_input is not None else self.layers[0], model_param_dict=self.model_param_dict,
                                          name=self.name, outer_param_dict=self.outer_param_dict, deterministic_initialization=self.deterministic_initialization,
                                          reuse=True, use_T=self.use_T, outer_method=self.outer_method)


if __name__ == '__main__':
    inp = tf.placeholder(tf.float32, (None, 84, 84, 3))
    net = BOMLNetMiniMetaReprV2(inp)
    print(net.out)
    print(boml.extension.metaparameters())
