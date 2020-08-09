from collections import OrderedDict
from functools import reduce

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python import layers

from boml import extension
from boml.networks.BOMLNet import BOMLNet


class BOMLNetMetaReprV1(BOMLNet):

    def __init__(self, _input, name='BMLNetC4LMetaRepr', outer_param_dict=OrderedDict(),
                 model_param_dict=OrderedDict(), task_parameter=None,
                 use_T=False, use_Warp=False, outer_method='Reverse',
                 activation=tf.nn.relu, var_collections=extension.METAPARAMETERS_COLLECTIONS,
                 conv_initializer=tf.contrib.layers.xavier_initializer_conv2d(tf.float32),
                 output_weight_initializer=tf.contrib.layers.xavier_initializer(tf.float32), norm=True,
                 data_type=tf.float32, channels=1, dim_hidden=[64, 64, 64, 64], kernel=3,max_pool=False,
                 deterministic_initialization=False, reuse=False):

        self.kernel = kernel
        self.channels = channels
        self.dim_hidden = dim_hidden
        self.datatype = data_type
        self.batch_norm = norm
        self.max_pool = max_pool
        self.stride = 2
        self.no_stride = 1
        self.activation = activation
        self.bias_initializer = tf.zeros_initializer(tf.float32)
        self.conv_initializer = conv_initializer
        self.output_weight_initializer = output_weight_initializer
        self.use_T = use_T
        self.use_Warp=use_Warp
        self.outer_method = outer_method
        self.flatten = False if self.outer_method == 'Implicit' else True

        super(BOMLNetMetaReprV1, self).__init__(_input=_input, outer_param_dict=outer_param_dict,
                                                 var_collections=var_collections, name=name, model_param_dict=model_param_dict, task_parameter=task_parameter,
                                                 deterministic_initialization=deterministic_initialization, reuse=reuse)

        self.betas = self.filter_vars('beta')

        self.moving_means = self.filter_vars('moving_mean')
        self.moving_variances = self.filter_vars('moving_variance')

        if not reuse:
            extension.remove_from_collection(extension.GraphKeys.MODEL_VARIABLES, *self.moving_means)
            extension.remove_from_collection(extension.GraphKeys.MODEL_VARIABLES, *self.moving_variances)
            print(name, 'MODEL CREATED')
        extension.remove_from_collection(extension.GraphKeys.METAPARAMETERS, *self.moving_means)
        extension.remove_from_collection(extension.GraphKeys.METAPARAMETERS, *self.moving_variances)

    def create_outer_parameters(self):

        for i in range(len(self.dim_hidden)):
            self.outer_param_dict['conv' + str(i)] = self.get_conv_weight(layer=i, initializer=self.conv_initializer)
            self.outer_param_dict['bias' + str(i)] = self.get_bias_weight(layer=i, initializer=self.bias_initializer)

        # hyper parameters of transformation layer
        if self.use_T:
            for i in range(len(self.dim_hidden)):
                self.outer_param_dict['conv' + str(i) + '_z'] = self.get_identity(self.dim_hidden[0],
                                                                                  name='conv' + str(i) + '_z',
                                                                                  conv=True)

        [tf.add_to_collections(extension.GraphKeys.METAPARAMETERS, hyper) for hyper in self.outer_param_dict.values()]

        return self.outer_param_dict

    def create_model_parameters(self, var_collections=extension.GraphKeys.METAPARAMETERS):
        if self.use_T:
            # hyper parameters of transformation layer
            for i in range(len(self.dim_hidden)):
                self.model_param_dict['conv' + str(i) + '_z'] = self.get_identity(self.dim_hidden[0],
                                                                                  name='conv' + str(i) + '_z', conv=True)
            self.model_param_dict['w' + str(len(self.dim_hidden)) + '_z'] = self.get_identity(self.dims[-1],
                                                                                              name='w' + str(len(self.dim_hidden)) + '_z', conv=False)
        elif self.use_Warp:
            for i in range(len(self.dim_hidden)):
                self.model_param_dict['conv' + str(i)+'_z'] = self.get_warp_weight(layer=i,
                                                                              initializer=self.conv_initializer)
                self.model_param_dict['bias' + str(i)+'_z'] = self.get_warp_bias(layer=i,
                                                                              initializer=self.bias_initializer)
        [tf.add_to_collections(var_collections, model_param) for model_param in self.model_param_dict.values()]

        return self.model_param_dict

    def conv_block(self, cweight, bweight):
        """ Perform, conv, batch norm, nonlinearity, and max pool """
        if self.max_pool:
            conv_out = tf.add(tf.nn.conv2d(self.out, cweight, self.no_stride, 'SAME'), bweight)
        else:
            conv_out = tf.add(tf.nn.conv2d(self.out, cweight, self.stride, 'SAME'), bweight)
        if self.batch_norm:
            batch_out = layers.batch_norm(inputs=conv_out, activation_fn=self.activation,
                                          variables_collections=self.var_collections)
        else:
            batch_out = self.activation(conv_out)
        if self.max_pool:
            final_out = tf.nn.max_pool(batch_out, self.stride, self.stride, 'VALID')
            return final_out
        else:
            return batch_out

    def conv_block_t(self, cweight, bweight, zweight):
        """ Perform, conv, batch norm, nonlinearity, and max pool """
        if self.max_pool:
            conv_out = tf.add(tf.nn.conv2d(self.out, cweight, self.no_stride, 'SAME'), bweight)
        else:
            conv_out = tf.add(tf.nn.conv2d(self.out, cweight, self.stride, 'SAME'), bweight)
        conv_output = tf.nn.conv2d(conv_out, zweight, self.no_stride, 'SAME')

        if self.batch_norm:
            batch_out = layers.batch_norm(inputs=conv_output, activation_fn=self.activation,
                                          variables_collections=self.var_collections)
        else:
            batch_out = self.activation(conv_output)
        if self.max_pool:
            final_out = tf.nn.max_pool(batch_out, self.stride, self.stride, 'VALID')
            return final_out
        else:
            return batch_out

    def get_identity(self, dim, name, conv=True):
        return tf.get_variable(initializer=tf.eye(dim, batch_shape=[1, 1]), name=name,dtype=self.datatype) \
            if conv else self.get_identity(initializer=tf.eye(dim), name=name, dtype=self.datatype)

    def get_conv_weight(self, layer, initializer):
        if layer == 0:
            return tf.get_variable(name='conv' + str(layer),
                                                shape=[self.kernel, self.kernel, self.channels, self.dim_hidden[0]],
                                                initializer=initializer, dtype=self.datatype)
        else:
            return tf.get_variable(name='conv' + str(layer), shape=
                                                [self.kernel, self.kernel, self.dim_hidden[layer - 1], self.dim_hidden[layer]],
                                                initializer=initializer, dtype=self.datatype)

    def get_bias_weight(self, layer, initializer):
        return tf.get_variable(name='bias' + str(layer), shape=
                                            [self.dim_hidden[layer]], initializer=initializer)

    def _forward(self):
        '''
        for i in range(4):
            self.conv_layer(filters=self.dim_hidden[i],stride=self.stride, max_pool=self.max_pool)
        flattened_shape = reduce(lambda a, v: a * v, self.layers[-1].get_shape().as_list()[1:])
        self + tf.reshape(self.out, shape=(-1, flattened_shape), name='representation')
        '''

        for i in range(len(self.dim_hidden)):
            if self.use_T:
                self + self.conv_block_t(self.outer_param_dict['conv' + str(i)], self.outer_param_dict['bias' + str(i)],
                                         self.model_param_dict['conv' + str(i) + '_z'])
            elif self.use_Warp:
                self + self.conv_block_Warp(self.outer_param_dict['conv' + str(i)], self.outer_param_dict['bias' + str(i)],
                                         self.model_param_dict['conv' + str(i) + '_z'], self.model_param_dict['bias'+str(i)+'_z'])
            else:
                self + self.conv_block(self.outer_param_dict['conv' + str(i)], self.outer_param_dict['bias' + str(i)])
        if self.flatten:
            flattened_shape = reduce(lambda a, v: a * v, self.layers[-1].get_shape().as_list()[1:])
            self + tf.reshape(self.out, shape=(-1, flattened_shape), name='representation')
        else:
            if self.max_pool:
                self + tf.reshape(self.out, [-1, np.prod([int(dim) for dim in self.out.get_shape()[1:]])])
            else:
                self + tf.reduce_mean(self.out, [1, 2])

    def re_forward(self, new_input):
        return BOMLNetMetaReprV1(_input=new_input if new_input is not None else self.layers[0], name=self.name,
                                  activation=self.activation, outer_param_dict=self.outer_param_dict, model_param_dict=self.model_param_dict,
                                  task_parameter=self.task_parameter, use_Warp=self.use_Warp, use_T=self.use_T,
                                  var_collections=self.var_collections, dim_hidden=self.dim_hidden,
                                  output_weight_initializer=self.output_weight_initializer, max_pool=self.max_pool,
                                  deterministic_initialization=self.deterministic_initialization, reuse=True,
                                  outer_method=self.outer_method)


def BOMLNetOmniglotMetaReprV1(_input, outer_param_dict=OrderedDict(),model_param_dict=OrderedDict(),
                              batch_norm=True, name='BMLNetC4LOmniglot', use_T=False,
                              use_Warp=False, outer_method='Reverse',**model_args):

    return BOMLNetMetaReprV1(_input=_input, name=name, model_param_dict=model_param_dict,
                              outer_param_dict=outer_param_dict, norm=batch_norm, use_T=use_T, use_Warp=use_Warp,
                              outer_method=outer_method, **model_args)


def BOMLNetMiniMetaReprV1(_input, outer_param_dict=OrderedDict(),model_param_dict=OrderedDict(),
                          batch_norm=True, name='BMLNetC4LMini', use_T=False,
                          use_Warp=False, outer_method='Reverse',**model_args):
    return BOMLNetMetaReprV1(_input=_input, name=name, use_T=use_T, use_Warp=use_Warp,
                              outer_param_dict=outer_param_dict, model_param_dict=model_param_dict, norm=batch_norm, channels=3,
                              dim_hidden=[32, 32, 32, 32], max_pool=True, outer_method=outer_method, **model_args)
