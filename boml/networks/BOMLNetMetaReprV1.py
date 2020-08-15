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
