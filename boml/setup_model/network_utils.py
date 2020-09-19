"""
Contains some misc utility functions
"""

from functools import reduce

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers


def conv_block(boml_net, cweight, bweight=None):

    """ Perform, conv, batch norm, nonlinearity, and max pool
    :param boml_net: instance of BOMLNet
    :param cweight: parameter of convolutional filter
    :param bweight: bias for covolutional filter
    """
    if boml_net.max_pool:
        conv_out = tf.add(
            tf.nn.conv2d(boml_net.out, cweight, boml_net.no_stride, "SAME"), bweight
        )
    else:
        conv_out = tf.add(
            tf.nn.conv2d(boml_net.out, cweight, boml_net.stride, "SAME"), bweight
        )

    if boml_net.batch_norm is not None:
        batch_out = boml_net.batch_norm(
            inputs=conv_out,
            activation_fn=boml_net.activation,
            variables_collections=boml_net.var_collections,
        )
    else:
        batch_out = boml_net.activation(conv_out)
    if boml_net.max_pool:
        final_out = tf.nn.max_pool(batch_out, boml_net.stride, boml_net.stride, "VALID")
        return final_out
    else:
        return batch_out


def conv_block_t(boml_net, conv_weight, conv_bias, zweight):
    """ Perform, conv, batch norm, nonlinearity, and max pool
    :param boml_net: instance of BOMLNet
    :param convweight: parameter of convolutional filter
    :param conv_bias: bias for covolutional filter
    :param zweight: parameters of covolutional filter for t-layer"""
    if boml_net.max_pool:
        conv_out = tf.add(
            tf.nn.conv2d(boml_net.out, conv_weight, boml_net.no_stride, "SAME"),
            conv_bias,
        )
    else:
        conv_out = tf.add(
            tf.nn.conv2d(boml_net.out, conv_weight, boml_net.stride, "SAME"), conv_bias
        )
    conv_output = tf.nn.conv2d(conv_out, zweight, boml_net.no_stride, "SAME")

    if boml_net.batch_norm is not None:
        batch_out = layers.batch_norm(
            conv_output,
            activation_fn=boml_net.activation,
            variables_collections=boml_net.var_collections,
        )
    else:
        batch_out = boml_net.activation(conv_output)
    if boml_net.max_pool:
        final_out = tf.nn.max_pool(batch_out, boml_net.stride, boml_net.stride, "VALID")
        return final_out
    else:
        return batch_out


def conv_block_warp(boml_net, cweight, bweight, zweight, zbias):
    """ Perform, conv, batch norm, nonlinearity, and max pool
    :param boml_net: instance of BOMLNet
    :param convweight: parameter of convolutional filter
    :param conv_bias: bias for covolutional filter
    :param zweight: parameters of covolutional filter for warp-layer
    :param zbias: bias of covolutional filter for warp-layer"""
    if boml_net.max_pool:
        conv_out = tf.add(
            tf.nn.conv2d(boml_net.out, cweight, boml_net.no_stride, "SAME"), bweight
        )
    else:
        conv_out = tf.add(
            tf.nn.conv2d(boml_net.out, cweight, boml_net.stride, "SAME"), bweight
        )

    conv_output = tf.add(
        tf.nn.conv2d(conv_out, zweight, boml_net.no_stride, "SAME"), zbias
    )

    if boml_net.batch_norm is not None:
        batch_out = layers.batch_norm(
            conv_output,
            activation_fn=boml_net.activation,
            variables_collections=boml_net.var_collections,
        )
    else:
        batch_out = boml_net.activation(conv_output)
    if boml_net.max_pool:
        final_out = tf.nn.max_pool(batch_out, boml_net.stride, boml_net.stride, "VALID")
        return final_out
    else:
        return batch_out


def get_conv_weight(boml_net, i, initializer):
    """
    :param boml_net: instance of BOMLNet
    :param i: int32, the i-th layer to be defined
    :param initializer: function to initialize the weights of convolutional filter
    :return: created convolutional weights
    """
    if i == 0:
        return tf.get_variable(
            "conv" + str(i),
            [
                boml_net.kernel,
                boml_net.kernel,
                boml_net.channels,
                boml_net.dim_hidden[0],
            ],
            initializer=initializer,
            dtype=boml_net.datatype,
        )
    else:
        return tf.get_variable(
            "conv" + str(i),
            [
                boml_net.kernel,
                boml_net.kernel,
                boml_net.dim_hidden[i - 1],
                boml_net.dim_hidden[i],
            ],
            initializer=initializer,
            dtype=boml_net.datatype,
        )


def get_bias_weight(boml_net, i, initializer):
    """
    :param boml_net: instance of BOMLNet
    :param i: int32, i-th layer to be defined
    :param initializer: function to initialize the bias
    :return: created convolutional bias
    """
    return tf.get_variable(
        "bias" + str(i),
        [boml_net.dim_hidden[i]],
        initializer=initializer,
        dtype=boml_net.datatype,
    )


def get_warp_weight(boml_net, layer, initializer):
    """
    :param boml_net: instance of BOMLNet
    :param layer: int32, i-th layer to be defined
    :param initializer: function to initializer the weights of convolutional filter
    :return: created convolutional parameters of Warp-Layer
    """
    return tf.get_variable(
        "conv" + str(layer) + "_z",
        [
            boml_net.kernel,
            boml_net.kernel,
            boml_net.dim_hidden[layer - 1],
            boml_net.dim_hidden[layer],
        ],
        initializer=initializer,
        dtype=boml_net.datatype,
    )


def get_warp_bias(boml_net, layer, initializer):
    """
    :param boml_net: instance of BOMLNet
    :param layer: int32, i-th layer to be defined
    :param initializer: function to initializer the weights of convolutional filter
    :return: created bias of convolutional filter
    """
    return tf.get_variable(
        "bias" + str(layer) + "_z",
        [boml_net.dim_hidden[layer]],
        initializer=initializer,
        dtype=boml_net.datatype,
    )


def get_identity(dim, name, conv=True):
    """
    :param dim: dimension of the identity matrix
    :param name: name for the variable
    :param conv: BOOLEAN, define variables as matrix or vector
    :return: created parameters of identity for T-layer
    """
    return (
        tf.Variable(tf.eye(dim, batch_shape=[1, 1]), name=name)
        if conv
        else tf.Variable(tf.eye(dim), name=name)
    )


def as_list(obj):
    """
    Makes sure `obj` is a list or otherwise converts it to a list with a single element.

    :param obj:
    :return: A `list`
    """
    return obj if isinstance(obj, list) else [obj]


def maybe_call(obj, *args, **kwargs):
    """
    Calls obj with args and kwargs and return its result if obj is callable, otherwise returns obj.
    """
    if callable(obj):
        return obj(*args, **kwargs)
    return obj


def as_tuple_or_list(obj):
    """
    Make sure that `obj` is a tuple or a list and eventually converts it into a list with a single element

    :param obj:
    :return: A `tuple` or a `list`
    """
    return obj if isinstance(obj, (list, tuple)) else [obj]


def maybe_get(obj, i):
    """

    :param obj: object
    :param i: the index
    :return: the i-th item if the `obj` instantiates the __getitem__ function
    """
    return obj[i] if hasattr(obj, "__getitem__") else obj


def merge_dicts(*dicts):
    return reduce(lambda a, nd: {**a, **nd}, dicts, {})


def to_one_hot_enc(seq, dimension=None):
    da_max = dimension or int(np.max(seq)) + 1
    _tmp = np.zeros((len(seq), da_max))
    _tmp[range(len(_tmp)), np.array(seq, dtype=int)] = 1
    return _tmp


def flatten_list(lst):
    from itertools import chain

    return list(chain(*lst))


def filter_vars(var_name, scope):
    import tensorflow as tf

    return [
        v
        for v in tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=scope.name if hasattr(scope, "name") else scope,
        )
        if v.name.endswith("%s:0" % var_name)
    ]


def name_from_vars(var_dict, *vars_):
    """
    Unfortunately this method doesn't return a very specific name....It gets a little messy

    :param var_dict:
    :param vars_:
    :return:
    """
    new_k_v = {}
    for v in vars_:
        for k, vv in var_dict.items():
            if v == vv:
                new_k_v[k] = v
    return name_from_dict(new_k_v)


def name_from_dict(_dict, *exclude_names):
    string_dict = {str(k): str(v) for k, v in _dict.items() if k not in exclude_names}
    return _tf_string_replace("_".join(flatten_list(list(sorted(string_dict.items())))))


def _tf_string_replace(_str):
    """
    Replace chars that are not accepted by tensorflow namings (eg. variable_scope)

    :param _str:
    :return:
    """
    return (
        _str.replace("[", "p")
        .replace("]", "q")
        .replace(",", "c")
        .replace("(", "p")
        .replace(")", "q")
        .replace(" ", "")
    )


def get_rand_state(rand):
    """
    Utility methods for getting a `RandomState` object.

    :param rand: rand can be None (new State will be generated),
                    np.random.RandomState (it will be returned) or an integer (will be treated as seed).

    :return: a `RandomState` object
    """
    if isinstance(rand, np.random.RandomState):
        return rand
    elif isinstance(rand, (int, np.ndarray, list)) or rand is None:
        return np.random.RandomState(rand)
    else:
        raise ValueError("parameter rand {} has wrong type".format(rand))


# SOME SCORING UTILS FUNCTIONS

half_int = lambda _m: 1.96 * np.std(_m) / np.sqrt(len(_m) - 1)


def mean_std_ci(measures, mul=1.0, tex=False):
    """
    Computes mean, standard deviation and 95% half-confidence interval for a list of measures.

    :param measures: list
    :param mul: optional multiplication coefficient (e.g. for percentage)
    :param tex: if True returns mean +- half_conf_interval for latex
    :return: a list or a string in latex
    """
    measures = np.array(measures) * mul
    ms = np.mean(measures), np.std(measures), half_int(measures)
    return ms if not tex else r"${:.2f} \pm {:.2f}$".format(ms[0], ms[2])


def leaky_relu(x, alpha, name=None):
    """
    Implements leaky relu with negative coefficient `alpha`
    """
    import tensorflow as tf

    with tf.name_scope(name, "leaky_relu_{}".format(alpha)):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)



