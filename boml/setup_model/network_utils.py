"""
Contains some misc utility functions
"""

from functools import reduce

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers


def conv_block(bomlnet, cweight, bweight):

    """ Perform, conv, batch norm, nonlinearity, and max pool """
    if bomlnet.max_pool:
        conv_out = tf.add(
            tf.nn.conv2d(bomlnet.out, cweight, bomlnet.no_stride, "SAME"), bweight
        )
    else:
        conv_out = tf.add(
            tf.nn.conv2d(bomlnet.out, cweight, bomlnet.stride, "SAME"), bweight
        )
    if bomlnet.batch_norm is not None:
        batch_out = bomlnet.batch_norm(
            conv_out,
            activation_fn=bomlnet.activation,
            variables_collections=bomlnet.var_collections,
        )
    else:
        batch_out = bomlnet.activation(conv_out)
    if bomlnet.max_pool:
        final_out = tf.nn.max_pool(batch_out, bomlnet.stride, bomlnet.stride, "VALID")
        return final_out
    else:
        return batch_out


def conv_block_t(bomlnet, conv_weight, conv_bias, zweight):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    if bomlnet.max_pool:
        conv_out = tf.add(
            tf.nn.conv2d(bomlnet.out, conv_weight, bomlnet.no_stride, "SAME"), conv_bias
        )
    else:
        conv_out = tf.add(
            tf.nn.conv2d(bomlnet.out, conv_weight, bomlnet.stride, "SAME"), conv_bias
        )
    conv_output = tf.nn.conv2d(conv_out, zweight, bomlnet.no_stride, "SAME")

    if bomlnet.batch_norm is not None:
        batch_out = layers.batch_norm(
            conv_output,
            activation_fn=bomlnet.activation,
            variables_collections=bomlnet.var_collections,
        )
    else:
        batch_out = bomlnet.activation(conv_output)
    if bomlnet.max_pool:
        final_out = tf.nn.max_pool(batch_out, bomlnet.stride, bomlnet.stride, "VALID")
        return final_out
    else:
        return batch_out


def conv_block_warp(bomlnet, cweight, bweight, zweight, zbias):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    if bomlnet.max_pool:
        conv_out = tf.add(
            tf.nn.conv2d(bomlnet.out, cweight, bomlnet.no_stride, "SAME"), bweight
        )
    else:
        conv_out = tf.add(
            tf.nn.conv2d(bomlnet.out, cweight, bomlnet.stride, "SAME"), bweight
        )

    conv_output = tf.add(
        tf.nn.conv2d(conv_out, zweight, bomlnet.no_stride, "SAME"), zbias
    )

    if bomlnet.batch_norm is not None:
        batch_out = layers.batch_norm(
            conv_output,
            activation_fn=bomlnet.activation,
            variables_collections=bomlnet.var_collections,
        )
    else:
        batch_out = bomlnet.activation(conv_output)
    if bomlnet.max_pool:
        final_out = tf.nn.max_pool(batch_out, bomlnet.stride, bomlnet.stride, "VALID")
        return final_out
    else:
        return batch_out


def get_conv_weight(bomlnet, layer, initializer):
    if layer == 0:
        return tf.get_variable(
            "conv" + str(layer),
            [bomlnet.kernel, bomlnet.kernel, bomlnet.channels, bomlnet.dim_hidden[0]],
            initializer=initializer,
            dtype=bomlnet.datatype,
        )
    else:
        return tf.get_variable(
            "conv" + str(layer),
            [
                bomlnet.kernel,
                bomlnet.kernel,
                bomlnet.dim_hidden[layer - 1],
                bomlnet.dim_hidden[layer],
            ],
            initializer=initializer,
            dtype=bomlnet.datatype,
        )


def get_warp_weight(bomlnet, layer, initializer):
    return tf.get_variable(
        "conv" + str(layer) + "_z",
        [
            bomlnet.kernel,
            bomlnet.kernel,
            bomlnet.dim_hidden[layer - 1],
            bomlnet.dim_hidden[layer],
        ],
        initializer=initializer,
        dtype=bomlnet.datatype,
    )


def get_warp_bias(bomlnet, layer, initializer):
    return tf.get_variable(
        "bias" + str(layer) + "_z",
        [bomlnet.dim_hidden[layer]],
        initializer=initializer,
        dtype=bomlnet.datatype,
    )


def get_bias_weight(bomlnet, layer, initializer):
    return tf.get_variable(
        "bias" + str(layer),
        [bomlnet.dim_hidden[layer]],
        initializer=initializer,
        dtype=bomlnet.datatype,
    )


def get_identity(dim, name, conv=True):
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
    return obj[i] if hasattr(obj, "__getitem__") else obj


def merge_dicts(*dicts):
    return reduce(lambda a, nd: {**a, **nd}, dicts, {})


def to_one_hot_enc(seq, dimension=None):
    da_max = dimension or int(np.max(seq)) + 1
    _tmp = np.zeros((len(seq), da_max))
    _tmp[range(len(_tmp)), np.array(seq, dtype=int)] = 1
    return _tmp
    #
    # def create_and_set(_p):
    #     _tmp = np.zeros(da_max)
    #     _tmp[int(_p)] = 1
    #     return _tmp
    #
    # return np.array([create_and_set(_v) for _v in seq])


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


def get_global_step(name="GlobalStep", init=0):
    import tensorflow as tf

    return tf.get_variable(
        name,
        initializer=init,
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES],
    )
