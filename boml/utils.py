"""
Contains some utility functions to run your training model and evaluate the performance.
"""
from __future__ import absolute_import, print_function, division

import sys

import numpy as np
import tensorflow as tf

# noinspection PyClassHasNoInit
import boml

Meta_Init_Method = ["MAML", "FOMAML", "WarpGrad", "MT-net"]
Meta_Init_Method = ["Reverse", "Truncated", "DARTS", "BA", "Implicit"]
METHOD_COLLECTIONS = [Meta_Init_Method, Meta_Init_Method]


def remove_from_collection(key, *lst):
    """
    Remove tensors in lst from collection given by key
    """
    try:
        # noinspection PyProtectedMember
        [tf.get_default_graph()._collections[key].remove(_e) for _e in lst]
    except ValueError:
        print(
            "WARNING: Collection -> {} <- does not contain some tensor in {}".format(
                key, lst
            ),
            file=sys.stderr,
        )


def as_tuple_or_list(obj):
    """
    Make sure that `obj` is a tuple or a list and eventually converts it into a list with a single element

    :param obj:
    :return: A `tuple` or a `list`
    """
    return obj if isinstance(obj, (list, tuple)) else [obj]


def merge_dicts(*dicts):
    """
    Merges dictionaries recursively. Accepts also `None` and returns always a (possibly empty) dictionary
    """
    from functools import reduce

    # if len(dicts) == 1: return dicts[0]
    return reduce(lambda a, nd: merge_two_dicts(a, nd if nd else {}), dicts, {})


def merge_two_dicts(x, y):
    z = x.copy()  # start with x's keys and values
    z.update(y)  # modifies z with y's keys and values & returns None
    return z


def as_list(obj):
    """
    Makes sure `obj` is a list or otherwise converts it to a list with a single element.
    """
    return obj if isinstance(obj, list) else [obj]


def vectorize_all(var_list, name=None):
    """Given a list of tensors returns their concatenated vectorization.
    Note that for matrices the vectorization is row-wise instead of column-wise as
    it should be in Magnus. Could it be a problem?

    :param var_list: **bold**
    :param name: optional name for resulting tensor

    :return: vectorization of `var_list`"""
    with tf.name_scope(name, "Vectorization", var_list) as scope:
        return tf.concat([tf.reshape(_w, [-1]) for _w in var_list], 0, name=scope)


def add_list(lst1, lst2):
    """
    sum elements in two lists by sequence
    :return: new list that sum elements in two lists by sequence
    """
    assert len(lst1) == len(lst2), "length of two list must be equal"
    return [ls1 + ls2 for ls1, ls2 in zip(lst1, lst2)]




def mean_std_ci(measures, mul=1.0, tex=False):
    """
    Computes mean, standard deviation and 95% half-confidence interval for a list of measures.

    :param measures: list
    :param mul: optional multiplication coefficient (e.g. for percentage)
    :param tex: if True returns mean +- half_conf_interval for latex
    :return: a list or a string in latex
    """
    half_int = lambda _m: 1.96 * np.std(_m) / np.sqrt(len(_m) - 1)
    measures = np.array(measures) * mul
    ms = np.mean(measures), np.std(measures), half_int(measures)
    return ms if not tex else r"${:.2f} \pm {:.2f}$".format(ms[0], ms[2])


def maybe_call(obj, *args, **kwargs):
    """
    Calls obj with args and kwargs and return its result if obj is callable, otherwise returns obj.
    """
    if callable(obj):
        return obj(*args, **kwargs)
    return obj


def dot(a, b, name=None):
    """
    Dot product between vectors `a` and `b` with optional name.
    If a and b are not vectors, formally this computes <vec(a), vec(b)>.
    """
    with tf.name_scope(name, "Dot", [a, b]):
        return tf.reduce_sum(a * b)


def maybe_eval(a, ss=None):
    """
    Run or eval `a` and returns the result if possible.

    :param a: object, or `tf.Variable` or `tf.Tensor`
    :param ss: `tf.Session` or get default session (if any)
    :return: If a is not a tensorflow evaluable returns it, or returns the
                resulting call
    """
    if ss is None:
        ss = tf.get_default_session()
    if hasattr(a, "eval") or hasattr(a, "run"):
        return ss.run(a)
    return a


def solve_int_or_generator(int_or_generator):
    return range(int_or_generator) if isinteger(int_or_generator) else int_or_generator


def cross_entropy(pred, label):
    """

    :param pred: output of the neural networks
    :param label: the true label paired with the input
    :return:
    """
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label)
    )


def classification_acc(pred, label):
    return tf.contrib.metrics.accuracy(
        tf.argmax(tf.nn.softmax(pred), 1), tf.argmax(label, 1)
    )


def set_gpu():
    # set general configuration
    gpu_config = tf.ConfigProto(allow_soft_placement=True)
    gpu_config.gpu_options.allow_growth = True
    return gpu_config


def get_rand_state(rand=0):
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


def maybe_add(a, b):
    """
    return a if b is None else a + b
    """
    return a if b is None else a + b


def val_or_zero(a, b):
    """
    return a if a is not None else tf.zeros_like(b)
    """
    return a if a is not None else tf.zeros_like(b)


def isinteger(num):
    """
    Judge whether the num is integer
    :param num:
    :return: BOOLEAN
    """
    return isinstance(num, (int, np.int_, np.int8, np.int16, np.int32, np.int64))


def feed_dicts(dat_lst, exs):
    """
    Generate the feed_dicts for boml_optimizer.run() with lists of
    :param dat_lst:
    :param exs:
    :return:
    """
    dat_lst = boml.utils.as_list(dat_lst)
    train_fd = boml.utils.merge_dicts(
        *[
            {_ex.x: dat.train.data, _ex.y: dat.train.target}
            for _ex, dat in zip(exs, dat_lst)
        ]
    )
    valid_fd = boml.utils.merge_dicts(
        *[
            {_ex.x_: dat.test.data, _ex.y_: dat.test.target}
            for _ex, dat in zip(exs, dat_lst)
        ]
    )

    return train_fd, valid_fd


def feed_dict(data_batch, ex):
    """
    Generate the feed_dicts for boml_optimizer.run() with data_batch and the instance of BOMLExperiment
    :param data_batch: each batch of data for exery iteration
    :param ex: instance of BOMLExperiment
    :return:
    """
    data_batch = data_batch[0]
    train_fd = {ex.x: data_batch.train.data, ex.y: data_batch.train.target}
    valid_fd = {ex.x_: data_batch.test.data, ex.y_: data_batch.test.target}
    return train_fd, valid_fd
