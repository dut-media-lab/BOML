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
Contains some utility functions to run your training model and evaluate the performance.
"""
from __future__ import absolute_import, print_function, division

import sys
import pickle
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
    :return: cross-entropy loss function
    """
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label)
    )


def mse(pred, label):
    """
    :param pred: output of the neural networks
    :param label: the true label paired with the input
    :return: msw loss function
    """
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))


def get_L2Reg(var_list=None, rate=0.0):
    """
    Return the L1 regularization item
    :param var_list: list of parameters
    :param rate: regularization rate
    :return: L2_regularization_part
    """
    regularizer = tf.contrib.layers.l2_regularizer(rate)
    reg_l2 = tf.contrib.layers.apply_regularization(regularizer, weights_list=var_list)
    return reg_l2


def get_L1Reg(var_list=None, rate=0.0):
    """
    Return the L1 regularization item
    :param var_list: list of parameters
    :param rate: regularization rate
    :return: L1_regularization_part
    """
    regularizer = tf.contrib.layers.l1_regularizer(rate)
    reg_l1 = tf.contrib.layers.apply_regularization(regularizer, weights_list=var_list)
    return reg_l1


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


class BatchQueueMock:
    # Class for debugging purposes for multi-thread issues
    def __init__(self, metadataset, n_batches, batch_size, rand):
        """
        :param metadataset: instance of data set
        :param n_batches: number of batches
        :param batch_size: size of batch
        :param rand: int, used for generating random numbers
        """
        self.metadataset = metadataset
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.rand = rand

    def get_all_batches(self):
        """
        :return:
        """
        return [
            d
            for d in self.metadataset.generate(
                self.n_batches, self.batch_size, self.rand
            )
        ]

    def get_single_batch(self):
        return [d for d in self.metadataset.generate(self.n_batches, 1, self.rand)]


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


def save_obj(file_path, obj):
    """
    :param file_path: path to save the pickle file
    :param obj:
    :return:
    """
    with open(file_path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_obj(file_path):
    """
    :param file_path: path to save the pickle file
    :return:
    """
    with open(file_path, "rb") as handle:
        b = pickle.load(handle)
    return b


