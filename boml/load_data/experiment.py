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
Simple container for useful quantities for a supervised learning BOMLExperiment, where data is managed
with feed dictionary
"""
import tensorflow as tf


class BOMLExperiment:
    def __init__(self, datasets, dtype=tf.float32):
        """
        Define data formats and manage attributes of the generated tasks.
        :param datasets: instance of Meta Dataset
        :param dtype: default to be tf.float32
        """

        self.datasets = datasets
        self.x = tf.placeholder(dtype, name="x", shape=self._compute_input_shape())
        self.y = tf.placeholder(dtype, name="y", shape=self._compute_output_shape())
        self.x_ = tf.placeholder(dtype, name="x_", shape=self._compute_input_shape())
        self.y_ = tf.placeholder(dtype, name="y_", shape=self._compute_output_shape())
        self.dtype = dtype
        self.model = None
        self.errors = {}
        self.scores = {}
        self.optimizers = {}

    # noinspection PyBroadException
    def _compute_input_shape(self):
        """
        :return: compute the shape of input according to the data formats
        """
        sh = self.datasets.train.dim_data
        return (None, sh) if isinstance(sh, int) else (None,) + sh

    # noinspection PyBroadException
    def _compute_output_shape(self):
        """
        :return: compute the shape of output according to the data formats
        """
        sh = self.datasets.train.dim_target
        return (None, sh) if isinstance(sh, int) else (None,) + sh
