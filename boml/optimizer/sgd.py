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
The base class in lower_iter to encapsulate the modified Momentum optimizer based on `tf.train.GradientDescentOptimizer`.
"""
from __future__ import print_function, absolute_import, division

from collections import OrderedDict

import tensorflow as tf

from boml.optimizer.opt import BOMLOpt

# import numpy as np

GRADIENT_NONE_MESSAGE = (
    "WARNING: the gradient w.r.t.the tf.Variable\n {}\n is None;\n "
    "Check the computational graph of the inner objective, and be sure you\n"
    "are not considering including variables that should not be there among the\n"
    "inner variables."
)


class BOMLOptSGD(BOMLOpt, tf.train.GradientDescentOptimizer):
    """Optimizer that implements the gradient descent algorithm.
    """
    def __init__(self, learning_rate, use_locking=False, name="GradientDescent"):
        """Construct a new gradient descent optimizer.

            Args:
              learning_rate: A Tensor or a floating point value.  The learning
                rate to use.
              use_locking: If True use locks for update operations.
              name: Optional name prefix for the operations created when applying
                gradients. Defaults to "GradientDescent".

            @compatibility(eager)
            When eager execution is enabled, `learning_rate` can be a callable that
            takes no arguments and returns the actual value to use. This can be useful
            for changing these values across different invocations of optimizer
            functions.
            @end_compatibility
            """
        super(BOMLOptSGD, self).__init__(learning_rate, use_locking, name)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """Apply gradients to variables.

        This is the second part of `minimize()`. It returns an `Operation` that
        applies gradients.

        Args:
          grads_and_vars: List of (gradient, variable) pairs as returned by
            `compute_gradients()`.
          global_step: Optional `Variable` to increment by one after the
            variables have been updated.
          name: Optional name for the returned operation.  Default to the
            name passed to the `Optimizer` constructor.

        Returns:
          An `Operation` that applies the specified gradients. If `global_step`
          was not None, that operation also increments `global_step`.

        Raises:
          TypeError: If `grads_and_vars` is malformed.
          ValueError: If none of the variables have gradients.
          RuntimeError: If you should use `_distributed_apply()` instead.
        """
        update_op = super(BOMLOptSGD, self).apply_gradients(
            grads_and_vars, global_step, name
        )

        dynamics = OrderedDict()
        for g, w in grads_and_vars:
            assert g is not None, GRADIENT_NONE_MESSAGE.format(w)
            wk = w - tf.cast(self._learning_rate_tensor, g.dtype) * g
            dynamics[w] = wk
        return update_op, dynamics

    def __str__(self):
        return "{}-lr={}".format(self._name, self._learning_rate)
