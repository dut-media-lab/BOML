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
The base class in lower_iter to encapsulate the modified Momentum optimizer based on `tf.train.MomentumOptimizer`.
"""
from __future__ import print_function, absolute_import, division

from collections import OrderedDict

import tensorflow as tf
from boml.optimizer.opt import BOMLOpt

GRADIENT_NONE_MESSAGE = (
    "WARNING: the gradient w.r.t.the tf.Variable\n {}\n is None;\n "
    "Check the computational graph of the inner objective, and be sure you\n"
    "are not considering including variables that should not be there among the\n"
    "inner variables."
)


class BOMLOptMomentum(BOMLOpt, tf.train.MomentumOptimizer):
    """Optimizer that implements the Momentum algorithm.

    Computes (if `use_nesterov = False`):

    ```
    accumulation = momentum * accumulation + gradient
    variable -= learning_rate * accumulation
    ```

    Note that in the dense version of this algorithm, `accumulation` is updated
    and applied regardless of a gradient's value, whereas the sparse version (when
    the gradient is an `IndexedSlices`, typically because of `tf.gather` or an
    embedding) only updates variable slices and corresponding `accumulation` terms
    when that part of the variable was used in the forward pass.
    """
    def __init__(
        self,
        learning_rate,
        momentum,
        use_locking=False,
        name="Momentum",
        use_nesterov=False,
    ):
        """Construct a new Momentum optimizer.

          Args:
            learning_rate: A `Tensor` or a floating point value.  The learning rate.
            momentum: A `Tensor` or a floating point value.  The momentum.
            use_locking: If `True` use locks for update operations.
            name: Optional name prefix for the operations created when applying
              gradients.  Defaults to "Momentum".
            use_nesterov: If `True` use Nesterov Momentum.
              See [Sutskever et al., 2013](
              http://jmlr.org/proceedings/papers/v28/sutskever13.pdf).
              This implementation always computes gradients at the value of the
              variable(s) passed to the optimizer. Using Nesterov Momentum makes the
              variable(s) track the values called `theta_t + mu*v_t` in the paper.
              This implementation is an approximation of the original formula, valid
              for high values of momentum. It will compute the "adjusted gradient"
              in NAG by assuming that the new gradient will be estimated by the
              current average gradient plus the product of momentum and the change
              in the average gradient.
        """
        assert use_nesterov is False, "Nesterov momentum not implemented yet..."
        super(BOMLOptMomentum, self).__init__(
            learning_rate, momentum, use_locking, name, use_nesterov
        )

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
        update_op = super(BOMLOptMomentum, self).apply_gradients(
            grads_and_vars, global_step, name
        )

        # builds up the dynamics here
        mn = self.get_slot_names()[0]
        dynamics = OrderedDict()
        for g, w in grads_and_vars:
            assert g is not None, GRADIENT_NONE_MESSAGE.format(w)

            m = self.get_slot(w, mn)
            mk = tf.cast(self._momentum_tensor, m.dtype) * m + g
            wk = w - tf.cast(self._learning_rate_tensor, mk.dtype) * mk
            dynamics[w] = wk
            dynamics[m] = mk

        return update_op, dynamics

    def __str__(self):
        return "{}-lr={}-m={}".format(self._name, self._learning_rate, self._momentum)

    @property
    def optimizer_params_tensor(self):
        return super(BOMLOptMomentum, self).optimizer_params_tensor + [
            self._momentum_tensor
        ]
