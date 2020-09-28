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
The base class in optimizer to encapsulate the modified optimizer based on conventional gradient descent optimizers.  To adapt to the
nested gradient computation of dynamical systems in lower iter and lower iter, BOML integrates modified optimizers (such as SGD with momentum) in optimizer.
"""
from __future__ import print_function, absolute_import, division

import tensorflow as tf


class BOMLOpt(tf.train.Optimizer):
    """
    mirror of the tf.train.Optimizer.
    """

    def minimize(
        self,
        loss_inner,
        var_list=None,
        global_step=None,
        gate_gradients=tf.train.Optimizer.GATE_OP,
        aggregation_method=None,
        colocate_gradients_with_ops=False,
        name=None,
        grad_loss=None,
    ):
        """ The `dynamics` contains a list of var_and_dynamics where var are both variables in `var_list` and also
        additional state (auxiliary) variables to be used in the process of back-propagation.
        """
        update_op, dynamics = super(BOMLOpt, self).minimize(
            loss_inner,
            global_step,
            var_list,
            gate_gradients,
            aggregation_method,
            colocate_gradients_with_ops,
            name,
            grad_loss,
        )
        return update_op, dynamics

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def learning_rate_tensor(self):
        return self._learning_rate_tensor

    @property
    def optimizer_params_tensor(self):
        return [self.learning_rate_tensor]
