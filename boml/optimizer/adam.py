"""
The base class in lower_iter to encapsulate the modified Adam optimizer based on `tf.train.AdamOptimizer`.
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


class BOMLOptAdam(BOMLOpt, tf.train.AdamOptimizer):
    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        use_locking=False,
        name="Adam",
    ):
        r"""Construct a new Adam optimizer.

        Initialization:

        $$m_0 := 0 \text{(Initialize initial 1st moment vector)}$$
        $$v_0 := 0 \text{(Initialize initial 2nd moment vector)}$$
        $$t := 0 \text{(Initialize timestep)}$$

        The update rule for `variable` with gradient `g` uses an optimization
        described at the end of section 2 of the paper:

        $$t := t + 1$$
        $$lr_t := \text{learning\_rate} * \sqrt{1 - beta_2^t} / (1 - beta_1^t)$$

        $$m_t := beta_1 * m_{t-1} + (1 - beta_1) * g$$
        $$v_t := beta_2 * v_{t-1} + (1 - beta_2) * g * g$$
        $$variable := variable - lr_t * m_t / (\sqrt{v_t} + \epsilon)$$

        The default value of 1e-8 for epsilon might not be a good default in
        general. For example, when training an Inception network on ImageNet a
        current good choice is 1.0 or 0.1. Note that since AdamOptimizer uses the
        formulation just before Section 2.1 of the Kingma and Ba paper rather than
        the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
        hat" in the paper.

        The sparse implementation of this algorithm (used when the gradient is an
        IndexedSlices object, typically because of `tf.gather` or an embedding
        lookup in the forward pass) does apply momentum to variable slices even if
        they were not used in the forward pass (meaning they have a gradient equal
        to zero). Momentum decay (beta1) is also applied to the entire momentum
        accumulator. This means that the sparse behavior is equivalent to the dense
        behavior (in contrast to some momentum implementations which ignore momentum
        unless a variable slice was actually used).

        Args:
          learning_rate: A Tensor or a floating point value.  The learning rate.
          beta1: A float value or a constant float tensor. The exponential decay
            rate for the 1st moment estimates.
          beta2: A float value or a constant float tensor. The exponential decay
            rate for the 2nd moment estimates.
          epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper.
          use_locking: If True use locks for update operations.
          name: Optional name for the operations created when applying gradients.
            Defaults to "Adam".  @compatibility(eager) When eager execution is
            enabled, `learning_rate`, `beta1`, `beta2`, and `epsilon` can each be a
            callable that takes no arguments and returns the actual value to use.
            This can be useful for changing these values across different
            invocations of optimizer functions. @end_compatibility
        """
        super(BOMLOptAdam, self).__init__(
            learning_rate, beta1, beta2, epsilon, use_locking, name
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
        update_op = super(BOMLOptAdam, self).apply_gradients(
            grads_and_vars, global_step, name
        )

        mn, vn = self.get_slot_names()
        dynamics = OrderedDict()

        with tf.name_scope(name, "Adam_Dynamics"):
            try:
                b1_pow, b2_pow = self._beta1_power, self._beta2_power
            except AttributeError:
                b1_pow, b2_pow = self._get_beta_accumulators()
            lr_k = self._lr_t * tf.sqrt(1.0 - b2_pow) / (1.0 - b1_pow)

            lr_k = tf.cast(lr_k, grads_and_vars[0][0].dtype)
            self._beta1_t = tf.cast(self._beta1_t, grads_and_vars[0][0].dtype)
            self._beta2_t = tf.cast(self._beta2_t, grads_and_vars[0][0].dtype)
            self._epsilon_t = tf.cast(self._epsilon_t, grads_and_vars[0][0].dtype)

            for g, w in grads_and_vars:
                assert g is not None, GRADIENT_NONE_MESSAGE.format(w)

                m = self.get_slot(w, mn)
                v = self.get_slot(w, vn)
                mk = tf.add(
                    self._beta1_t * m, (1.0 - self._beta1_t) * g, name=m.op.name
                )
                vk = tf.add(
                    self._beta2_t * v, (1.0 - self._beta2_t) * g * g, name=v.op.name
                )

                wk = tf.subtract(
                    w, lr_k * mk / (tf.sqrt(vk + self._epsilon_t ** 2)), name=w.op.name
                )
                # IMPORTANT NOTE: epsilon should be outside sqrt as from the original implementation,
                # but this brings to numerical instability of the outergradient.

                dynamics[w] = wk
                dynamics[m] = mk
                dynamics[v] = vk

            b1_powk = b1_pow * self._beta1_t
            b2_powk = b2_pow * self._beta2_t

            dynamics[b1_pow] = b1_powk
            dynamics[b2_pow] = b2_powk
        return update_op, dynamics

    def __str__(self):
        return "{}-lr={}-b1={}-b=2{}-ep={}".format(
            self._name, self._lr, self._beta1, self._beta2, self._epsilon
        )

    @property
    def learning_rate(self):
        return self._lr

    @property
    def learning_rate_tensor(self):
        return self._lr_t

    @property
    def optimizer_params_tensor(self):
        return super(BOMLOptAdam, self).optimizer_params_tensor + [
            self._beta1_t,
            self._beta2_t,
        ]
