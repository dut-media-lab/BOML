from collections import OrderedDict

import tensorflow as tf

from boml.lower_iter import BOMLInnerGradTrad


class BOMLInnerGradAggr(BOMLInnerGradTrad):
    def __init__(self, update_op, dynamics, objective, outer_objective):
        self.outer_objective = outer_objective
        super().__init__(update_op=update_op, dynamics=dynamics, objective=objective)

    @staticmethod
    def compute_gradients(bml_opt, loss_inner, loss_outer=None, param_dict=OrderedDict(), var_list=None, **inner_kargs):

        minimize_kargs = {inner_arg: inner_kargs[inner_arg] for inner_arg in
                          set(inner_kargs.keys()) - set(param_dict.keys())}

        assert loss_inner is not None, 'argument:inner_objective must be initialized'
        assert {'alpha', 's', 't', 't_tensor'} <= param_dict.keys(), \
            'Necessary hyper_parameters must be initialized before calling minimize()'
        # alpha, loss_outer, s, t, t_tensor = sorted(param_dict.items(), key=lambda x: x[0])
        update_op, dynamics = BOMLInnerGradAggr.bml_inner_grad_aggr(inner_optimizer=bml_opt, loss_inner=loss_inner,
                                                                    loss_outer=loss_outer, param_dict=param_dict,
                                                                    var_list=var_list, *minimize_kargs)
        return BOMLInnerGradAggr(update_op=update_op, dynamics=dynamics, objective=loss_inner,
                                 outer_objective=loss_outer)

    @staticmethod
    def bml_inner_grad_aggr(inner_optimizer, loss_inner, loss_outer, param_dict=OrderedDict(), global_step=None,
                            var_list=None, gate_gradients=1, aggregation_method=None,
                            colocate_gradients_with_ops=False, name=None,
                            grad_loss=None):

        grads_and_vars_inner = inner_optimizer.compute_gradients(loss_inner, var_list=var_list,
                                                                 gate_gradients=gate_gradients,
                                                                 aggregation_method=aggregation_method,
                                                                 colocate_gradients_with_ops=colocate_gradients_with_ops,
                                                                 grad_loss=grad_loss)
        grads_and_vars_outer = inner_optimizer.compute_gradients(loss_outer, var_list=var_list,
                                                                 gate_gradients=gate_gradients,
                                                                 aggregation_method=aggregation_method,
                                                                 colocate_gradients_with_ops=colocate_gradients_with_ops,
                                                                 grad_loss=grad_loss)
        vars_with_grad = [v for g, v in grads_and_vars_inner if g is not None]
        if not vars_with_grad:
            raise ValueError(
                "No gradients provided for any variable, check your graph for ops"
                " that do not support gradients, between variables %s and loss %s." %
                ([str(v) for _, v in grads_and_vars_inner], loss_inner))
        grads_and_vars = BOMLInnerGradAggr.combine_grads(inner_grads=grads_and_vars_inner,
                                                         outer_grads=grads_and_vars_outer,
                                                         alpha=param_dict['alpha'], s=param_dict['s'],
                                                         t=param_dict['t'],
                                                         t_tensor=param_dict['t_tensor'])

        return inner_optimizer.apply_gradients(grads_and_vars, global_step=global_step,
                                               name=name)

    @staticmethod
    def combine_grads(inner_grads, outer_grads, alpha, s, t, t_tensor):

        combine_grads = []
        if not alpha.get_shape().as_list():
            for _ in range(len(inner_grads)):
                ll_part = (1 - alpha / t_tensor) * t * inner_grads[_][0]
                ul_part = alpha / t_tensor * s * outer_grads[_][0]
                combine_grads.append((ll_part + ul_part, inner_grads[_][1]))
        else:
            for _ in range(len(inner_grads)):
                ll_part = (1 - tf.norm(tf.matmul(alpha, t_tensor), ord=1)) * t * inner_grads[_][0]
                ul_part = tf.norm(tf.matmul(alpha, t_tensor), ord=1) * s * outer_grads[_][0]
                combine_grads.append((ll_part + ul_part, inner_grads[_][1]))
        return combine_grads

    @property
    def apply_updates(self):
        """
        Descent step, as returned by `tf.train.Optimizer.apply_gradients`.
        :return:
        """
        assert self._updates_op is not None, 'descent step operation must be initialized before being called'
        return self._updates_op

    @property
    def iteration(self):
        """
        Performs a descent step (as return by `tf.train.Optimizer.apply_gradients`) and computes the values of
        the variables after it.

        :return: A list of operation that, after performing one iteration, return the value of the state variables
                    being optimized (possibly including auxiliary variables)
        """
        if self._iteration is None:
            with tf.control_dependencies([self._updates_op]):  # ?
                self._iteration = self._state_read()  # performs an iteration and returns the
                # value of all variables in the state (ordered according to dyn)

        return self._iteration

    @property
    def initialization(self):
        """
        :return: a list of operations that return the values of the state variables for this
                    learning dynamics after the execution of the initialization operation. If
                    an initial dynamics is set, then it also executed.
        """
        if self._initialization is None:
            with tf.control_dependencies([tf.variables_initializer(self.state)]):
                if self._init_dyn is not None:  # create assign operation for initialization
                    self._initialization = [k.assign(v) for k, v in self._init_dyn.items()]
                    # return these new initialized values (and ignore variable initializers)
                else:
                    self._initialization = self._state_read()  # initialize state variables and
                    # return the initialized value

        return self._initialization
