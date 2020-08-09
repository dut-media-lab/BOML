from __future__ import absolute_import, print_function, division

from collections import OrderedDict, deque

# import py_bml.OuterOpt.outer_opt_utils as utils
import tensorflow as tf
from tensorflow.python.training import slot_creator

import boml.extension
from boml import utils
from boml.ul_problem.BOMLOuterGrad import BOMLOuterGrad

RAISE_ERROR_ON_DETACHED = False


class BOMLOuterGradDarts(BOMLOuterGrad):

    def __init__(self, inner_method='Reverse', truncate_iter=-1, name='BMLOuterGradDarts'):
        """
       Utility method to initialize truncated reverse HG (not necessarily online),
       :param truncate_iter: Maximum number of iterations that will be stored
       :param name: a name for the operations and variables that will be created
       :return: ReverseHG object
           """
        super(BOMLOuterGradDarts, self).__init__(name)
        self._inner_method = inner_method
        self._alpha_iter = tf.no_op()
        self._reverse_initializer = tf.no_op()
        self._diff_initializer = tf.no_op()
        self._darts_initializer = tf.no_op()
        self._history = deque(maxlen=truncate_iter + 1) if truncate_iter >= 0 else []

    # noinspection SpellCheckingInspection
    def compute_gradients(self, outer_objective, optimizer_dict, meta_param=None, param_dict=OrderedDict()):
        """
        Function that adds to the computational graph all the operations needend for computing
        the hypergradients in a "dynamic" way, without unrolling the entire optimization graph.
        The resulting computation, while being roughly 2x more expensive then unrolling the
        optimizaiton dynamics, requires much less (GPU) memory and is more flexible, allowing
        to set a termination condition to the parameters optimizaiton routine.

        :param optimizer_dict: OptimzerDict object resulting from the inner objective optimization.
        :param outer_objective: A loss function for the outer parameters (scalar tensor)
        :param meta_param: Optional list of outer parameters to consider. If not provided will get all variables in the
                            hyperparameter collection in the current scope.

        :return: list of outer parameters involved in the computation
        """
        meta_param = super(BOMLOuterGradDarts, self).compute_gradients(outer_objective, optimizer_dict, meta_param)

        with tf.variable_scope(outer_objective.op.name):

            ex = self.param_dict['experiment']
            model = self.param_dict['experiment'].model
            loss_func = self.param_dict['loss_func']
            grad_hyper = [self._create_outergradient(outer_objective, hyper) for hyper in meta_param]

            darts_derivatives = [grad for grad in tf.gradients(outer_objective, list(optimizer_dict.state))]
            darts_vector = tf.concat(axis=0, values=utils.vectorize_all(darts_derivatives))
            self.Epsilon = 0.01 / tf.norm(tensor=darts_vector, ord=2)
            darts_derivaives = [self.Epsilon * darts_derivative for darts_derivative in darts_derivatives]
            fin_diff_part = self._create_darts_derivatives(var_list=optimizer_dict.state,
                                                           darts_derivatives=darts_derivaives)
            self._diff_initializer = tf.group(self._diff_initializer,
                                              tf.variables_initializer(fin_diff_part),
                                              tf.variables_initializer(grad_hyper))
            right_diff_0 = dict(zip(model.task_parameter.keys(), [tf.add(state, fin_diff)
                                                                  for state, fin_diff in
                                                                  zip(model.task_parameter.values(),
                                                                      fin_diff_part)]))
            left_diff_0 = dict(zip(model.task_parameter.keys(), [tf.subtract(state, fin_diff)
                                                                 for state, fin_diff in
                                                                 zip(model.task_parameter.values(),
                                                                     fin_diff_part)]))

            left_diff = tf.gradients(loss_func(pred=model.re_forward(task_parameter=left_diff_0).out,
                                               label=ex.y, method='BileveOptim'), xs=meta_param)
            right_diff = tf.gradients(loss_func(pred=model.re_forward(task_parameter=right_diff_0).out,
                                                label=ex.y, method='BilevelOptim'), xs=meta_param)

            grad_param = [tf.divide(tf.add(right_dif, -left_dif), 2 * self.Epsilon)
                          for right_dif, left_dif in zip(right_diff, left_diff)]

            hyper_grads = [grad_hyperparameter - self.param_dict['learning_rate'] * grad_parameter
                           for grad_hyperparameter, grad_parameter in zip(grad_hyper, grad_param)]
            doo_dhypers = self._create_hyper_gradients(hyper_grads=hyper_grads,
                                                       meta_param=meta_param)
            self._darts_initializer = tf.group(self._darts_initializer, tf.variables_initializer(doo_dhypers))
            for h, doo_dh in zip(meta_param, doo_dhypers):
                assert doo_dh is not None, BOMLOuterGrad._ERROR_HYPER_DETACHED.format(doo_dh)
                self._hypergrad_dictionary[h].append(doo_dh)
            return meta_param

    @staticmethod
    def _create_hyper_gradients(hyper_grads, meta_param):
        hyper_gradients = [slot_creator.create_slot(v.initialized_value(), utils.val_or_zero(der, v), 'alpha')
                           for v, der in zip(meta_param, hyper_grads)]
        [tf.add_to_collection(boml.extension.GraphKeys.OUTERGRADIENTS, hyper_grad) for hyper_grad in hyper_gradients]
        boml.extension.remove_from_collection(boml.extension.GraphKeys.GLOBAL_VARIABLES, *hyper_gradients)
        # this prevents the 'automatic' initialization with tf.global_variables_initializer.
        return hyper_gradients

    @staticmethod
    def _create_darts_derivatives(var_list, darts_derivatives):
        derivatives = [slot_creator.create_slot(v.initialized_value(), utils.val_or_zero(der, v), 'alpha') for v, der
                   in zip(var_list, darts_derivatives)]
        [tf.add_to_collection(boml.extension.GraphKeys.DARTS_DERIVATIVES, der) for der in derivatives]
        boml.extension.remove_from_collection(boml.extension.GraphKeys.GLOBAL_VARIABLES, *derivatives)
        # this prevents the 'automatic' initialization with tf.global_variables_initializer.
        return derivatives

    @staticmethod
    def _create_outergradient_from_dodh(hyper, doo_dhypers):
        """
        Creates one hyper-gradient as a variable. doo_dhypers:  initialization, that is the derivative of
        the outer objective w.r.t this hyper
        """
        hgs = slot_creator.create_slot(hyper, utils.val_or_zero(doo_dhypers, hyper), 'outergradient')
        boml.extension.remove_from_collection(boml.extension.GraphKeys.GLOBAL_VARIABLES, hgs)
        return hgs

    @staticmethod
    def _create_outergradient(outer_obj, hyper):
        return BOMLOuterGradDarts._create_outergradient_from_dodh(hyper, tf.gradients(outer_obj, hyper)[0])

    def _state_feed_dict_generator(self, history, T_or_generator):
        for t, his in zip(utils.solve_int_or_generator(T_or_generator), history):
            yield t, utils.merge_dicts(
                *[od.state_feed_dict(h) for od, h in zip(sorted(self._optimizer_dicts), his)]
            )

    def apply_gradients(self, inner_objective_feed_dicts=None, outer_objective_feed_dicts=None,
                        initializer_feed_dict=None, param_dict=OrderedDict(), train_batches=None, experiments=[], global_step=None, session=None,
                        online=False, callback=None):
        callback = utils.as_tuple_or_list(callback)
        # same thing for T
        T_or_generator = utils.as_tuple_or_list(param_dict['T'])

        ss = session or tf.get_default_session()

        self._history.clear()

        def _adjust_step(_t):
            if online:
                _T = utils.maybe_eval(global_step, ss)
                if _T is None:
                    _T = 0
                tot_t = T_or_generator[0]
                if not isinstance(tot_t, int): return _t  # when using a generator there is little to do...
                return int(_t + tot_t * _T)
            else:
                return _t

        if not online:
            _fd = utils.maybe_call(initializer_feed_dict, utils.maybe_eval(global_step, ss))
            self._save_history(ss.run(self.initialization, feed_dict=_fd))

        # else:  # not totally clear if i should add this
        #     self._save_history(ss.run(list(self.state)))

        T = 0  # this is useful if T_or_generator is indeed a generator...
        for t in utils.solve_int_or_generator(T_or_generator[0]):
            # nonlocal t  # with nonlocal would not be necessary the variable T... not compatible with 2.7

            _fd = utils.maybe_call(inner_objective_feed_dicts, _adjust_step(t))

            self._save_history(ss.run(self.iteration, feed_dict=_fd))
            T = t

            utils.maybe_call(callback[0], _adjust_step(t), _fd, ss)  # callback

        # initialization of support variables (supports stochastic evaluation of outer objective via global_step ->
        # variable)

        _fd = utils.maybe_call(outer_objective_feed_dicts, utils.maybe_eval(global_step, ss))
        # now adding also the initializer_feed_dict because of tf quirk...
        darts_init_fd = utils.merge_dicts(_fd, utils.maybe_call(inner_objective_feed_dicts,
                                                                     _adjust_step(t)))
        maybe_init_fd = utils.maybe_call(initializer_feed_dict, utils.maybe_eval(global_step, ss))
        darts_init_fd = utils.merge_dicts(darts_init_fd, maybe_init_fd)
        ss.run(self._diff_initializer, feed_dict=darts_init_fd)

        del self._history[-1]  # do not consider last point

        for pt, state_feed_dict in self._state_feed_dict_generator(reversed(self._history), T_or_generator[-1]):
            # this should be fine also for truncated reverse... but check again the index t
            t = T - pt - 1  # if T is int then len(self.history) is T + 1 and this numerator
            # shall start at T-1

            new_fd = utils.merge_dicts(state_feed_dict, utils.maybe_call(inner_objective_feed_dicts,
                                                                         _adjust_step(t)))

            new_fd = utils.merge_dicts(new_fd, utils.maybe_call(outer_objective_feed_dicts,
                                                                _adjust_step(t)))
            ss.run(self._darts_initializer, new_fd)
            if len(callback) == 2:
                utils.maybe_call(callback[1], _adjust_step(t), new_fd, ss)

    def _save_history(self, weights):
        self._history.append(weights)

    def hypergrad_callback(self, hyperparameter=None, flatten=True):
        """callback that records the partial hypergradients on the reverse pass"""
        values = []
        gs = list(self._hypergrad_dictionary.values()) if hyperparameter is None else \
            self._hypergrad_dictionary[hyperparameter]
        if flatten:
            gs = utils.vectorize_all(gs)

        # noinspection PyUnusedLocal
        def _callback(_, __, ss):
            values.append(ss.run(gs))  # these should not depend from any feed dictionary

        return values, _callback
