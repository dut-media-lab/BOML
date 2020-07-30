from collections import OrderedDict

import tensorflow as tf
from tensorflow.python.training import slot_creator

import py_bml.extension
from py_bml import utils
from py_bml.ULProblem.BMLOuterGrad import BMLOuterGrad


class BMLOuterGradSimple(BMLOuterGrad):
    def __init__(self, inner_method='Simple', history=None, name='BMLOuterOptSimple'):
        super(BMLOuterGradSimple, self).__init__(name)
        self._inner_method = inner_method
        self._history = history or []
        self._reverse_initializer = tf.no_op()
        self.reptile_initializer = tf.no_op()
    # noinspection SpellCheckingInspection

    def compute_gradients(self, outer_objective, optimizer_dict, meta_param=None, param_dict =OrderedDict()):
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
        meta_param = super(BMLOuterGradSimple, self).compute_gradients(outer_objective, optimizer_dict, meta_param)

        with tf.variable_scope(outer_objective.op.name):
            '''
            if len(meta_param) == len(list(optimizer_dict.state)):
                doo_dhypers = tf.gradients(outer_objective,list(optimizer_dict.state))
                #doo_dhypers = tf.gradients(outer_objective, list(optimizer_dict.model_fast_weights.values()))
            else:
                '''

            if 'Reptile' in param_dict.keys() :
                reptile_grads = [model_weight - fast_weight for fast_weight, model_weight
                              in zip(list(optimizer_dict.state), meta_param[:len(optimizer_dict.state)])]
                '''
                doo_dhypers =[model_weight - fast_weight for fast_weight, model_weight
                              in zip(list(optimizer_dict.inner_param_fast_tensor.values()), list(optimizer_dict.state))]
                
                doo_dhypers = [tf.subtract(weight, model_weight) for model_weight, weight
                               in zip(list(optimizer_dict.state), meta_param[:len(optimizer_dict.state)])]
            '''
                reptile_grads += tf.gradients(outer_objective, meta_param[len(optimizer_dict.state):])
                doo_dhypers = self._create_hyper_gradients(hyper_grads=reptile_grads,
                                                           meta_param=meta_param)
                self.reptile_initializer = tf.group(self.reptile_initializer, tf.variables_initializer(doo_dhypers))
            elif param_dict['use_Warp']:
                doo_dhypers = optimizer_dict.outer_param_tensor + optimizer_dict.model_param_tensor
                doo_dhypers += tf.gradients(outer_objective, meta_param[len(doo_dhypers):])
            else:
                doo_dhypers = tf.gradients(outer_objective,
                                           list(optimizer_dict.state) + meta_param[len(optimizer_dict.state):])

            for h, doo_dh in zip(meta_param, doo_dhypers):
                assert doo_dh is not None, BMLOuterGrad._ERROR_HYPER_DETACHED.format(doo_dh)
                self._hypergrad_dictionary[h].append(doo_dh)

            return meta_param

    def _state_feed_dict_generator(self, history, T_or_generator):
        for t, his in zip(utils.solve_int_or_generator(T_or_generator), history):
            yield t, utils.merge_dicts(
                *[od.state_feed_dict(h) for od, h in zip(sorted(self._optimizer_dicts), his)]
            )

    @staticmethod
    def _create_hyper_gradients(hyper_grads, meta_param):
        hyper_gradients = [slot_creator.create_slot(v.initialized_value(), utils.val_or_zero(der, v), 'alpha')
                           for v, der in zip(meta_param, hyper_grads)]
        [tf.add_to_collection(py_bml.extension.GraphKeys.OUTERGRADIENTS, hyper_grad) for hyper_grad in
         hyper_gradients]
        py_bml.extension.remove_from_collection(py_bml.extension.GraphKeys.GLOBAL_VARIABLES, *hyper_gradients)
        # this prevents the 'automatic' initialization with tf.global_variables_initializer.
        return hyper_gradients

    def apply_gradients(self, inner_objective_feed_dicts=None, outer_objective_feed_dicts=None,
                        initializer_feed_dict=None, param_dict=OrderedDict(), train_batches=None, experiments =[], global_step=None, session=None,
                        online=False, callback=None):

        ss = session or tf.get_default_session()

        self._history.clear()

        if not online:
            _fd = utils.maybe_call(initializer_feed_dict, utils.maybe_eval(global_step, ss))
            self._save_history(ss.run(self.initialization, feed_dict=_fd))

        #maybe_init_fd = utils.maybe_call(inner_objective_feed_dicts, utils.maybe_eval(global_step, ss))
        #feed_dicts = utils.merge_dicts(maybe_init_fd, outer_objective_feed_dicts)
        
        if 'Reptile' in param_dict.keys():
            assert len(experiments) > 0, 'must initialize the list of experiment class for assignments of feed dicts'
            mini_datasets = utils.get_mini_dataset(train_batches, exs=experiments,
                                                  num_classes=param_dict['output_shape'], num_shots=param_dict['num_shots']
                                                   , inner_batch_size=param_dict['num_shots'], inner_iters=param_dict['T'])
            for i in range(param_dict['T']):
                mini_batch_feed_dict = utils.feed_train_dicts(mini_batches=mini_datasets, exs=experiments,
                                                              num_classes=param_dict['output_shape'])
                ss.run(self.iteration, mini_batch_feed_dict)
            ss.run(self.reptile_initializer, mini_batch_feed_dict)

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
