import sys
from collections import OrderedDict

import tensorflow as tf
from tensorflow.python.training import slot_creator

import py_bml.extension
from py_bml import utils
from py_bml.ULProblem.BMLOuterGrad import BMLOuterGrad

RAISE_ERROR_ON_DETACHED = False


class BMLOuterGradForward(BMLOuterGrad):
    def __init__(self, inner_method='Trad', name='BMLOuterGradForward'):
        super(BMLOuterGradForward, self).__init__(name)
        self._inner_method = inner_method
        self._forward_initializer = tf.no_op()
        self._zs = {}  # hyperparameter - zs dictionary
        self._z_iter = tf.no_op()
        self._iteration = None
        self.A_dot_zs = {}

    _HYPER_RANK_ERROR_MESSAGE = """
    ForwardHG: Only scalar outer parameters accepted.\n
     Hyperparameter tensor {} has rank {}.\n
     Use keyword argument extension.get_outerparameter(..., scalar=True) on hyperparameter creation.
    """

    def compute_gradients(self, outer_objective, optimizer_dict, meta_param=None, param_dict=OrderedDict()):
        meta_param = super(BMLOuterGradForward, self).compute_gradients(outer_objective, optimizer_dict, meta_param)

        # scalar_meta_param

        with tf.variable_scope(outer_objective.op.name):
            # dynamics_vec = vectorize_all(optimizer_dict.dynamics)  # in the new implementation there's no need of
            # vectorizing... it might be more efficient since it's better to avoid too many reshaping operations...
            d_oo_d_state = tf.gradients(outer_objective, list(optimizer_dict.state))

            with tf.name_scope('DUMMY'):  # variables to compute forward propagation
                # TODO avoid this computation if optimizer_dict has already been seen.
                aux_vs = [tf.zeros_like(v) for v in optimizer_dict.state]
                dynamics_dot_aux_v = utils.reduce_all_sums(list(optimizer_dict.dynamics), aux_vs)

                der_dynamics_dot_aux_v = tf.gradients(dynamics_dot_aux_v, list(optimizer_dict.state))
                # this is a list of jacobians times aux_vs that have the same dimension of states variables.

                init_dynamics_dot_aux_v = None
                if optimizer_dict.init_dynamics:
                    init_dynamics_dot_aux_v = utils.reduce_all_sums(
                        optimizer_dict.init_dynamics, aux_vs)

            for hyp in meta_param:
                assert hyp.shape.ndims == 0, BMLOuterGradForward._HYPER_RANK_ERROR_MESSAGE.format(hyp, hyp.shape.ndims)

                d_init_dyn_d_hyp = None if init_dynamics_dot_aux_v is None else \
                    tf.gradients(init_dynamics_dot_aux_v, hyp)[0]
                d_dyn_d_hyp = tf.gradients(dynamics_dot_aux_v, hyp)[0]
                d_oo_d_hyp = tf.gradients(outer_objective, hyp)[0]

                # ------------------------------------------------------------
                # check detached outer parameters (for which outer gradient would be always null)
                hyper_ok = d_init_dyn_d_hyp is not None or d_dyn_d_hyp is not None or d_oo_d_hyp is not None
                if RAISE_ERROR_ON_DETACHED:
                    # try:
                    assert hyper_ok, BMLOuterGrad._ERROR_HYPER_DETACHED.format(hyp)
                    # ex
                else:
                    if not hyper_ok:
                        print(BMLOuterGrad._ERROR_HYPER_DETACHED.format(hyp), file=sys.stderr)
                        meta_param.remove(hyp)
                # -------------------------------------------------------------

                # UPDATE OF TOTAL DERIVATIVE OF STATE W.R.T. HYPERPARAMETER
                zs = BMLOuterGradForward._create_zs(
                    optimizer_dict, hyp, None if d_init_dyn_d_hyp is None else tf.gradients(d_init_dyn_d_hyp, aux_vs)
                )  # this is one z for each variable
                self._zs[hyp] = zs  # store a reference for the total derivatives for easy access
                Bs = tf.gradients(d_dyn_d_hyp, aux_vs)

                A_dot_zs = tf.gradients(utils.reduce_all_sums(der_dynamics_dot_aux_v, zs), aux_vs)

                self.A_dot_zs[hyp] = A_dot_zs

                _z_iter = tf.group(*[
                    z.assign(utils.maybe_add(A_dot_z, B)) for z, A_dot_z, B
                    in zip(zs, A_dot_zs, Bs)
                ])
                self._z_iter = tf.group(self._z_iter, _z_iter)

                # -- HYPERGRADIENT -----
                d_E_T = [utils.dot(d_oo_d_s, z) for d_oo_d_s, z in zip(d_oo_d_state, zs)
                         if d_oo_d_s is not None and z is not None]  # list of dot products
                hg = utils.maybe_add(tf.reduce_sum(d_E_T), d_oo_d_hyp)  # sum the partial dot products and possibly ->
                # adds the ''direct derivative'' term d(E( . , \lambda))/d \lambda

                self._hypergrad_dictionary[hyp].append(hg)
                self._forward_initializer = tf.group(self._forward_initializer,
                                                     tf.variables_initializer(zs))
        return meta_param

    @staticmethod
    def _create_zs(optimizer_dict, hyper, d_init_dynamics_d_hyper):
        if d_init_dynamics_d_hyper is None: d_init_dynamics_d_hyper = [None] * len(optimizer_dict)
        with tf.variable_scope('Z'):
            z = [slot_creator.create_slot(v, utils.val_or_zero(der, v), hyper.op.name) for v, der
                 in zip(optimizer_dict.state, d_init_dynamics_d_hyper)]
            [tf.add_to_collection(py_bml.extension.GraphKeys.ZS, lm) for lm in z]
            # in this case it is completely fine to keep zs into the global variable...
            return z

    def apply_gradients(self, inner_objective_feed_dicts=None, outer_objective_feed_dicts=None,
                        initializer_feed_dict=None, param_dict=OrderedDict(), train_batches=None, experiments=[], global_step=None, session=None,
                        online=False, callback=None):

        ss = session or tf.get_default_session()

        if not online:
            self._run_batch_initialization(ss, utils.maybe_call(
                initializer_feed_dict, utils.maybe_eval(global_step, ss)))

        for t in utils.solve_int_or_generator(param_dict['T']):
            _fd = utils.maybe_call(inner_objective_feed_dicts, t)
            self._forward_step(ss, _fd)
            utils.maybe_call(callback, t, _fd, ss)

    def _forward_step(self, ss, _fd):
        ss.run(self._z_iter, _fd)
        ss.run(self.iteration, _fd)

    def _run_batch_initialization(self, ss, fd):
        ss.run(self.initialization, feed_dict=fd)
        ss.run(self._forward_initializer, feed_dict=fd)

    @property
    def w_dots(self):
        return [{h: self._zs[h][k] for h in self._zs} for k, _ in enumerate(self.state)]

    def z_callback(self, hyperparameter=None, flatten=True):
        zs_values = []
        zs = list(self._zs.values()) if hyperparameter is None else self._zs[hyperparameter]
        if flatten: zs = utils.vectorize_all(zs)

        def _callback(_, __, ss):
            zs_values.append(ss.run(zs))

        return zs_values, _callback
