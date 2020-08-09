from collections import OrderedDict

import tensorflow as tf

from boml.ll_problem import BOMLInnerGradTrad
from boml.utils import add_list


class BOMLInnerGradSimple(BOMLInnerGradTrad):
    def __init__(self, update_op, dynamics, objective, inner_param_tensor=None,
                 outer_param_tensor=[], model_param_tensor=[]):
        self._inner_param_fast_tensor = inner_param_tensor
        self._outer_param_tensor = outer_param_tensor
        self._model_param_tensor = model_param_tensor
        super().__init__(update_op=update_op, dynamics=dynamics, objective=objective)

    @staticmethod
    def compute_gradients(bml_opt, loss_inner, loss_outer=None, param_dict=OrderedDict(),
                 var_list=None, **inner_kargs):

        minimize_kargs = {inner_arg: inner_kargs[inner_arg] for inner_arg in
                          set(inner_kargs.keys()) - set(param_dict.keys())}

        assert loss_inner is not None, 'argument:inner_objective must be initialized'
        update_op, dynamics = bml_opt.minimize(loss_inner, var_list, *minimize_kargs)
        if len(param_dict) > 0 and len(var_list) > 0:
            fast_param, outer_grad, model_grad = BOMLInnerGradSimple.bml_inner_grad_trad(
                                                                                                           loss_inner=loss_inner,param_dict=param_dict, var_list=var_list)

            return BOMLInnerGradSimple(update_op=update_op, dynamics=dynamics, objective=loss_inner,
                                       inner_param_tensor=fast_param, outer_param_tensor=outer_grad, model_param_tensor=model_grad)
        else:
            return BOMLInnerGradSimple(update_op=update_op, dynamics=dynamics, objective=loss_inner)

    @staticmethod
    def bml_inner_grad_trad(loss_inner, param_dict=OrderedDict(), var_list=[]):

        assert len(var_list) > 0, 'no task_specific variables to optimizer'
        task_model = param_dict['experiment'].model
        task_param = task_model.task_parameter
        outer_param_grad = []
        model_param_grad = []
        grads = tf.gradients(loss_inner, list(task_param.values()))
        if param_dict['use_Warp']:
            outer_param_loss =param_dict['outer_loss_func'](pred=task_model.re_forward(param_dict['experiment'].x_).out,
                                                      label=param_dict['experiment'].y_, method='MetaInit')
            model_param_loss =param_dict['model_loss_func'](pred=task_model.re_forward(param_dict['experiment'].x_).out,
                                                      label=param_dict['experiment'].y_, method='MetaInit')
            outer_param_grad = tf.gradients(outer_param_loss, list(var_list))
            model_param_grad = tf.gradients(model_param_loss, list(task_model.model_param_dict.values()))

        if param_dict['first_order']:
            grads_dict = dict(zip(task_model.task_parameter.keys(), [tf.stop_gradient(grad) for grad in grads]))
        else:
            grads_dict = dict(zip(task_model.task_parameter.keys(), grads))

        task_model.task_parameter = OrderedDict(zip(task_param.keys(),
                                                    [task_param[key] - param_dict['learning_rate'] * grads_dict[key] for key
                                                     in task_param.keys()]))

        for _ in range(param_dict['T'] - 1):
            task_model = task_model.re_forward(param_dict['experiment'].x)
            task_param = task_model.task_parameter

            iter_loss = param_dict['loss_func'](pred=task_model.out, label=param_dict['experiment'].y,method='MetaInit')

            if param_dict['use_Warp']:
                outer_param_loss = param_dict['outer_loss_func'](pred=task_model.re_forward(param_dict['experiment'].x_).out,
                                                           label=param_dict['experiment'].y_, method='MetaInit')
                model_param_loss = param_dict['model_loss_func'](pred=task_model.re_forward(param_dict['experiment'].x_).out,
                                                           label=param_dict['experiment'].y_, method='MetaInit')
                outer_param_grad = add_list(outer_param_grad,tf.gradients(outer_param_loss, list(var_list)))
                model_param_grad = add_list(model_param_grad,tf.gradients(model_param_loss, list(task_model.model_param_dict.values())))

            grads = tf.gradients(iter_loss, list(task_param.values()))

            if param_dict['first_order']:
                grads_dict = dict(zip(task_model.task_parameter.keys(), [tf.stop_gradient(grad) for grad in grads]))
            else:
                grads_dict = dict(zip(task_model.task_parameter.keys(), grads))

            task_model.task_parameter = OrderedDict(zip(task_param.keys(),
                                                        [task_param[key] - param_dict['learning_rate'] * grads_dict[key] for
                                                         key in task_param.keys()]))
        param_dict['experiment'].model = task_model

        return task_model.task_parameter, outer_param_grad, model_param_grad

    @property
    def inner_param_fast_tensor(self):
        """
        :return: temporary weights dictionary used in maml and fomaml for back propagation
        """
        assert self._inner_param_fast_tensor is not None, \
            'temporary weights dictionary must be initialized before being called'
        return self._inner_param_fast_tensor

    @property
    def outer_param_tensor(self):
        """
        :return: temporary weights dictionary used in maml and fomaml for back propagation
        """
        assert self._outer_param_tensor is not None, \
            'temporary weights dictionary must be initialized before being called'
        return self._outer_param_tensor

    @property
    def model_param_tensor(self):
        """
        :return: temporary weights dictionary used in maml and fomaml for back propagation
        """
        assert self._model_param_tensor is not None, \
            'temporary weights dictionary must be initialized before being called'
        return self._model_param_tensor

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

    @property
    def dynamics(self):
        """
        :return: A generator for the dynamics (state_variable_{k+1})
        """
        return self._dynamics.values()

    @property
    def dynamics_dict(self):
        return self._dynamics

    @property
    def state(self):
        """
        :return: A generator for all the state variables (optimized variables and possibly auxiliary variables)
        being optimized
        """
        return self._dynamics.keys()  # overridden in Adam

    def _state_read(self):
        """
        :return: generator of read value op for the state variables
        """
        return [v.read_value() for v in self.state]  # not sure about read_value vs value

    def state_feed_dict(self, his):
        """
        Builds a feed dictionary of (past) states
        """
        return {v: his[k] for k, v in enumerate(self.state)}

    def set_init_dynamics(self, init_dictionary):
        """
        With this function is possible to set an initializer for the dynamics. Multiple calls of this method on the
        same variable will override the dynamics.

        :param init_dictionary: a dictionary of (state_variable: tensor or variable, that represents the initial
                                dynamics Phi_0.
        """
        if self._init_dyn is None:
            self._init_dyn = OrderedDict([(v, tf.identity(v)) for v in self.state])  # do nothing
        for k, v in init_dictionary.items():
            assert k in self._init_dyn, 'Can set initial dynamics only for state variables in this object, got %s' % k
            self._init_dyn[k] = v

    @property
    def init_dynamics(self):
        """
        :return: The initialization dynamics if it has been set, or `None` otherwise.
        """
        return None if self._init_dyn is None else list(self._init_dyn.items())

    def __lt__(self, other):  # make OptimizerDict sortable
        # TODO be sure that this is consistent
        assert isinstance(other, BOMLInnerGradSimple)
        return hash(self) < hash(other)

    def __len__(self):
        return len(self._dynamics)
