from collections import OrderedDict

import tensorflow as tf

from boml.lower_iter import BOMLInnerGradTrad
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
        fast_param, outer_grad, model_grad = BOMLInnerGradSimple.bml_inner_grad_trad(
            loss_inner=loss_inner, param_dict=param_dict, var_list=var_list)

        return BOMLInnerGradSimple(update_op=update_op, dynamics=dynamics,
                                   objective=loss_inner,inner_param_tensor=fast_param,
                                   outer_param_tensor=outer_grad, model_param_tensor=model_grad)

    @staticmethod
    def bml_inner_grad_trad(loss_inner, param_dict=OrderedDict(), var_list=[]):

        assert len(var_list) > 0, 'no task_specific variables to optimizer'
        task_model = param_dict['experiment'].model
        task_param = task_model.task_parameter
        outer_param_grad = []
        model_param_grad = []
        grads = tf.gradients(loss_inner, list(task_param.values()))
        if param_dict['use_Warp']:
            outer_param_loss = param_dict['outer_loss_func'](
                pred=task_model.re_forward(param_dict['experiment'].x_).out,
                label=param_dict['experiment'].y_, method='MetaInit')
            model_param_loss = param_dict['model_loss_func'](
                pred=task_model.re_forward(param_dict['experiment'].x_).out,
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
                outer_param_loss = param_dict['outer_loss_func'](
                    pred=task_model.re_forward(param_dict['experiment'].x_).out,
                    label=param_dict['experiment'].y_, method='MetaInit')
                model_param_loss = param_dict['model_loss_func'](
                    pred=task_model.re_forward(param_dict['experiment'].x_).out,
                    label=param_dict['experiment'].y_, method='MetaInit')
                outer_param_grad = add_list(outer_param_grad, tf.gradients(outer_param_loss, list(var_list)))
                model_param_grad = add_list(model_param_grad, tf.gradients(model_param_loss, list(task_model.model_param_dict.values())))

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
    def apply_updates(self):
        """
        Descent step, as returned by `tf.train.Optimizer.apply_gradients`.
        :return:
        """
        assert self._updates_op is not None, 'descent step operation must be initialized before being called'
        return self._updates_op

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

