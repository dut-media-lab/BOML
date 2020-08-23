from collections import defaultdict, OrderedDict

import numpy as np
import tensorflow as tf

from boml import utils, extension
from boml.load_data import ImageNetMetaDataset, OmniglotMetaDataset, BMLExperiment

try:
    importlib = __import__('importlib')
    bml_networks = importlib.import_module('boml.setup_model')
    inner_grads = importlib.import_module('boml.lower_iter')
    hyper_grads = importlib.import_module('boml.upper_iter')
    bml_optimizer = importlib.import_module('boml.optimizer')
except ImportError:
    print('Error encountered when importing boml modules in BOMLOptimizer')


class BOMLOptimizer(object):
    """
    Wrapper for performing gradient-based metaparameter optimization
    """

    def __init__(self, method=None, inner_method=None, outer_method=None, truncate_iter=-1, experiments=[]):
        """
        BMLHOptimizer is the main class in `pybml`, which takes responsibility for
        the whole process of model construnction and back propagation.

        :param Method: define basic method for following training process, it should be included in ['MetaInit', 'MetaRepr'],
         'MetaInit' type includes methods like 'MAML, FOMAML, TNet, WarpGrad'; 'MetaRepr' type includes methods like
         'BDA, RHG, TRHG, Implicit HG, DARTS';
        :param inner_method: method chosen for solving LLproblem, including ['Trad' ,'Simple', 'Aggr'], 'MetaRepr' type choose
        either 'Trad' for traditional optimization strategies or 'Aggr' for Gradient Aggragation optimization 'MetaInit' type
        should choose 'Simple', and set specific parameters for detailed method choices like FOMAML or TNet.
        :param outer_method: method chosen for solving LLproblem, including ['Reverse' ,'Simple', 'Forward', 'Implcit'],
        'MetaInit' type should choose Simple, and set specific parameters for detailed method choices like FOMAML
        :param truncate_iter: specific parameter for Truncated Reverse method, defining number of iterations to truncate
         in the Back propagation process
        :param experiments: list of experiment objects that has already been initialized
        :return :an initialized instance of BMLHOptimizer
        """
        assert method in ('MetaRepr', 'MetaInit'),\
            'initialize method arguement, should be in list \an [MetaRepr,MetaInitl] ' \
            'MetaRepr based methods include [BDA,FHG,RHG,TRHG],' \
            'HperOptim based methods include [MAML,FOMAML,MSGD]'
        self._method = method
        assert inner_method in ('Aggr', 'Simple', 'Trad'), \
            'initialize method arguement, should be in list [Aggr, Simple, Trad]'
        self.inner_method = inner_method

        if self.inner_method in ('Aggr', 'Trad') and outer_method =='Reverse':
            outer_method = 'Reverse'
        elif self.inner_method == 'Simple' and outer_method == 'Simple':
            outer_method = 'Simple'
        elif self.inner_method == 'Trad' and outer_method == 'Implicit':
            outer_method = 'Implicit'
        elif self.inner_method in ('Aggr', 'Trad') and outer_method == 'Darts':
            outer_method = 'Darts'
        else:
            print('Invalid combination of inner and outer methods, \
            please check the initialization for different level of problems or '
                  'extend the base classes to formulate your own problems definition')
            raise AssertionError
        self.outer_method = outer_method
        self._inner_gradient = getattr(inner_grads, '%s%s' % ('BOMLInnerGrad', self.inner_method))
        if truncate_iter >= 0:
            self._outer_gradient = getattr(hyper_grads, '%s%s' % ('BOMLOuterGrad', self.outer_method))(
                inner_method=self.inner_method, truncate_iter=truncate_iter)
        else:
            self._outer_gradient = getattr(hyper_grads, '%s%s' % ('BOMLOuterGrad', self.outer_method))(
                inner_method=self.inner_method)
        self.io_opt = None
        self._learning_rate = None
        self._meta_learning_rate = None
        self.oo_opt = None
        self.data_set = None
        self._meta_model = None
        self._fin_hts = None
        self._global_step = None
        self._o_optim_dict = defaultdict(lambda: set())
        self._param_dict = OrderedDict()
        self._global_step = tf.get_variable('global_step', initializer=0, trainable=False)
        self._inner_objective = None
        self._truncate_iterations = -1
        self.experiments = experiments
        # self.inner_objectives = []

    def meta_learner(self, _input, dataset, meta_model='v1', name='Hyper_Net', use_T=False, use_Warp=False,**model_args):
        """
        This method must be called once at first to build meta modules and initialize meta parameters and neural networks.
        :param _input: orginal input for neural network construction
        :param dataset: which dataset to use for training and testing. It should be initialized before being passed into the function
        :param meta_model: model chosen for neural network construction, 'v1' for C4L with fully connected layer,
        'v2' for Residual blocks with fully connected layer.
        :param name:  name for Meta model modules used for BMLNet initialization
        :param use_T: whether to use T layer for C4L neural networks
        :return: BMLNet object containing the dict of hyper parameters
        """

        self.param_dict['use_T'] = use_T
        self.param_dict['use_Warp'] = use_Warp
        self.param_dict['output_shape'] = dataset.train.dim_target
        if use_Warp:
            if 'model_loss_func' in model_args.keys():
                self.param_dict['model_loss_func'] = model_args['model_loss_func']
            else:
                self.param_dict['model_loss_func'] = utils.cross_entropy
            if 'outer_loss_func' in model_args.keys():
                self.param_dict['outer_loss_func'] = model_args['outer_loss_func']
            else:
                self.param_dict['outer_loss_func'] = utils.cross_entropy

        assert isinstance(dataset.train, (OmniglotMetaDataset, ImageNetMetaDataset)), \
            'The dataset does not match the model chosen for meta_learner'
        self.data_set = dataset
        assert meta_model.startswith('V'),\
            'The dataset does not match the model chosen for meta_learner, V1,V2,...or Vk'
        if self.method == 'MetaInit':
            if isinstance(dataset.train, OmniglotMetaDataset):
                meta_learner = getattr(bml_networks, '%s' % ('BOMLNetOmniglotMetaInit' + meta_model))(
                    _input=_input, dim_output=dataset.train.dim_target, name=name, use_T=use_T,use_Warp=use_Warp,**model_args)
            elif isinstance(dataset.train, ImageNetMetaDataset):
                meta_learner = getattr(bml_networks, '%s' % ('BOMLNetMiniMetaInit' + meta_model))(
                    _input=_input, dim_output=dataset.train.dim_target, name=name, use_T=use_T, use_Warp=use_Warp,**model_args)
        elif self.method == 'MetaRepr':
            if isinstance(dataset.train, OmniglotMetaDataset):
                meta_learner = getattr(bml_networks, '%s' % ('BOMLNetOmniglotMetaRepr' + meta_model))(
                    _input=_input, name=name, use_T=use_T, outer_method=self.outer_method, **model_args)
            elif isinstance(dataset.train, ImageNetMetaDataset):
                meta_learner = getattr(bml_networks, '%s' % ('BOMLNetMiniMetaRepr' + meta_model))(
                    _input=_input, name=name, use_T=use_T, outer_method=self.outer_method,**model_args)
        else:
            print('initialize method arguement, should be in list \an [MetaRepr,MetaInitl]')
            raise AssertionError
        self.param_dict['meta_model'] = meta_model
        self._meta_model = meta_learner

        return meta_learner

    def base_learner(self, _input, meta_learner, name='task_net',
                     weights_initializer=tf.zeros_initializer):
        """
        This method has to be called for every experiment and takes responsibility
        for defining task-specific modules and inner optimizers.
        :param _input: orginal input for neural network construction of task-specific module
        :param meta_learner: returned value of Meta_model function, which is a instance of BMLNet or its child classes
        :param name: name for Base model modules used for BMLNet initialization
        :param weights_initializer: initializer function for task_specific network, called by 'MetaRepr' method
        :return: task-specific model part
        """
        if self.method == 'MetaInit':
            if isinstance(self.data_set.train, OmniglotMetaDataset):
                base_learner = getattr(bml_networks, '%s' % ('BOMLNetOmniglotMetaInit' + self.param_dict['meta_model']))(
                    _input=_input, outer_param_dict=meta_learner.outer_param_dict, model_param_dict=meta_learner.model_param_dict,
                    dim_output=meta_learner.dims[-1], name=name, use_T=meta_learner.use_T, use_Warp =meta_learner.use_Warp)
            elif isinstance(self.data_set.train, ImageNetMetaDataset):
                base_learner = getattr(bml_networks, '%s' % ('BOMLNetMiniMetaInit' + self.param_dict['meta_model']))(
                    _input=_input, outer_param_dict=meta_learner.outer_param_dict, model_param_dict=meta_learner.model_param_dict,
                    dim_output=meta_learner.dims[-1], name=name, use_T=meta_learner.use_T, use_Warp=meta_learner.use_Warp)
        elif self.method == 'MetaRepr':
            base_learner = getattr(bml_networks, 'BOMLNetFeedForward')(
                _input=_input, dims=self.data_set.train.dim_target, output_weight_initializer=weights_initializer,
                name=name, use_T=meta_learner.use_T)
        else:
            print('initialize method arguement, should be in list \an [MetaRepr,MetaInitl]')
            raise AssertionError
        return base_learner

    def ll_problem(self, inner_objective, learning_rate, T, inner_objective_optimizer='SGD', outer_objective=None,
                   learn_lr=False, alpha_init=0.0, s=1.0, t=1.0, learn_alpha=False, learn_st=False,
                   learn_alpha_itr=False, var_list=None,
                   init_dynamics_dict=None, first_order=False, loss_func=utils.cross_entropy,
                   momentum=0.5,beta1=0.0, beta2=0.999,experiment=None, **inner_kargs):
        """
        After construction of neural networks, solutions to lower level problems should be regulated in LL_Problem.
        :param inner_objective: loss function for the inner optimization problem
        :param learning_rate: step size for inner loop optimization
        :param T: numbers of steps for inner gradient descent optimization
        :param inner_objective_optimizer: Optimizer type for the outer parameters, should be in list ['SGD','Momentum','Adam']
        :param outer_objective: loss function for the outer optimization problem, which need to be claimed in BDA agorithm
        :param learn_lr: BOOLEAN type, which determines whether to define learning rate as an outer parameter
        :param alpha_init: initial value of ratio of inner objective to outer objective in BDA algorithm
        :param s: coefficients of aggregation of outer objectives in BDA algorithm, default to be 1.0
        :param t: coefficients of aggregation of inner objectives in BDA algorithm, default to be 1.0
        :param learn_alpha: specify parameter for BDA algorithm to decide whether to initialize alpha as a hyper parameter
        :param learn_alpha_itr: parameter for BDA algorithm to specify whether to initialize alpha as a vector, of which
        every dimension's value is step-wise scale factor fot the optimization process
        :param learn_st: specify parameter for BDA algorithm to decide whether to initialize s and t as hyper parameters
        :param first_order: specific parameter to define whether to use implement first order MAML, default to be `FALSE`
        :param loss_func: specifying which type of loss function is used for the maml-based method, which should be
        consistent with the form to compute the inner objective
        :param momentum: specific parameter for Optimizer.BMLOptMomentum to set initial value of momentum
        :param beta1: specific parameter for Optimizer.BMLOptMomentum to set initial value of Adam
        :param beta2: specific parameter for Optimizer.BMLOptMomentum to set initial value of Adam
        :param regularization: whether to add regularization terms in the inner objective
        :param experiment: instance of Experiment to use in the Lower Level Problem, especifially needed in the
         `MetaInit` type of methods.
        :param var_list: optional list of variables (of the inner optimization problem)from
        :param init_dynamics_dict: optional dictrionary that defines Phi_0 (see `OptimizerDict.set_init_dynamics`)
        :param inner_kargs: optional arguments to pass to `py_bml.core.optimizer.minimize`
        :return: `OptimizerDict` from py_bml.core.
        """
        if self._learning_rate is None:
            if learn_lr:
                self.param_dict['learn_lr'] = learn_lr
                self._learning_rate = tf.Variable(name='lr', initial_value=learning_rate, dtype=tf.float32,
                                                  collections=extension.METAPARAMETERS_COLLECTIONS, trainable=False)
            else:
                self._learning_rate = tf.constant(learning_rate, name='lr')
            self.param_dict['learning_rate'] = self.learning_rate
        if self.io_opt is None:
            if self.inner_method == 'Simple' or inner_objective_optimizer == 'SGD':
                self.io_opt = getattr(bml_optimizer, '%s%s' % ('BOMLOpt', inner_objective_optimizer))(
                    learning_rate=self._learning_rate, name=inner_objective_optimizer)
            elif inner_objective_optimizer == 'Adam':
                self.io_opt = getattr(bml_optimizer, '%s%s' % ('BOMLOpt', 'Adam'))(
                    learning_rate=self._learning_rate, beta1=beta1,beta2=beta2, name=inner_objective_optimizer)
            elif inner_objective_optimizer == 'Momentum':
                self.io_opt = getattr(bml_optimizer, '%s%s' % ('BOMLOpt', 'Momentum'))(
                    learning_rate=self._learning_rate, momentum=momentum, name=inner_objective_optimizer)
            else:
                self.io_opt = getattr(bml_optimizer, '%s%s' % ('BOMLOpt', inner_objective_optimizer))(
                    learning_rate=self._learning_rate, name=inner_objective_optimizer)
        assert isinstance(self.io_opt, getattr(bml_optimizer, 'BOMLOpt')), 'Must use an optimizer that extends ' \
                                                                          'the class boml.optimizers' \
                                                                          'found {} instead'.format(type(self.io_opt))
        if self.method == 'MetaRepr':
            if self.inner_method == 'Aggr':
                assert outer_objective is not None, \
                    'BDA must have upper_level loss functions passed to lower-level problems optimization process'
                if not (('s' in self._param_dict.keys()) or ('t' in self._param_dict.keys())):
                    if learn_st:
                        s = extension.get_outerparameter('s', s)
                        t = extension.get_outerparameter('t', t)
                    else:
                        s = tf.constant(s, name='s')
                        t = tf.constant(t, name='t')
                    self._param_dict['s'] = s
                    self._param_dict['t'] = t
                if 'alpha' not in self._param_dict.keys():
                    if learn_alpha_itr:
                        alpha_vec = np.ones((1, T), dtype=np.float32) * alpha_init
                        alpha = extension.get_outerparameter(name='alpha', initializer=alpha_vec)
                        t_tensor = tf.placeholder(shape=(T, 1), dtype=tf.float32, name='t_tensor')
                    else:
                        '''
                        if self.outer_method == 'Darts' and learn_alpha:
                            print('In One-Stage RAD, alpha escapes the back propagation process, '
                                  'so it can not be optimized with OS-RAD method  ')
                            raise AssertionError
                        else:
                        '''
                        alpha = extension.get_outerparameter(initializer=alpha_init,
                                                   name='alpha') if learn_alpha else tf.constant(
                            alpha_init, name='alpha')
                        t_tensor = tf.placeholder(dtype=tf.float32, name='t_tensor')
                    self._param_dict['alpha'] = alpha
                    self._param_dict['t_tensor'] = t_tensor
        elif self.method == 'MetaInit':
            self._param_dict['first_order'] = first_order
        else:
            print('illegal initialization value for argument:method, should be in [meta_repr, maml]')
            raise AssertionError
        assert isinstance(experiment,
                          BMLExperiment), 'MetaInit based methods require specialized ' \
                                          'task model for each generated task,' \
                                          'please refer to basic instruction for modules of networks'
        self._param_dict['loss_func'] = loss_func
        self._param_dict['experiment'] = experiment

        if 'T' in self._param_dict.keys():
            assert self._param_dict['T'] == T, 'all probems are supposed to take same gradient descent steps, ' \
                                                 'which means T must be initialized same as before'
        else:
            self._param_dict['T'] = T

        inner_grad = self._inner_gradient.compute_gradients(bml_opt=self.io_opt,
                                                   loss_inner=inner_objective, loss_outer=outer_objective,
                                                            param_dict=self._param_dict, var_list=var_list,
                                                            **{inner_arg: inner_kargs[inner_arg] for inner_arg in
                                                               set(inner_kargs.keys()) - set(self.param_dict.keys())})
        if hasattr(inner_grad, 'objective'):
            inner_objective = inner_grad.objective
            if self._inner_objective is None:
                self._inner_objective = [inner_objective]
            else:
                self._inner_objective = tf.concat((self._inner_objective, [inner_objective]), axis=0)
        else:
            pass

        if init_dynamics_dict:
            inner_grad.set_init_dynamics(init_dynamics_dict)
        return inner_grad

    def ul_problem(self, outer_objective, meta_learning_rate,inner_grad, mlr_decay=1.e-5,
                   meta_param=None, outer_objective_optimizer='Adam', epsilon=1.0,
                   momentum=0.5, tolerance=lambda _k: 0.1 * (0.9 ** _k), global_step=None):
        """
        Set the outer optimization problem and the descent procedure for the optimization of the
        outer parameters. Can be called at least once for every call of inner_problem, passing the resulting
         `OptimizerDict`. It can be called multiple times with different objective, optimizers and hyper_list s.

        :param outer_objective: scalar tensor for the outer objective
        :param meta_learning_rate: step size for outer loop optimization
        :param inner_grad: Returned value of py_bml.BMLHOptimizer.LLProblem()
        :param meta_param: optional list of outer parameters and model parameters
        :param outer_objective_optimizer: Optimizer type for the outer parameters,
        should be in list ['SGD','Momentum','Adam']
        :param darts: BOOLEAN, specific parameters to define whether to implement 'DARTS' algorithm
        :param epsilon: Float, cofffecients to be used in DARTS algorithm
        :param momentum: specific parameters to be used to initialize 'Momentum' algorithm
        :param beta1: specific parameters to be used to initialize 'Adam'
        :param beta2: specific parameters to be used to initialize 'Adam'
        :param tolerance: specific function template for Implicit HG Algorithm
        :param global_step: optional global step. By default tries to use the last variable
        in the collection GLOBAL_STEP
        :return: itself
        """
        if self._meta_learning_rate is None:
            self._meta_learning_rate = tf.train.inverse_time_decay(meta_learning_rate, self.global_step, decay_steps=1.,
                                                                   decay_rate=mlr_decay)
        if self.oo_opt is None:
            if outer_objective_optimizer == 'Momentum':
                self.oo_opt = tf.train.MomentumOptimizer(learning_rate=self._meta_learning_rate,
                                                         momentum=momentum)
            elif outer_objective_optimizer == 'Adam':
                self.oo_opt = tf.train.AdamOptimizer(learning_rate=self._meta_learning_rate)
            elif outer_objective_optimizer == 'SGD':
                self.oo_opt = tf.train.GradientDescentOptimizer(learning_rate=self._meta_learning_rate)
            else:
                print('optimizer must be in the list as follows: [SGD, Adam, Momentum]')
                raise IndexError
        assert isinstance(self._outer_gradient, getattr(hyper_grads, 'BOMLOuterGrad')), \
            'Wrong name for inner method,should be in list \n [Reverse, Simple, Forward, Implicit]'
        if self.outer_method == 'Darts' and (not hasattr(self.outergradient, 'Epsilon')):
            assert self.param_dict['T'] == 1, 'Darts requires single gradient step to optimize task parameters'
            assert isinstance(self._outer_gradient, getattr(hyper_grads, 'BOMLOuterGradDarts')), \
                'Wrong name for outer method,should be in list [Darts]'
            setattr(self.outergradient, 'Epsilon', tf.cast(epsilon, tf.float32))
            setattr(self.outergradient, 'param_dict', self.param_dict)
        if self.outer_method == 'Implicit' and (not hasattr(self.outergradient, 'tolerance')):
            self.outergradient.set_tolerance(tolerance=tolerance)
        meta_param = self.outergradient.compute_gradients(outer_objective, inner_grad, meta_param=meta_param, param_dict =self.param_dict)
        self._o_optim_dict[self.oo_opt].update(meta_param)

        if global_step is not None:
            self._global_step = global_step
        return self

    def minimize(self, inner_objective, learning_rate,meta_learning_rate, T, inner_objective_optimizer='SGD', outer_objective=None,
                   learn_lr=False, alpha_init=0.0, s=1.0, t=1.0, learn_alpha=False, learn_st=False,
                   learn_alpha_itr=False, var_list=None,
                   init_dynamics_dict=None, first_order=False, loss_func=utils.cross_entropy,
                   momentum=0.5,beta1=0.0, beta2=0.999,experiment=None,
                   meta_param=None, outer_objective_optimizer='Adam', epsilon=1.0,
                 tolerance=lambda _k: 0.1 * (0.9 ** _k), global_step=None, **inner_kargs):
        """
        calling once 'inner_problem', 'outer_problem' and 'aggregate_all', and optionally set an initial dynamics.
        For more complex uses (like inner problems batching) use the methods separately. This can be used only when
        a single problem setting are expected for your problem definition.
        Returns method `BOMLOptimizer.run`, that runs one hyperiteration.
        """
        inner_grad = self.ll_problem(inner_objective=inner_objective,learning_rate=learning_rate,
                                     inner_objective_optimizer=inner_objective_optimizer,experiment=experiment,
                                     outer_objective=outer_objective, T=T, first_order=first_order,learn_lr=learn_lr,
                                     learn_alpha=learn_alpha, learn_st=learn_st, learn_alpha_itr=learn_alpha_itr,
                                     var_list=var_list,alpha_init=alpha_init,s=s,t=t,momentum=momentum,beta2=beta2,beta1=beta1,
                                     loss_func=loss_func, init_dynamics_dict=init_dynamics_dict, **inner_kargs)
        self.ul_problem(outer_objective=outer_objective,meta_learning_rate=meta_learning_rate,inner_grad=inner_grad,
                        outer_objective_optimizer=outer_objective_optimizer, meta_param=meta_param,
                        global_step=global_step,tolerance=tolerance, momentum=momentum,epsilon=epsilon)

        return

    def aggregate_all(self, aggregation_fn=None, gradient_clip=None):
        """
        To be called when no more dynamics or problems will be added, computes the updates
        for the outer parameters. This behave nicely with global_variables_initializer.

        :param aggregation_fn: Optional operation to aggregate multiple hypergradients (for the same hyperparameter),
                                by (default: reduce_mean)
        :param gradient_clip: Optional operation like normalizing to be applied to hypergradients before performing
                            a descent step (default: nothing).

        :return: the run method of this object.
        """
        if self._fin_hts is None:
            # in this way also far.optimizer can be used
            _maybe_first_arg = lambda _v: _v[0] if isinstance(_v, tuple) else _v

            def maybe_first_arg(_v):
                return _v[0] if isinstance(_v, tuple) else _v

            self._fin_hts = tf.group(*[maybe_first_arg(opt.apply_gradients(
                self.outergradient.hgrads_hvars(meta_param=hll, aggregation_fn=aggregation_fn,
                                                process_fn=gradient_clip)))
                for opt, hll in self._o_optim_dict.items()])
            if self._global_step:
                with tf.control_dependencies([self._fin_hts]):
                    self._fin_hts = self._global_step.assign_add(1).op
        else:
            raise ValueError('BOMLOptimizer.Aggregate_all has already been called on ' +
                             'this object, further calls have no effect')
        return self.run

    def run(self, inner_objective_feed_dicts=None, outer_objective_feed_dicts=None,
            initializer_feed_dict=None, session=None):
        """
        Run an hyper-iteration (i.e. train the model(s) and compute hypergradients) and updates the outer parameters.

        :param inner_objective_feed_dicts: an optional feed dictionary for the inner problem. Can be a function of
                                            step, which accounts for, e.g. stochastic gradient descent.
        :param outer_objective_feed_dicts: an optional feed dictionary for the outer optimization problem
                                            (passed to the evaluation of outer objective). Can be a function of
                                            hyper-iterations steps (i.e. global variable), which may account for, e.g.
                                            stochastic evaluation of outer objective.
        :param initializer_feed_dict:  an optional feed dictionary for the initialization of inner problems variables.
                                            Can be a function of
                                            hyper-iterations steps (i.e. global variable), which may account for, e.g.
                                            stochastic initialization.
        :param session: optional session
        """
        self._outer_gradient.apply_gradients(inner_objective_feed_dicts,
                                             outer_objective_feed_dicts,
                                             initializer_feed_dict, param_dict=self._param_dict,
                                             session=session, global_step=self._global_step)
        ss = session or tf.get_default_session()

        def _opt_fd():
            # e.g. hyper-learning rate is a placeholder
            _io_fd = utils.maybe_call(inner_objective_feed_dicts, utils.maybe_eval(self._global_step)) \
                if inner_objective_feed_dicts else {}
            _oo_fd = utils.maybe_call(outer_objective_feed_dicts, utils.maybe_eval(self._global_step)) \
                if outer_objective_feed_dicts else {}
            return utils.merge_dicts( _io_fd, _oo_fd)

        ss.run(self._hyperit, _opt_fd())

    @property
    def meta_model(self):
        '''
        :return: the created BMLNet object
        '''
        return self._meta_model

    @property
    def outergradient(self):
        """
        :return: the outergradient object underlying this wrapper.
        """
        return self._outer_gradient

    @property
    def learning_rate(self):
        """
        :return: the outergradient object underlying this wrapper.
        """
        return self._learning_rate

    @property
    def meta_learning_rate(self):
        """
        :return: the outergradient object underlying this wrapper.
        """
        return self._meta_learning_rate

    @property
    def method(self):
        """
        :return: the method for whole algorithm.
        """
        return self._method

    @property
    def global_step(self):
        """
        :return: globalstep used in whole graph.
        """
        return self._global_step

    @property
    def param_dict(self):
        """
        :return: dict that holds hyper_params used in the inner optimization process.
        """
        return self._param_dict

    @property
    def _hyperit(self):
        """
        iteration of minimization of outer objective(s), assuming the hyper-gradients are already computed.
        """
        assert self._fin_hts is not None, 'Must call BMLHOptimizer.Aggregate_all before performing optimization.'
        return self._fin_hts

    @property
    def inner_objectives(self):
        return self._outer_gradient.inner_objectives

