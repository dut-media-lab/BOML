# Welcome to the documentation for PyBML 

## Contents <div0 id="a0"></div0>
1. [Introduction ](#a1)<br>
2. [Installation and requirements ](#a2)
3. [Quickly build your bilevel meta-learning model ](#a3)
    - [Core Modules ](#a31)
    - [Core Built-in functions of BMLHOptimizer ](#a32)
    - [Simple Training Example](#a33)
4. [Modification and Extension ](#a4)
5. [Author and liscense](#a5)

## Introduction <div1 id="a1"></div1>
PyBML is a python toolbox for bilevel meta learning in few shot learning field. Before reading the documentation, you could refer to [View on GitHub](https://github.com/liuyaohua918/pybml/edit/master/README.md) for a brief introduction about meta learning and PyBML. <br>
Here we provide detailed instruction to quickly get down to your research, test performance of popular algorithms and new ideas.

## Installation and requirements  <div2 id="a2"></div2>
PyBML implements various meta learning algorithms based on [TensorFlow](https://www.tensorflow.org/install/pip), which is one of the most popular macheine learning platform. Besides, [Numpy](https://numpy.org/install/) and basical image processing modules are required for  installation. <br>
We also provide [requirements.txt](https://github.com/liuyaohua918/pybml/requirements.txt) as reference for version control.
  ```
  1. Install from GitHub page：

  git clone https://github.com/liuyaohua918/pybml.git

  python setup.py install 

  pip install requirements.txt

  2. use pip instruction

  pip install py_bml
  ```
 
## Quickly build your bilevel meta-learning model <div3 id="a3"></div3>
  - Core Modules: <div3 id="a31"></div3>
    1. Dataloader
       - Related: 
            - pybml.Dataloader.meta_omniglot <br>
            - pybml.Dataloader.meta_mini_imagenet <br>
            - pybml.Dataloader.mnist <br>
        ```
        pybml.meta_omniglot(
            folder=DATA_FOLDER, 
            std_num_classes=None, 
            examples_train=None, 
            examples_test=None, 
            one_hot_enc=True, 
            _rand=0, 
            n_splits=None)
        pybml.meta_mini_imagenet(
          folder=DATA_FOLDER, 
          sub_folders=None, 
          std_num_classes=None,
          examples_train=None, 
          examples_test=None, 
          resize=84, 
          one_hot_enc=True, 
          load_all_images=True,
          h5=False):
        ```
        pybml.DataLoader manages different datasets and generate bathes of tasks for training and testing.
       - Args：<br>
            - folder: root folder name. Use os module to modify the path to the datasets<br>
            - std_num_classes: standard number of classes for N-way classification<br>
             - examples_train:standard number of examples to be picked in each generated per classes for training (eg .1 shot, examples_train=1)<br>
            - examples_test: standard number of examples to be picked in each generated per classes for testing
            - one_hot_enc: one hot encoding<br>
            - _rand: random seed or RandomState for generate training, validation, testing meta-datasets split<br>
            - n_splits: num classes per split<br>
       - Usage:
          ```
          dataset = py_bml.meta_omniglot(args.num_classes,
                                  (args.num_examples, args.examples_test))
          ```
       - Returns: an initialized instance of data loader 
    2. Experiment
       - Aliases: 
           - pybml.Dataloader.Experiment
        ```
        pybml.Experiment(
            dataset=None, 
            dtype=tf.float32)
        ```
        pybml.Experiment manages inputs, outputs and task-specific parameters.
       - Args:
          - dataset: initialized instance of Dataloader<br>
          - dtype: default tf.float32<br>
       - Attributes:<br>
          - x: input placeholder of input for your defined lower level problem<br>
          - y: label placeholder of output for yourdefined lower level problem<br>
          - x_:input placeholder of input for your defined upper level problem<br>
          - y_:label placeholder of output for your defined upper level problem<br>
          - model: used to restore the task-specific model <br>
          - errors: dictionary to restore defined loss functions of different levels<br> 
          - scores: dictionary to restore defined accuracies functions<br> 
          - optimizers: dictonary to restore optimized chosen for inner and outer loop optimization<br>
       - Usage:
        ```
        ex = pybml.Experiment(datasets = dataset)
        ex.errors['training'] = py_bml.utils.cross_entropy_loss(pred=ex.model.out, label=ex.y, method='HyperOptim')
        ex.scores['accuracy'] = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(ex.model.out), 1), tf.argmax(ex.y, 1))
        ex.optimizers['apply_updates'], _ = pybml.BMLOptSGD(learning_rate=lr0).minimize(ex.errors['training'],var_list=ex.model.var_list)
        ```
       - Returns: an initialized instance of Experiment 
  
    3. BMLHOptimizer 
       - Aliases: 
         - pybml.Core.BMLHOptimizer
        ```
        pybml.BMLHOptimizer(
            Method=None, 
            inner_method=None, 
            outer_method=None, 
            truncate_iter=-1,
            experiments=[]
            )
        ```
        BMLHOptimizer is the main class in `pybml`, which takes responsibility for the whole process of model construnction and back propagation. 
       - Args:
          - Method: define basic method for following training process, it should be included in [`HyperOptim`, `BilevelOptim`], `HyperOptim` type includes methods like `MAML`, `FOMAML`, `TNet`, `WarpGrad`; `BilevelOptim` type includes methods like `BDA`, `RHG`, `TRHG`, `Implicit HG`, `DARTS`;<br>
          - inner_method: method chosen for solving LLproblem, including [`Trad` ,`Simple`, `Aggr`], `BilevelOptim` type choose either `Trad` for traditional optimization strategies or `Aggr` for Gradient Aggragation optimization 'HyperOptim' type should choose `Simple`, and set specific parameters for detailed method choices like FOMAML or TNet.<br>
          - outer_method: method chosen for solving LLproblem, including [`Reverse` ,`Simple`, `Forward`, `Implcit`], `HyperOptim` type should choose Simple, and set specific parameters for detailed method choices like `FOMAML`
          - truncate_iter: specific parameter for `Truncated RHG` method, defining number of iterations to truncate in the Back propagation process<br>
          - experiments: list of experiment objects that has already been initialized <br>
        - Usage:
        ```
        ex = pybml.Experiment(pybml.meta_omniglot(5,1,15))
        pybml_ho = pybml.BMLHOptimizer(
            Method='HyperOptim', 
            inner_method='Simple', 
            outer_method='Simple',
            experiments=ex)
        ```
       - Utility functions:
          - learning_rate(): returns defined inner learning rate
          - meta_learning_rate(): returns defined outer learning rate 
          - Method: return defined method type 
          - param_dict: return the dictionary that restores general parameters, like use_T,use_Warp, output shape of defined model, learn_lr, s, t, alpha, first_order.
       - Returns: an initialized instance of BMLHOptimizer
  - Core Built-in functions of BMLHOptimizer: <div3 id="a32"></div3> 
    1. BMLHOptimizer.Meta_model:
       - Aliases: 
         - pybml.Core.BMLHOptimizer.Meta_model()
        ```
        pybml.Core.BMLHOptimizer.Meta_model(
            _input, 
            dataset, 
            meta_model='v1', 
            name='Hyper_Net', 
            use_T=False, 
            use_Warp=False,
            **model_args
        )
        ```
       This method must be called once at first to build meta modules and initialize meta parameters and neural networks.
       - Args:
          - _input: orginal input for neural network construction;
          - dataset: which dataset to use for training and testing. It should be initialized before being passed into the function
          - meta_model: model chosen for neural network construction, `v1` for C4L with fully connected layer,`v2` for Residual blocks with fully connected layer.
          - name: name for Meta model modules used for BMLNet initialization
          - use_T: whether to use T layer for C4L neural networks
    2. BMLHOptimizer.Base_model:
       - Aliases: 
          - pybml.Core.BMLHOptimizer.Base_model()
        ```
        pybml.Core.BMLHOptimizer.Base_model(
            _input, 
            meta_learner, name='Task_Net',
            weights_initializer=tf.zeros_initializer
        )
        ```
       This method has to be called for every experiment and takes responsibility for defining task-specific modules and inner optimizers.
       - Args:
          - _input: orginal input for neural network construction of task-specific module;
          - meta_learner: returned value of Meta_model function, which is a instance of BMLNet or its child classes
          - name: name for Base model modules used for BMLNet initialization
          - weights_initializer: initializer function for task_specific network, called by 'BilevelOptim' method
       - Returns: task-specific model part
    3. BMLHOptimizer.LL_problem:
       - Aliases: 
             - pybml.Core.BMLHOptimizer.LL_Problem()
        ```
        pybml.Core.BMLHOptimizer.LL_Problem(
              inner_objective,
              learning_rate, 
              T, 
              inner_objective_optimizer='SGD', 
              outer_objective=None,
              learn_lr=False, 
              alpha_init=0.0, 
              s=1.0, t=1.0, 
              learn_alpha=False, 
              learn_st=False,
              learn_alpha_itr=False, 
              var_list=None,
              init_dynamics_dict=None, 
              first_order=False, 
              loss_func=utils.cross_entropy_loss, 
              momentum=0.5,
              beta1=0.0,
              beta2=0.999,
              regularization=None, 
              experiment=None, 
              scalor=0.0, 
              **inner_kargs
        )
        ```
       After construction of neural networks, solutions to lower level problems should be regulated in LL_Problem.
       - Args:
          - inner_objective: loss function for the inner optimization problem
          - learning_rate: step size for inner loop optimization
          - T: numbers of steps for inner gradient descent optimization
          - inner_objective_optimizer: Optimizer type for the outer parameters, should be in list [`SGD`,`Momentum`,`Adam`]
          - outer_objective: loss function for the outer optimization problem, which need to be claimed in BDA agorithm
          - alpha_init: initial value of ratio of inner objective to outer objective in BDA algorithm
          - s,t: coefficients of aggregation of inner and outer objectives in BDA algorithm, default to be 1.0
          - learn_alpha: specify parameter for BDA algorithm to decide whether to initialize alpha as a hyper parameter
          - learn_alpha_itr: parameter for BDA algorithm to specify whether to initialize alpha as a vector, of which every dimension's value is step-wise scale factor fot the optimization process        
          - learn_st: specify parameter for BDA algorithm to decide whether to initialize s and t as hyper parameters
          - first_order: specific parameter to define whether to use implement first order MAML, default to be `FALSE`
          - loss_func: specifying which type of loss function is used for the maml-based method, which should be consistent with the form to compute the inner objective
          - momentum: specific parameter for Optimizer.BMLOptMomentum to set initial value of momentum
          - beta1, beta2: specific parameter for Optimizer.BMLOptMomentum to set initial value of Adam
          - regularization: whether to add regularization terms in the inner objective 
          - experiment: instance of Experiment to use in the Lower Level Problem, especifially needed in the `HyperOptim` type of method.
          - scalor: coefficient of regularization term in the objective function.
          - var_list: optional list of variables (of the inner optimization problem)from
          - init_dynamics_dict: optional dictrionary that defines Phi_0 (see `OptimizerDict.set_init_dynamics`)
          - inner_kargs: optional arguments to pass to `py_bml.core.optimizer.minimize`
       - Returns: task-specific model part
   
    4. BMLHOptimizer.UL_Problem
       - Aliases:
          - pybml.Core.BMLHOptimizer.UL_Problem()
            ```
            pybml.Core.BMLHOptimizer.UL_Problem(
                outer_objective, 
                meta_learning_rate, 
                inner_grad,
                meta_param=None, 
                outer_objective_optimizer='Adam', 
                Reptile=False, 
                Darts=False, 
                epsilon=1.0,
                beta1=0.9,beta2=0.999, 
                momentum=0.5, 
                global_step=None
            )
            ```
        This method define upper level problems and choose optimizers to optimize meta parameters, which should be called afer LL_Problem.
        - Args:
            - outer_objective: scalar tensor for the outer objective
            - meta_learning_rate: step size for outer loop optimization
            - inner_grad: Returned value of py_bml.BMLHOptimizer.LLProblem()
            - meta_param: optional list of outer parameters and model parameters
            - outer_objective_optimizer: Optimizer type for the outer parameters, should be in list [`SGD`,`Momentum`,`Adam`]
            - Reptile: BOOLEAN, specific parameters to define whether to implement `Reptile` algorithm
            - Darts: BOOLEAN, specific parameters to define whether to implement 'DARTS' algorithm
            - epsilon: Float, cofffecients to be used in DARTS algorithm
            - momentum: specific parameters to be used to initialize `Momentum` algorithm
            - beta1, beta2: specific parameters to be used to initialize `Adam`
            - global_step: optional global step. By default tries to use the last variable in the collection GLOBAL_STEP
        - Returns：meta_param list, used for debugging
    5. Aggregate_all:
       - Aliases: 
           - pybml.Core.BMLHOptimizer.Aggregate_all()
          ```
          pybml.Core.BMLHOptimizer.Aggregate_all(
              aggregation_fn=None, 
              gradient_clip=None
              )

          ```
       - Args:
          - aggregation_fn:Optional operation to aggregate multiple outer_gradients (for the same meta parameter),by (default: reduce_mean)
          - gradient_clip: optional operation to clip the aggregated outer gradients
       - Returns: None
     Finally, Aggregate_all has to be called to aggregate gradient of different tasks, and define operations to apply outer gradients and update meta parametes.
    6. run:
       - Aliases: 
          - pybml.Core.BMLHOptimizer.run()
        ```
        pybml.Core.BMLHOptimizer.run(
            inner_objective_feed_dicts=None,
            outer_objective_feed_dicts=None,
            train_batches=None,
            initializer_feed_dict=None, 
            session=None, 
            online=False,
            _skip_hyper_ts=False, 
            _only_hyper_ts=False, 
            callback=None
        )
        ```
       - Args:
          - inner_objective_feed_dicts: an optional feed dictionary for the inner problem. Can be a function of step, which accounts for, e.g. stochastic gradient descent.
          - outer_objective_feed_dicts: an optional feed dictionary for the outer optimization problem (passed to the evaluation of outer objective). Can be a function of hyper-iterations steps (i.e. global variable), which may account for, e.g. stochastic evaluation of outer objective.
          - train_batches: used for Reptile Algorithm, which needs to generates mini batches of images and labels during one training step
          - initializer_feed_dict:  an optional feed dictionary for the initialization of inner problems variables. Can be a function of hyper-iterations steps (i.e. global variable), which may account for, e.g. stochastic initialization.
          - session: optional session
          - online: default `False` if `True` performs the online version of the algorithms (i.e. does not reinitialize the state after at each run).
          - callback: optional callback function of signature (step (int), feed_dictionary, `tf.Session`) -> None that are called after every forward iteration.
       - Returns: None
  - Simple Running Example <div3 id="a33"></div3>
    ```
        from py_bml import utils
        from py_bml.script_helper import *
        dataset = py_bml.meta_omniglot(args.num_classes, (args.examples_train, args.examples_test))
        ex = py_bml.BMLExperiment(dataset)
        # build network structure and define hyperparameters
        pybml_ho = py_bml.BMLHOptimizer('HyperOptim', 'Aggr', 'Reverse')
        meta_learner = pybml_ho.Meta_learner(ex.x, dataset, 'v1', args.use_T)
        ex.model = pybml_ho.Base_learner(meta_learner.out, meta_learner)
        # define Lower-level problems
        loss_inner = utils.cross_entropy_loss(ex.model.out, ex.y)
        inner_grad = pybml_ho.LL_problem(loss_inner, args.lr, args.T, experiment=ex)
        # define Upper-level problems
        loss_outer = utils.cross_entropy_loss(ex.model.re_forward(ex.x_).out, ex.y_)
        pybml_ho.UL_problem(loss_outer, args.mlr, inner_grad, hyper_list=py_bml.extension.hyperparameters())
        pybml_ho.Aggregate_all()
        # meta training step
        with utils.get_default_session():
            for itr in range(args.meta_train_iterations):
                train_batch = BatchQueueMock(dataset.train, 1,args.meta_batch_size，utils.get_rand_state())
                tr_fd, v_fd = feed_dicts(train_batch)
                pybml_ho.run(tr_fd, v_fd)
    ``` 
    
## Modification and extension  <div4 id="a4"></div4>
  - Extensible Base Calsses and Modules
    1. BMLNet 
       - Aliases：
          - pybml.Networks.BMLNet
       - Methods to be overridden:
            - forward()：
                uses defined convolutional neural networks with initial input
            - re_forward(new_input): 
                reuses defined convolutional with new input and update the output results 
            - create_outer_parameters(): 
              this method creates parameters of upper level problems, and adds them to define collections called `METAPARAMETERS`
                - Args: 
                  - var_collections: collections to restore meta parameters created in the so called scope 
                - Returns: dictionary that indexes the outer parameters 
            - create_model_parameters(): 
                this method creates model parameters of upper level problems like `T layer` or `Warp Layer` , and adds them to define collections called `METAPARAMETERS`
       - Utility functions:
            - get_conv_weight(bmlnet, layer, initializer):
                - Args:
                  - bmlnet: initialized instance of BMLNet
                  - layer: int32, the layer-th weight of convolutional block to be created
                  - initializer: the tensorflow initializer used to initialize the filters 
              -Returns: created parameter
            - get_bias_weight(bmlnet, layer, initializer):
                - Args:
                  - bmlnet: initialized instance of BMLNet
                  - layer: int32, the layer-th bias of convolutional block to be created
                  - initializer: the tensorflow initializer used to initialize the bias
                - Returns: created parameter
            - get_identity(dim, name, conv=True):
                - Args:
                  - dim: the dimension of identity metrix
                  - name: name to initialize the metrix
                  - conv: BOOLEAN , whether initialize the metrix or initialize the real value, default to be True
                - Returns: the created parameter
            - conv_block(bmlnet, cweight, bweight):
               uses defined convolutional weight and bias with current ouput of bmlnet
                - Args:
                  - bmlnet: initialized instance of BMLNet
                  - cweight: parameter of convolutional filter
                  - bweight: parameter of bias for convolutional neural networks
            - conb_block_t(bmlnet, conv_weight, conv_bias, zweight):
               uses defined convolutional weight, bias, and weights of t layer  with current ouput of bmlnet
                - Args:
                  - bmlnet: initialized instance of BMLNet
                  - cweight: parameter of convolutional filter
                  - bweight: parameter of bias for convolutional neural networks
            - conv_block_warp(bmlnet, cweight, bweight, zweight, zbias):
              uses defined convolutional weight, bias and filters of warp layer  with current ouput of bmlnet
                - Args:
                  - bmlnet: initialized instance of BMLNet
                  - cweight: parameter of convolutional filter
                  - bweight: parameter of bias for convolutional neural networks
    2. BMLInnerGrad
       - Aliases:
          - pybml.LLProblem.BMLInnerGrad
       - Methods to be overridden:
           - compute_gradients(bml_opt, loss_inner, loss_outer=None,inner_method=None, param_dict=OrderedDict(), var_list=None, **inner_kargs):
        delivers equivalent functionality to the method called compute_gradients() in `tf.train.Optimizer`
            - Args:
              - bml_opt: instance of pybml.Optimizer.BMLOpt, which is automatically create by the method in `pybml.Core.BMLHOptimizer` 
              - loss_inner: inner objective, which could be passed by `pybml.Core.BMLHOptimizer.LL_Problem` or called directly.
              - loss_outer: outer objective,which could be passed automatically by `pybml.Core.BMLHOptimizer.LL_Problem`, or called directly 
              - param_dict: automatically passed by 'pybml.Core.BMLHOptimizer.LL_Problem'
              - var_list: list of lower level variables
              - inner_kargs: optional arguments, which are same as `tf.train.Optimizer`
            - Returns：self   
       - Utility functions:
          - apply_updates():
            Descent step, as returned by `tf.train.Optimizer.apply_gradients`.
          - initialization():
            a list of operations that return the values of the state variables for this learning dynamics after the execution of the initialization operation. If an initial dynamics is set, then it also executed.
          - state(): 
            A generator for all the state variables (optimized variables and possibly auxiliary variables) being optimized
    3. BMLOuterGrad
       - Aliases: 
         - pybml.UL_Problem.BMLOuterGrad
       - Methods to be overridden:
         - compute_gradients(outer_objective, bml_inner_grad, meta_param=None):
            - Args:
              - bml_inner_grad: OptimzerDict object resulting from the inner objective optimization.
              - outer_objective: A loss function for the outer parameters (scalar tensor)
              - meta_param: Optional list of outer parameters to consider. If not provided will get all variables in the hyperparameter collection in the current scope.
            - Returns: list of meta parameters involved in the computation
          - apply_gradients( inner_objective_feed_dicts=None, outer_objective_feed_dicts=None, initializer_feed_dict=None, param_dict=OrderedDict(), train_batches=None, experiments= [], global_step=None, session=None, online=False, callback=None)
            - Args:
              - inner_objective_feed_dicts: Optional feed dictionary for the inner objective
              - outer_objective_feed_dicts: Optional feed dictionary for the outer objective
                                              (note that this is not used in ForwardHG since hypergradients are not
                                              variables)
              - initializer_feed_dict: Optional feed dictionary for the inner objective
              - global_step: Optional global step for the optimization process
              - param_dict: dictionary of parameters passed by `pybml.Core.BMLHOptimizer`
              - train_batches: mini batches of data, needed when Reptile Algorithm are implemented
              - session: Optional session (otherwise will take the default session)
              - experiments: list of instances of `Experiment`, needed when Reptile Algorithm are implemented
              - online: Performs the computation of the outer gradient in the online (or "real time") mode. Note that `ReverseHG` and `ForwardHG` behave differently.
              - callback: callback funciton for the forward optimization
       - Utility functions:
         - hgrads_hvars(hyper_list=None, aggregation_fn=None, process_fn=None):
                Method for getting outergradient and outer parameters as required by apply_gradient methods from tensorflow optimizers.
                - Args：
                  - meta_param: Optional list of outer parameters to consider. If not provided will get all variables in the hyperparameter collection in the current scope.
                  - aggregation_fn: Optional operation to aggregate multiple hypergradients (for the same hyperparameter),
                                      by default reduce_mean
                  - process_fn: Optional operation like clipping to be applied.
         - initialization():
                 Returns groups of operation that initializes the variables in the computational graph
          - state():
            returns current state values of lower level variables 
    4. BMLOpt
       - Aliases: 
           - pybml.Optimizer.BMLOpt
       - Methods to be overridden:
           - minimize(loss_inner, var_list=None, global_step=None, gate_gradients=tf.train.Optimizer.GATE_OP,
               aggregation_method=None, colocate_gradients_with_ops=False, name=None, grad_loss=None):
               - Returns: an `bml_inner_grad` object relative to this minimization, same as `tf.train.Optimizer.minimize.`
       - Utility functions:
           - learning_rate():
                 - Returns: the step size of this optimizer
       - Utility Functions
           - get_dafault_session():
                get and return the default tensorflow session
       - BatchQueueMock:
        generates batches of taskes and feed them into corresponding placeholders.
  - Utility Modules:
      - get_default_session():
        gets and returns the default tensorflow session
      - BatchQueueMock():
          responsible for generates batches of taskes and feed them into corresponding placeholders.
      - cross_entropy_loss(pred, label, method):
        return loss function that matches different methods in [`BilevelOptim`,`HyperOptim`]
      - vectorize_all(var_list, name=None):
        Vectorize the variables in the list named var_list with the given name
      - remove_from_collectinon(key,*var_list):
        removes the variables in the var_list according to the given Graph key
      - set_gpu():
        set primary parameters of GPU configuration.
      
## Authors and license<div5 id="a5"></div5>
MIT License
MIT License

Copyright (c) 2020 Yaohua Liu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.


