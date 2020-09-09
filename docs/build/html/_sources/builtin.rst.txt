Core Builtin Functions 
========================================

.. highlight:: sh
  
1. BOMLOptimizer.meta_learner:

   - Aliases: 
		- boml.boml_optimizer.BOMLOptimizer.meta_learner()
		
	::
	
		boml.boml_optimizer.BOMLOptimizer.meta_learner(
			_input, 
			dataset, 
			meta_model='V1', 
			name='Hyper_Net', 
			use_T=False, 
			use_Warp=False,
			**model_args
		)
	
   This method must be called once at first to build meta modules and initialize meta parameters and neural networks.
   
   - Args:
   
	  - _input: orginal input for neural network construction;
	  - dataset: which dataset to use for training and testing. It should be initialized before being passed into the function
	  - meta_model: model chosen for neural network construction, `V1` for C4L with fully connected layer,`V2` for Residual blocks with fully connected layer.
	  - name: name for Meta model modules used for BOMLNet initialization
	  - use_T: whether to use T layer for C4L neural networks
	  - use_Warp: whether to use Warp layer for C4L neural networks
	  - model_args: optional arguments to set specific parameters of neural networks.

2. BOMLOptimizer.base_learner:

   - Aliases: 
	  - boml.boml_optimizer.BOMLOptimizer.base_learner()
	  
	::
	
		boml.boml_optimizer.BOMLOptimizer.base_learner(
			_input, 
			meta_learner, name='Task_Net',
			weights_initializer=tf.zeros_initializer
		)
	
   This method has to be called for every experiment and takes responsibility for defining task-specific modules and inner optimizer.
   
   - Args:
   
	  - _input: orginal input for neural network construction of task-specific module;
	  - meta_learner: returned value of meta_learner function, which is a instance of BOMLNet or its child classes
	  - name: name for Base model modules used for BOMLNet initialization
	  - weights_initializer: initializer function for task_specific network, called by 'MetaRepr' method
   
   - Returns: task-specific model part

3. BOMLOptimizer.ll_problem:

	- Aliases: 
		- boml.boml_optimizer.BOMLOptimizer.ll_problem()
		
	::
	
		boml.boml_optimizer.BOMLOptimizer.ll_problem(
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
			  loss_func=utils.cross_entropy, 
			  momentum=0.5,
			  beta1=0.0,
			  beta2=0.999,
			  regularization=None, 
			  experiment=None, 
			  scalor=0.0, 
			  **inner_kargs
		)
	
   After construction of neural networks, solutions to lower level problems should be regulated in ll_problem.
   
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
	  - momentum: specific parameter for Optimizer.BOMLOptMomentum to set initial value of momentum
	  - regularization: whether to add regularization terms in the inner objective 
	  - experiment: instance of Experiment to use in the Lower Level Problem, especifially needed in the `MetaRper` type of method.
	  - var_list: optional list of variables (of the inner optimization problem)
	  - inner_kargs: optional arguments to pass to `boml.boml_optimizer.BOMLOptimizer.compute_gradients`
   
   - Returns: task-specific model part

4. BOMLOptimizer.ul_problem

   - Aliases:
   
	  - boml.boml_optimizer.BOMLOptimizer.ul_problem()
	  
		::
		
			boml.boml_optimizer.BOMLOptimizer.ul_Problem(
				outer_objective, 
				meta_learning_rate, 
				inner_grad,
				meta_param=None, 
				outer_objective_optimizer='Adam', 
				epsilon=1.0,
				momentum=0.5, 
				global_step=None
			)
		
	This method define upper level problems and choose optimizer to optimize meta parameters, which should be called afer ll_problem.
	
	- Args:
	
		- outer_objective: scalar tensor for the outer objective
		- meta_learning_rate: step size for outer loop optimization
		- inner_grad: Returned value of boml.BOMLOptimizer.LLProblem()
		- meta_param: optional list of outer parameters and model parameters
		- outer_objective_optimizer: Optimizer type for the outer parameters, should be in list [`SGD`,`Momentum`,`Adam`]
		- epsilon: Float, cofffecients to be used in DARTS algorithm
		- momentum: specific parameters to be used to initialize `Momentum` algorithm
	
	- Returns：meta_param list, used for debugging
	
5. aggregate_all:

   - Aliases: 
   
	   - boml.boml_optimizer.BOMLOptimizer.aggregate_all()
	   
	  ::
		  boml.boml_optimizer.BOMLOptimizer.aggregate_all(
			  aggregation_fn=None, 
			  gradient_clip=None
			  )

   - Args:
	  - aggregation_fn:Optional operation to aggregate multiple outer_gradients (for the same meta parameter),by (default: reduce_mean)
	  - gradient_clip: optional operation to clip the aggregated outer gradients
   
   - Returns: None
 
 Finally, aggregate_all has to be called to aggregate gradient of different tasks, and define operations to apply outer gradients and update meta parametes.

6. run:

   - Aliases: 
		- boml.boml_optimizer.BOMLOptimizer.run()
		
		::
		
			boml.boml_optimizer.BOMLOptimizer.run(
				inner_objective_feed_dicts=None,
				outer_objective_feed_dicts=None,
				session=None, 
				_skip_hyper_ts=False, 
				_only_hyper_ts=False, 
				callback=None
			)
	
   - Args:
	  - inner_objective_feed_dicts: an optional feed dictionary for the inner problem. Can be a function of step, which accounts for, e.g. stochastic gradient descent.
	  - outer_objective_feed_dicts: an optional feed dictionary for the outer optimization problem (passed to the evaluation of outer objective). Can be a function of hyper-iterations steps (i.e. global variable), which may account for, e.g. stochastic evaluation of outer objective.
	  - session: optional session
	  - callback: optional callback function of signature (step (int), feed_dictionary, `tf.Session`) -> None that are called after every forward iteration.
   
   - Returns: None
