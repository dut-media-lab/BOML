Extensible Modules
============================

.. highlight:: sh

- Extensible Base Calsses and Modules
	1. BOMLNet 
	
	   - Aliases：
	   
		  - boml.networks.BOMLNet
	   
	   - Methods to be overridden:
	   
			- forward()：
				It uses defined convolutional neural networks with initial input
			- re_forward(new_input): 
				It reuses defined convolutional with new input and update the output results 
			- create_outer_parameters(): 
				This method creates parameters of upper level problems, and adds them to define collections called `METAPARAMETERS`
				
				- Args: 
				  - var_collections: collections to restore meta parameters created in the so called scope 
				- Returns: dictionary that indexes the outer parameters 
				
			- create_model_parameters(): 
				
				This method creates model parameters of upper level problems like `T layer` or `Warp layer` , and adds them to define collections called `METAPARAMETERS`.
			
	   - Utility functions:
	   
			- get_conv_weight(boml_net, layer, initializer):
			
				- Args:
				  - boml_net: initialized instance of BOMLNet
				  - layer: int32, the layer-th weight of convolutional block to be created
				  - initializer: the tensorflow initializer used to initialize the filters 
			  
			  -Returns: created parameter
			- get_bias_weight(boml_net, layer, initializer):
			
				- Args:
				
				  - boml_net: initialized instance of BOMLNet
				  - layer: int32, the layer-th bias of convolutional block to be created
				  - initializer: the tensorflow initializer used to initialize the bias
				  
				- Returns: created parameter
				
			- get_identity(dim, name, conv=True):
			
				- Args:
				
				  - dim: the dimension of identity metrix
				  - name: name to initialize the metrix
				  - conv: BOOLEAN , whether initialize the metrix or initialize the real value, default to be True
				  
				- Returns: the created parameter
				
			- conv_block(boml_net, cweight, bweight):
			
			   It uses defined convolutional weight and bias with current ouput of boml_net
			   
				- Args:
				  - boml_net: initialized instance of BOMLNet
				  - cweight: parameter of convolutional filter
				  - bweight: parameter of bias for convolutional neural networks
				  
			- conb_block_t(boml_net, conv_weight, conv_bias, zweight):
			
			   uses defined convolutional weight, bias, and weights of t layer  with current ouput of boml_net
			   
				- Args:
				
				  - boml_net: initialized instance of BOMLNet
				  - conv_weight: parameter of convolution filter for convolutional neural networks
				  - conv_bias: parameter of bias for convolutional neural networks
				  - zweight: parameter of convolution filter for T-layer
				  
			- conv_block_warp(boml_net, cweight, bweight, zweight, zbias):
			
			  uses defined convolutional weight, bias and filters of warp layer  with current ouput of boml_net
			  
				- Args:
				
				  - boml_net: initialized instance of BOMLNet
				  - cweight: parameter of convolution filter for convolutional neural networks
				  - bweight: parameter of bias for convolutional neural networks
				  - zweight: parameter of convolution filter for Warp-layer
				  - zbias: parameter of bias for Warp-layer
				  
	2. BOMLInnerGrad

	   - Aliases:
	   
			- boml.LLProblem.BOMLInnerGrad
		  
	   - Methods to be overridden:
	   
			- compute_gradients( boml_opt, loss_inner, loss_outer=None, inner_method=None, param_dict=OrderedDict(), var_list=None, \*\*inner_kargs):
			
			The method delivers equivalent functionality to the method called compute_gradients() in `tf.train.Optimizer`.
			
				- Args:
				
				  - boml_opt: instance of boml.optimizer.BOMLOpt, which is automatically create by the method in `boml.boml_optimizer.BOMLOptimizer` 
				  - loss_inner: inner objective, which could be passed by `boml.boml_optimizer.BOMLOptimizer.ll_problem` or called directly.
				  - loss_outer: outer objective,which could be passed automatically by `boml.boml_optimizer.BOMLOptimizer.ll_problem`, or called directly 
				  - param_dict: automatically passed by 'boml.boml_optimizer.BOMLOptimizer.ll_problem'
				  - var_list: list of lower level variables
				  - inner_kargs: optional arguments, which are same as `tf.train.Optimizer`
				  
				- Returns：self
				
	   - Utility functions:
	   
		  - apply_updates():
		  
			Descent step, as returned by `tf.train.Optimizer.apply_gradients`.
			
		  - initialization():
		  
			 list of operations that return the values of the state variables for this learning dynamics after the execution of the initialization operation. If an initial dynamics is set, then it also executed.
		  
		  - state(): 
		  
			A generator for all the state variables (optimized variables and possibly auxiliary variables) being optimized.
			
	3. BOMLOuterGrad

	   - Aliases: 
	   
		 - boml.ul_problem.BOMLOuterGrad
		 
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
			  - global_step: Optional global step for the optimization process
			  - param_dict: dictionary of parameters passed by `boml.boml_optimizer.BOMLOptimizer`
			  - session: Optional session (otherwise will take the default session)
			  - experiments: list of instances of `Experiment`, needed when Reptile Algorithm are implemented

	   - Utility functions:
	   
			- hgrads_hvars( meta_param=None, aggregation_fn=None, gradient_clip=None):
		 
				Method for getting outergradient and outer parameters as required by apply_gradient methods from tensorflow optimizer.
				
				- Args：
				
				  - meta_param: Optional list of outer parameters to consider. If not provided will get all variables in the hyperparameter collection in the current scope.
				  - aggregation_fn: Optional operation to aggregate multiple hypergradients (for the same hyperparameter),
									  by default reduce_mean
				  - gradient_clip: Optional operation like clipping to be applied.
				  
		 - initialization():
		 
			Returns groups of operation that initializes the variables in the computational graph.
			
		  - state():
		  
			The method returns current state values of lower level variables.
			
	4. BOMLOpt

	   - Aliases: 
	   
		   - boml.optimizer.BOMLOpt
		   
	   - Methods to be overridden:
	   
		   - minimize(loss_inner, var_list=None, global_step=None, gate_gradients=tf.train.Optimizer.GATE_OP,
			   aggregation_method=None, colocate_gradients_with_ops=False, name=None, grad_loss=None):
			   
			   - Returns: an `bml_inner_grad` object relative to this minimization, same as `tf.train.Optimizer.minimize.`
			   
	   - Utility functions:
	   
		   - learning_rate():
		   
				 - Returns: the step size of this BOMLOptimizer
				 
	   - Utility Functions
	   
		   - get_dafault_session():
		   
				The method gets and returns the default tensorflow session
				

- Utility Modules:

	- get_default_session():

	The method gets and returns the default tensorflow session

	- BatchQueueMock():

	The class is responsible for generates batches of taskes and feed them into corresponding placeholders.
	  
	- cross_entropy(pred, label, method):

	It returns loss function that matches different methods in [MetaRepr,`MetaRper`]

	- vectorize_all(var_list, name=None):

	The method vectorize the variables in the list named var_list with the given name

	- remove_from_collectinon(key,*var_list):

	The method removes the variables in the var_list according to the given Graph key

	- set_gpu():

	The method sets primary parameters of GPU configuration.

