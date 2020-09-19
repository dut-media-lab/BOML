Core Modules
=====================

.. highlight:: sh

1. load_data

   - Related: 
		- boml.load_data.meta_omniglot 
		- boml.load_data.meta_mini_imagenet 
		- boml.load_data.mnist 
		- ...
		
	:: 
	
		boml.meta_omniglot(
			folder=DATA_FOLDER, 
			std_num_classes=None, 
			examples_train=None, 
			examples_test=None, 
			one_hot_enc=True, 
			_rand=0, 
			n_splits=None)
		boml.meta_mini_imagenet(
		  folder=DATA_FOLDER, 
		  sub_folders=None, 
		  std_num_classes=None,
		  examples_train=None, 
		  examples_test=None, 
		  resize=84, 
		  one_hot_enc=True, 
		  load_all_images=True,
		  h5=False
		  )
	
	boml.load_data manages different datasets and generate batches of tasks for training and testing.

   - Args：
		- folder: str, root folder name. Use os module to modify the path to the datasets
		- std_num_classes: number of classes for N-way classification
		- examples_train: number of examples to be picked in each generated per classes for training (eg .1 shot, examples_train=1)
		- examples_test: number of examples to be picked in each generated per classes for testing
		- one_hot_enc: whether to adopt one hot encoding
		- _rand: random seed or RandomState for generate training, validation, testing meta-datasets split
		- n_splits: num classes per split
   - Usage:
	  :: 
	  
		  dataset = boml.meta_omniglot(
		  args.num_classes,
		  args.num_examples, 
		  args.examples_test
		  )
	  
   - Returns: an initialized instance of data loader 
2. Experiment

   - Aliases: 
		- boml.load_data.Experiment
		
	::
	
		boml.Experiment(
			dataset=None, 
			dtype=tf.float32
			)
	
	boml.Experiment manages inputs, outputs and task-specific parameters.
	
   - Args:
	  - dataset: initialized instance of load_data
	  - dtype: default tf.float32
   - Attributes:
	  - x: input placeholder of input for your defined lower level problem
	  - y: label placeholder of output for yourdefined lower level problem
	  - x\_:input placeholder of input for your defined upper level problem
	  - y\_:label placeholder of output for your defined upper level problem
	  - model: used to restore the task-specific model 
	  - errors: dictionary to restore defined loss functions of different levels 
	  - scores: dictionary to restore defined accuracies functions 
	  - optimizer: dictonary to restore optimized chosen for inner and outer loop optimization
   - Usage:
	::	
	
		ex = boml.Experiment(datasets = dataset)
		ex.errors['training'] = boml.utils.cross_entropy(
		pred=ex.model.out, 
		label=ex.y, 
		method='MetaRper')
		
		ex.scores['accuracy'] = tf.contrib.metrics.accuracy(
		tf.argmax(tf.nn.softmax(ex.model.out), 1),
		tf.argmax(ex.y, 1))
		
		ex.optimizer['apply_updates'], _ = boml.BOMLOptSGD(
		learning_rate=lr0
		).minimize(
		ex.errors['training'],
		var_list=ex.model.var_list
		)
		
   - Returns: an initialized instance of Experiment 

3. BOMLOptimizer

   - Aliases: 
		- boml.boml_optimizer.BOMLOptimizer
		
	::
	
		boml.BOMLOptimizer(
			Method=None, 
			inner_method=None, 
			outer_method=None, 
			truncate_iter=-1,
			experiments=[]
			)
	
	BOMLOptimizer is the main class in `boml`, which takes responsibility for the whole process of model construnction and back propagation. 
   
   - Args:
	  - Method: define basic method for following training process, it should be included in [`MetaInit`, `MetaRepr`], `MetaInit` type includes methods like `MAML`, `FOMAML`, `MT-net`, `WarpGrad`; `MetaRepr` type includes methods like `BA`, `RHG`, `TG`, `HOAG`, `DARTS`;
	  - inner_method: method chosen for solving LLproblem, including [`Trad` , `Simple`, `Aggr`], MetaRepr type choose either `Trad` for traditional optimization strategies or `Aggr` for Gradient Aggragation optimization. 'MetaInit' type should choose `Simple`, and set specific parameters for detailed method choices like FOMAML or MT-net.
	  - outer_method: method chosen for solving LLproblem, including [`Reverse` ,`Simple`, `DARTS`, `Implcit`], `MetaInit` type should choose `Simple`, and set specific parameters for detailed method choices like `FOMAML`
	  - truncate_iter: specific parameter for `Truncated Gradient` method, defining number of iterations to truncate in the Back propagation process
	  - experiments: list of Experiment objects that has already been initialized 
   
   - Usage:
	::
	
		ex = boml.Experiment(
		boml.meta_omniglot(5,1,15)
		)
		boml_ho = boml.BOMLOptimizer(
			Method='MetaRper', 
			inner_method='Simple', 
			outer_method='Simple',
			experiments=ex
			)
	
   - Utility Functions:
	  - learning_rate(): returns defined inner learning rate
	  - meta_learning_rate(): returns defined outer learning rate 
	  - Method: return defined method type 
	  - param_dict: return the dictionary that restores general parameters, like use_t,use_warp, output shape of defined model, learn_lr, s, t, alpha, first_order.
   
   - Returns: an initialized instance of BOMLOptimizer
   
