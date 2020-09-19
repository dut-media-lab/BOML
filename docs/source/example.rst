Simple Running Example
=======================
.. highlight:: sh

::

	from boml import utils
	from test_script.script_helper import *
	dataset = boml.load_data.meta_omniglot(std_num_classes=args.num_classes,
										   examples_train=args.examples_train,
										   examples_test=args.examples_test)
	ex = boml.BOMLExperiment(dataset)
	# build network structure and define hyperparameters
	boml_ho = boml.BOMLOptimizer(method="MetaRepr", inner_method="Simple", outer_method="Simple")
	meta_learner = boml_ho.meta_learner(_input=ex.x, dataset=dataset, meta_model="V1")
	ex.model = boml_ho.base_learner(_input=meta_learner.out, meta_learner=meta_learner)
	# define LL objectives and LL calculation process
	loss_inner = utils.cross_entropy(pred=ex.model.out, label=ex.y)
	inner_grad = boml_ho.ll_problem(inner_objective=loss_inner, learning_rate=args.lr,
									T=args.T, experiment=ex)
	# define UL objectives and UL calculation process
	loss_outer = utils.cross_entropy(pred=ex.model.re_forward(ex.x_).out, label=ex.y_)
	boml_ho.ul_problem(outer_objective=loss_outer,
					   meta_learning_rate=args.mlr, inner_grad=inner_grad,
					   meta_param=boml.extension.metaparameters()
	)
	# aggregate all the defined operations
	boml_ho.aggregate_all()
	# meta training step
	with utils.get_default_session():
		for itr in range(args.meta_train_iterations):
			train_batch = BatchQueueMock(dataset.train, 1,args.meta_batch_size，utils.get_rand_state())
			tr_fd, v_fd = feed_dicts(train_batch)
			boml_ho.run(tr_fd, v_fd)