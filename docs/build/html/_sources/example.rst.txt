Simple Running Example
=======================
.. highlight:: sh

::

	from boml import utils
	from boml.script_helper import *
	dataset = boml.meta_omniglot(args.num_classes, (args.examples_train, args.examples_test))
	ex = boml.BOMLExperiment(dataset)
	# build network structure and define hyperparameters
	boml_ho = boml.BOMLOptimizer('HyperOptim', 'Aggr', 'Reverse')
	meta_learner = boml_ho.Meta_learner(ex.x, dataset, 'v1', args.use_T)
	ex.model = boml_ho.Base_learner(meta_learner.out, meta_learner)
	# define Lower-level problems
	loss_inner = utils.cross_entropy_loss(ex.model.out, ex.y)
	inner_grad = boml_ho.LL_problem(loss_inner, args.lr, args.T, experiment=ex)
	# define Upper-level problems
	loss_outer = utils.cross_entropy_loss(ex.model.re_forward(ex.x_).out, ex.y_)
	boml_ho.UL_problem(loss_outer, args.mlr, inner_grad, hyper_list=boml.extension.hyperparameters())
	boml_ho.Aggregate_all()
	# meta training step
	with utils.get_default_session():
		for itr in range(args.meta_train_iterations):
			train_batch = BatchQueueMock(dataset.train, 1,args.meta_batch_size，utils.get_rand_state())
			tr_fd, v_fd = feed_dicts(train_batch)
			boml_ho.run(tr_fd, v_fd)