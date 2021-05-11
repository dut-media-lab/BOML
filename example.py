from boml import utils
from test_script.script_helper import *

dataset = boml.load_data.meta_omniglot(
    std_num_classes=args.classes,
    examples_train=args.examples_train,
    examples_test=args.examples_test,
)
ex = boml.BOMLExperiment(dataset)
# build network structure and define hyperparameters
boml_ho = boml.BOMLOptimizer(
    method="MetaInit", inner_method="Simple", outer_method="Simple"
)
meta_learner = boml_ho.meta_learner(_input=ex.x, dataset=dataset, meta_model="V1")
ex.model = boml_ho.base_learner(_input=ex.x, meta_learner=meta_learner)
# define LL objectives and LL calculation process
loss_inner = utils.cross_entropy(pred=ex.model.out, label=ex.y)
accuracy = utils.classification_acc(pred=ex.model.out, label=ex.y)
inner_grad = boml_ho.ll_problem(
    inner_objective=loss_inner,
    learning_rate=args.lr,
    T=args.T,
    experiment=ex,
    var_list=ex.model.var_list,
)
# define UL objectives and UL calculation process
loss_outer = utils.cross_entropy(pred=ex.model.re_forward(ex.x_).out, label=ex.y_)
boml_ho.ul_problem(
    outer_objective=loss_outer,
    meta_learning_rate=args.meta_lr,
    inner_grad=inner_grad,
    meta_param=tf.get_collection(boml.extension.GraphKeys.METAPARAMETERS),
)
# aggregate all the defined operations
boml_ho.aggregate_all()
# meta training iteration
with tf.Session() as sess:
    tf.global_variables_initializer().run(session=sess)
    for itr in range(args.meta_train_iterations):
        # generate the feed_dict for calling run() everytime
        train_batch = BatchQueueMock(
            dataset.train, 1, args.meta_batch_size, utils.get_rand_state(1)
        )
        tr_fd, v_fd = utils.feed_dict(train_batch.get_single_batch(), ex)
        # meta training step
        boml_ho.run(tr_fd, v_fd)
        if itr % 100 == 0:
            loss_list=sess.run([loss_inner,loss_outer],utils.merge_dicts(tr_fd,v_fd))
            print('Iteration {}: Inner_loss {} , Outer_loss {}'.format(itr, loss_list[0],loss_list[1]))
