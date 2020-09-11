import os
import sys

sys.path.append('../')
os.environ['DATASETS_FOLDER'] = '../'
os.environ['EXPERIMENTS_FOLDER'] = '../'
import boml.extension

import boml as boml
from test_script.script_helper import *
import numpy as np
import inspect, time
from shutil import copyfile

map_dict = {'omniglot': {'data_loader': boml.meta_omniglot, 'model': boml.BOMLNetOmniglotMetaInitV1},
            'miniimagenet': {'data_loader': boml.meta_mini_imagenet, 'model': boml.BOMLNetMiniMetaInitV1}}


def build(metasets, learn_lr, lr0, MBS, T, mlr0,mlr_decay=1.e-5, process_fn=None, method='MetaInit',inner_method='Simple', outer_method='Simple',
          use_T=False, use_Warp=False, first_order=False):

    exs = [dl.BOMLExperiment(metasets) for _ in range(MBS)]

    boml_ho = boml.BOMLOptimizer(method=method, inner_method=inner_method, outer_method=outer_method,experiments=exs)
    meta_model = boml_ho.meta_learner(_input=exs[0].x, dataset=metasets, meta_model='V1',
                                        name='HyperRepr', use_T=use_T,use_Warp=use_Warp)

    for k, ex in enumerate(exs):
        ex.model = boml_ho.base_learner(_input=ex.x, meta_learner=meta_model,
                                         name='Task_Net_%s' % k)
        ex.errors['training'] = boml.utils.cross_entropy(pred=ex.model.out, label=ex.y, method=method)
        ex.scores['accuracy'] = boml.utils.classification_acc(pred=ex.model.out,label=ex.y)
        ex.optimizers['apply_updates'], _ = boml.BOMLOptSGD(learning_rate=lr0).minimize(ex.errors['training'],
                                                                                        var_list=ex.model.var_list)
        optim_dict = boml_ho.ll_problem(inner_objective=ex.errors['training'], learning_rate=lr0,
                                         inner_objective_optimizer=args.inner_opt,
                                         T=T, experiment=ex, var_list=ex.model.var_list, learn_lr=learn_lr,
                                         first_order=first_order)
        ex.errors['validation'] = boml.utils.cross_entropy(pred=ex.model.re_forward(ex.x_).out, label=ex.y_, method=method)
        boml_ho.ul_problem(outer_objective=ex.errors['validation'], meta_learning_rate=mlr0, inner_grad=optim_dict,
                            outer_objective_optimizer=args.outer_opt,mlr_decay=mlr_decay,
                            meta_param=tf.get_collection(boml.extension.GraphKeys.METAPARAMETERS))
    meta_learner = boml_ho.meta_model
    meta_learning_rate = boml_ho.meta_learning_rate
    apply_updates = boml_ho.outergradient.apply_updates
    inner_objectives= boml_ho.inner_objectives
    iteration =boml_ho.innergradient.iteration
    print(boml_ho.io_opt)
    print(boml_ho.oo_opt)
    boml_ho.aggregate_all(gradient_clip=process_fn)
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=10)
    return exs, boml_ho, saver


# training and testing function
def train_and_test(metasets, name_of_exp,method, inner_method, outer_method, use_T=False, use_Warp=False,
                   first_order=False, reptile=False, logdir='logs/', seed=None, lr0=0.04, learn_lr=False,
                   learn_alpha=False, learn_alpha_itr=False, learn_st=False,
                   mlr0=0.001, mlr_decay=1.e-5, alpha_decay=1.e-5, T=5, resume=True, MBS=4, n_meta_iterations=5000,
                   weights_initializer=tf.zeros_initializer,
                   batch_norm_before_classifier=False, process_fn=None, save_interval=5000, print_interval=5000,
                   n_test_episodes=1000, scalor=0.0, regularization=None, alpha=0.0, threshold=0.0):
    params = locals()
    print('params: {}'.format(params))

    ''' Problem Setup '''
    np.random.seed(seed)
    tf.set_random_seed(seed)

    exp_dir = logdir + '/' + name_of_exp
    print('\nExperiment directory:', exp_dir + '...')
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    executing_file_path = inspect.getfile(inspect.currentframe())
    print('copying {} into {}'.format(executing_file_path, exp_dir))
    copyfile(executing_file_path, os.path.join(exp_dir, executing_file_path.split('/')[-1]))

    exs, boml_ho, saver = build(metasets, learn_lr, lr0,MBS, T, mlr0,mlr_decay, process_fn,
                                 method, inner_method, outer_method, use_T, use_Warp, first_order)

    sess = tf.Session(config=boml.utils.set_gpu())

    meta_train(exp_dir, metasets, exs, boml_ho, saver, sess, n_test_episodes, MBS, seed, resume, T,
               n_meta_iterations, print_interval, save_interval)

    meta_test(exp_dir, metasets, exs, boml_ho, saver, sess, args.classes, args.examples_train, lr0,
              n_test_episodes, MBS, seed, T, list(range(n_meta_iterations)))


# training and testing function
def build_and_test(metasets, exp_dir,method, inner_method, outer_method, use_T=False,use_Warp=False, first_order=False,
                   seed=None, lr0=0.04, T=5, MBS=4,
                   process_fn=None, n_test_episodes=600, iterations_to_test=list(range(100000))):
    params = locals()
    print('params: {}'.format(params))

    mlr_decay = 1.e-5
    alpha_decay = 1.e-5
    mlr0 = 0.001
    learn_lr = False

    ''' Problem Setup '''
    np.random.seed(seed)
    tf.set_random_seed(seed)

    exs, pybml_ho, saver = build(metasets, learn_lr, lr0, MBS, T, mlr0,
                                 mlr_decay,process_fn,method, inner_method, outer_method, use_T,use_Warp, first_order)

    sess = tf.Session(config=boml.utils.set_gpu())

    meta_test_up_to_T(exp_dir, metasets, exs, pybml_ho, saver, sess, args.classes, args.examples_train, lr0,
                      n_test_episodes, MBS, seed, T, iterations_to_test)


def main():
    print(args.__dict__)
    metasets = map_dict[args.dataset]['data_loader'](std_num_classes=args.classes, examples_train=
    args.examples_train, examples_test=args.examples_test)

    if args.clip_value > 0.:
        def process_fn(t):
            return tf.clip_by_value(t, -args.clip_value, args.clip_value)
    else:
        process_fn = None

    logdir = args.logdir + args.dataset

    if args.mode == 'train':
        train_and_test(metasets, exp_string,method=args.method, inner_method=args.inner_method,
                       outer_method=args.outer_method, use_T=args.use_T, first_order=args.first_order, logdir=logdir,
                       seed=args.seed,use_Warp=args.use_Warp,
                       lr0=args.lr, learn_lr=args.learn_lr, mlr0=args.meta_lr, T=args.T,
                       resume=args.resume, MBS=args.meta_batch_size, n_meta_iterations=args.n_meta_iterations,
                       process_fn=process_fn, save_interval=args.save_interval, print_interval=args.print_interval,
                       n_test_episodes=args.test_episodes)

    elif args.mode == 'test':
        build_and_test(metasets, exp_dir=args.expdir,method=args.method, inner_method=args.inner_method,
                       outer_method=args.outer_method, use_T=args.use_T, use_Warp=args.use_Warp,
                       first_order=args.first_order, seed=args.seed, lr0=args.lr,
                       T=args.T, MBS=args.meta_batch_size, process_fn=process_fn,
                       n_test_episodes=args.test_episodes, iterations_to_test=args.iterations_to_test)

if __name__ == "__main__":
    main()
