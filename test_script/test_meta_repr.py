import matplotlib

matplotlib.use('Agg')
import sys
import inspect, os
sys.path.append('../')
os.environ['DATASETS_FOLDER'] = '../'
os.environ['EXPERIMENTS_FOLDER'] = '../'
import boml.extension
from test_script.script_helper import *

# from hr_resnet import hr_res_net_tcml_v1_builder, hr_res_net_tcml_Omniglot_builder
from shutil import copyfile
from boml.setup_model import BOMLNetOmniglotMetaReprV1, BOMLNetMiniMetaReprV1
import boml as boml

import numpy as np
import tensorflow as tf

dl.DATASET_FOLDER = 'datasets'

map_dict = {'omniglot': {'data_loader': dl.datasets.load_full_dataset.meta_omniglot, 'model': BOMLNetOmniglotMetaReprV1},
            'miniimagenet': {'data_loader': dl.datasets.load_full_dataset.meta_mini_imagenet, 'model': BOMLNetMiniMetaReprV1}}


def build(metasets, learn_lr, learn_alpha, learn_alpha_itr, learn_st, lr0, MBS, T, mlr0, mlr_decay,weights_initializer,
          process_fn=None, alpha_itr=0.0, method=None, inner_method=None, outer_method=None,
          use_T=False,use_Warp=True, truncate_iter=-1):
    exs = [dl.BOMLExperiment(metasets) for _ in range(MBS)]
    boml_ho = boml.BOMLOptimizer(method=method, inner_method=inner_method, outer_method=outer_method,
                                   truncate_iter=truncate_iter, experiments=exs)

    hyper_repr_model = boml_ho.meta_learner(_input=exs[0].x, dataset=metasets, meta_model='V1', name=method,
                                             use_T=use_T,use_Warp=use_Warp)

    for k, ex in enumerate(exs):
        repr_out = hyper_repr_model.re_forward(ex.x).out
        repr_out_val = hyper_repr_model.re_forward(ex.x_).out
        ex.model = boml_ho.base_learner(_input=repr_out, meta_learner=hyper_repr_model,
                                         weights_initializer=weights_initializer,
                                         name='Classifier_%s' % k)

        ex.errors['training'] = boml.utils.cross_entropy(pred=ex.model.out, label=ex.y, method=method)
        ex.errors['validation'] = boml.utils.cross_entropy(label=ex.y_, pred=ex.model.re_forward(
            repr_out_val).out, method=method)
        ex.scores['accuracy'] = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ex.y, 1), tf.argmax(ex.model.out, 1)),
                                                       tf.float32), name='accuracy')
        inner_objective = ex.errors['training']
        ex.optimizers['apply_updates'], _ = boml.BOMLOptSGD(learning_rate=lr0).minimize(ex.errors['training'],
                                                                                         var_list=ex.model.var_list)
        optim_dict = boml_ho.ll_problem(inner_objective=inner_objective, learning_rate=lr0,
                                        inner_objective_optimizer=args.inner_opt, outer_objective=ex.errors['validation'],
                                        alpha_init=alpha_itr, s=1.0, t=1.0, T=T,experiment=ex,learn_lr=learn_lr,
                                        learn_alpha_itr=learn_alpha_itr, learn_alpha=learn_alpha, learn_st=learn_st,
                                        var_list=ex.model.var_list)

        boml_ho.ul_problem(outer_objective=ex.errors['validation'], inner_grad=optim_dict,
                           outer_objective_optimizer=args.outer_opt, meta_learning_rate=mlr0,mlr_decay=mlr_decay,
                           meta_param=tf.get_collection(boml.extension.GraphKeys.METAPARAMETERS))
    meta_learning_rate = boml_ho.meta_learning_rate
    apply_updates = boml_ho.outergradient.apply_updates
    inner_objectives = boml_ho.inner_objectives
    iteration = boml_ho.innergradient.iteration
    print(boml_ho.io_opt)
    print(boml_ho.oo_opt)
    print(boml_ho.io_opt.learning_rate)
    print(boml_ho.io_opt.learning_rate_tensor)
    print(boml_ho.io_opt.optimizer_params_tensor)
    boml_ho.aggregate_all(gradient_clip=process_fn)
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=10)
    return exs, boml_ho, saver


# training and testing function
def train_and_test(metasets, name_of_exp, method, inner_method, outer_method, use_T=False,use_Warp=False,
                   truncate_iter=-1, logdir='logs/', seed=None, lr0=0.04, learn_lr=False,
                   learn_alpha=False, learn_alpha_itr=False, learn_st=False,
                   mlr0=0.001, mlr_decay=1.e-5, T=5, resume=True, MBS=4, n_meta_iterations=5000,
                   weights_initializer=tf.zeros_initializer,process_fn=None, save_interval=5000, print_interval=5000,
                   n_test_episodes=1000, alpha=0.0):
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

    exs, boml_ho, saver = build(metasets, learn_lr, learn_alpha, learn_alpha_itr, learn_st, lr0,
                                MBS, T, mlr0,mlr_decay, weights_initializer, process_fn,
                                alpha, method, inner_method, outer_method, use_T,use_Warp, truncate_iter)

    sess = tf.Session(config=boml.utils.set_gpu())

    meta_train(exp_dir, metasets, exs, boml_ho, saver, sess, n_test_episodes, MBS, seed, resume, T,
               n_meta_iterations, print_interval, save_interval)

    meta_test(exp_dir, metasets, exs, boml_ho, saver, sess, args.classes, args.examples_train, lr0,
              n_test_episodes, MBS, seed, T, list(range(n_meta_iterations)))


# training and testing function
def build_and_test(metasets, exp_dir, method, inner_method, outer_method, use_T=False,use_Warp=False, truncate_iter=-1,
                   seed=None, lr0=0.04, T=5, MBS=4, learn_alpha=False, learn_alpha_itr=False,
                   weights_initializer=tf.zeros_initializer, process_fn=None,
                   n_test_episodes=600, iterations_to_test=list(range(100000)), alpha=0.0, darts=False):
    params = locals()
    print('params: {}'.format(params))
    mlr_decay = 1.e-5
    mlr0 = 0.001
    learn_lr = False
    learn_st = False

    ''' Problem Setup '''
    np.random.seed(seed)
    tf.set_random_seed(seed)

    exs, boml_ho, saver = build(metasets, learn_lr, learn_alpha, learn_alpha_itr, learn_st, lr0,
                                 MBS, T, mlr0, mlr_decay, weights_initializer, process_fn,
                                 alpha, method, inner_method, outer_method, use_T,use_Warp, truncate_iter)

    sess = tf.Session(config=boml.utils.set_gpu())

    meta_test_up_to_T(exp_dir, metasets, exs, boml_ho, saver, sess, args.classes, args.examples_train, lr0,
                      n_test_episodes, MBS, seed, T, iterations_to_test)


def main():
    print(args.__dict__)

    metasets = map_dict[args.dataset]['data_loader'](std_num_classes=args.classes, examples_train=
    args.examples_train, examples_test=args.examples_test)


    weights_initializer = tf.contrib.layers.xavier_initializer() if args.xavier else tf.zeros_initializer

    if args.clip_value > 0.:
        def process_fn(t):
            return tf.clip_by_value(t, -args.clip_value, args.clip_value)
    else:
        process_fn = None

    logdir = args.logdir + args.dataset

    if args.mode == 'train':
        train_and_test(metasets, exp_string, method=args.method, inner_method=args.inner_method,
                       outer_method=args.outer_method, use_T=args.use_T, truncate_iter=args.truncate_iter,
                       logdir=logdir, seed=args.seed, use_Warp=args.use_Warp,
                       lr0=args.lr, learn_lr=args.learn_lr, learn_alpha=args.learn_alpha,
                       learn_alpha_itr=args.learn_alpha_itr, learn_st=args.learn_st, mlr0=args.meta_lr,
                       mlr_decay=args.meta_lr_decay_rate, T=args.T,
                       resume=args.resume, MBS=args.meta_batch_size, n_meta_iterations=args.n_meta_iterations,
                       weights_initializer=weights_initializer,
                       process_fn=process_fn, save_interval=args.save_interval, print_interval=args.print_interval,
                       n_test_episodes=args.test_episodes,
                       alpha=args.alpha)

    elif args.mode == 'test':
        build_and_test(metasets, exp_dir=args.expdir, method=args.method, inner_method=args.inner_method,
                       outer_method=args.outer_method, use_T=args.use_T,use_Warp=args.use_Warp, truncate_iter=args.truncate_iter,
                       seed=args.seed, lr0=args.lr, T=args.T, MBS=args.meta_batch_size, weights_initializer=weights_initializer,
                       learn_alpha=args.learn_alpha, learn_alpha_itr=args.learn_alpha_itr,process_fn=process_fn,
                       n_test_episodes=args.test_episodes, iterations_to_test=args.iterations_to_test,
                       alpha=args.alpha, darts=args.darts)


if __name__ == "__main__":
    main()
