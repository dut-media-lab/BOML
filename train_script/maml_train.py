import os
import sys

sys.path.append('../')
os.environ['DATASETS_FOLDER'] = '../'
os.environ['EXPERIMENTS_FOLDER'] = '../'
import py_bml.extension

import py_bml as pybml
from train_script.script_helper import *
import numpy as np
import inspect, time
from shutil import copyfile

map_dict = {'omniglot': {'data_loader': pybml.meta_omniglot, 'model': pybml.BMLNetOmniglotHO_v1},
            'miniimagenet': {'data_loader': pybml.meta_mini_imagenet, 'model': pybml.BMLNetMiniHO_v1}}


def build(metasets, learn_lr, learn_alpha, learn_alpha_itr, learn_st, lr0, MBS, T, mlr0, mlr_decay,
          alpha_decay,
          batch_norm_before_classifier, weights_initializer,
          process_fn=None, scalor=0.0, regularization=None, alpha_itr=0.0, inner_method='Simple', outer_method='Simple',
          use_T=False, use_Warp=False, first_order=False,reptile=False):
    exs = [dl.BMLExperiment(metasets) for _ in range(MBS)]

    pybml_ho = pybml.BMLHOptimizer(Method='HyperOptim', inner_method=inner_method, outer_method=outer_method,experiments=exs)
    meta_model = pybml_ho.Meta_model(_input=exs[0].x, dataset=metasets, meta_model='v1',
                                        name='HyperRepr', use_T=use_T,use_Warp=use_Warp)

    for k, ex in enumerate(exs):
        ex.model = pybml_ho.Base_model(_input=ex.x, meta_learner=meta_model,
                                         name='Task_Net_%s' % k)
        ex.errors['training'] = py_bml.utils.cross_entropy_loss(pred=ex.model.out, label=ex.y, method='HyperOptim')
        ex.scores['accuracy'] = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(ex.model.out), 1),
                                                            tf.argmax(ex.y, 1))
        ex.optimizers['apply_updates'], _ = pybml.BMLOptSGD(learning_rate=lr0).minimize(ex.errors['training'],
                                                                                        var_list=ex.model.var_list)
        optim_dict = pybml_ho.LL_problem(inner_objective=ex.errors['training'], learning_rate=lr0,
                                         inner_objective_optimizer='SGD',
                                         T=T, experiment=ex, var_list=ex.model.var_list, learn_lr=learn_lr,
                                         first_order=first_order,regularization=regularization)
        ex.errors['validation'] = py_bml.utils.cross_entropy_loss(pred=ex.model.re_forward(ex.x_).out, label=ex.y_, method='HyperOptim')
        pybml_ho.UL_problem(outer_objective=ex.errors['validation'], meta_learning_rate=mlr0, inner_grad=optim_dict,
                            outer_objective_optimizer='SGD', reptile=reptile,
                            meta_param=tf.get_collection(py_bml.extension.GraphKeys.METAPARAMETERS))

    pybml_ho.Aggregate_all(gradient_clip=process_fn)
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=10)
    return exs, pybml_ho, saver


# training and testing function
def train_and_test(metasets, name_of_exp, inner_method, outer_method, use_T=False, use_Warp=False,
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

    exs, pybml_ho, saver = build(metasets, learn_lr, learn_alpha, learn_alpha_itr, learn_st, lr0,
                                 MBS, T, mlr0,
                                 mlr_decay, alpha_decay,
                                 batch_norm_before_classifier, weights_initializer, process_fn, scalor, regularization,
                                 alpha, inner_method, outer_method, use_T, use_Warp, first_order,reptile)

    sess = tf.Session(config=py_bml.utils.set_gpu())

    meta_train(exp_dir, metasets, exs, pybml_ho, saver, sess, n_test_episodes, MBS, seed, resume, T,
               n_meta_iterations, print_interval, save_interval)

    meta_test(exp_dir, metasets, exs, pybml_ho, saver, sess, args.classes, args.examples_train, lr0,
              n_test_episodes, MBS, seed, T, list(range(n_meta_iterations)))


# training and testing function
def build_and_test(metasets, exp_dir, inner_method, outer_method, use_T=False,use_Warp=False, first_order=False,reptile=False,
                   seed=None, lr0=0.04, T=5, MBS=4, learn_alpha=False, learn_alpha_itr=False,
                   weights_initializer=tf.zeros_initializer, batch_norm_before_classifier=False,
                   process_fn=None, n_test_episodes=600, iterations_to_test=list(range(100000)), scalor=0.0,
                   regularization=None, alpha=0.0, threshold=0.0):
    params = locals()
    print('params: {}'.format(params))

    mlr_decay = 1.e-5
    alpha_decay = 1.e-5
    mlr0 = 0.001
    learn_lr = False
    learn_st = False

    ''' Problem Setup '''
    np.random.seed(seed)
    tf.set_random_seed(seed)

    exs, pybml_ho, saver = build(metasets, learn_lr, learn_alpha, learn_alpha_itr, learn_st, lr0,
                                 MBS, T, mlr0, mlr_decay,
                                 alpha_decay, batch_norm_before_classifier, weights_initializer, process_fn, scalor,
                                 regularization,
                                 alpha, inner_method, outer_method, use_T,use_Warp, first_order,reptile)

    sess = tf.Session(config=py_bml.utils.GPU_CONFIG())

    meta_test_up_to_T(exp_dir, metasets, exs, pybml_ho, saver, sess, args.classes, args.examples_train, lr0,
                      n_test_episodes, MBS, seed, T, iterations_to_test)


def main():
    print(args.__dict__)
    try:
        if args.dataset == 'omniglot':
            metasets = map_dict[args.dataset]['data_loader'](std_num_classes=args.classes, examples_train=
            args.examples_train, examples_test=args.examples_test)
        elif args.dataset == 'miniimagenet':
            metasets = map_dict[args.dataset]['data_loader'](std_num_classes=args.classes, examples_train=
            args.examples_train, examples_test=args.examples_test)
    except KeyError:
        raise ValueError('dataset FLAG must be omniglot or miniimagenet')

    weights_initializer = tf.contrib.layers.xavier_initializer() if args.xavier else tf.zeros_initializer

    if args.clip_value > 0.:
        def process_fn(t):
            return tf.clip_by_value(t, -args.clip_value, args.clip_value)
    else:
        process_fn = None

    logdir = args.logdir + args.dataset

    if args.mode == 'train':
        train_and_test(metasets, exp_string, inner_method=args.inner_method,
                       outer_method=args.outer_method, use_T=args.use_T, first_order=args.first_order, logdir=logdir,
                       seed=args.seed, reptile=args.reptile,use_Warp=args.use_Warp,
                       lr0=args.lr, learn_lr=args.learn_lr, learn_alpha=args.learn_alpha,
                       learn_alpha_itr=args.learn_alpha_itr, learn_st=args.learn_st, mlr0=args.meta_lr,
                       mlr_decay=args.meta_lr_decay_rate, alpha_decay=args.alpha_decay, T=args.T,
                       resume=args.resume, MBS=args.meta_batch_size, n_meta_iterations=args.n_meta_iterations,
                       weights_initializer=weights_initializer, batch_norm_before_classifier=args.batch_norm,
                       process_fn=process_fn, save_interval=args.save_interval, print_interval=args.print_interval,
                       n_test_episodes=args.test_episodes, scalor=args.scalor, regularization=args.regularization,
                       alpha=args.alpha)

    elif args.mode == 'test':
        build_and_test(metasets, args.exp_dir, inner_method=args.inner_method,
                       outer_method=args.outer_method, use_T=args.use_T, use_Warp=args.use_Warp,
                       first_order=args.first_order, seed=args.seed, lr0=args.lr,reptile=args.reptile,
                       T=args.T, MBS=args.meta_batch_size, weights_initializer=weights_initializer,
                       learn_alpha=args.learn_alpha, learn_alpha_itr=args.learn_alpha_itr,
                       batch_norm_before_classifier=args.batch_norm, process_fn=process_fn,
                       n_test_episodes=args.test_episodes, iterations_to_test=args.iterations_to_test,
                       scalor=args.scalor, regularization=args.regularization, alpha=args.alpha)


if __name__ == "__main__":
    main()
