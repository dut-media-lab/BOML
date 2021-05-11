import time
import boml
import boml.load_data as dl
import argparse
import pickle
import os
import tensorflow as tf
import numpy as np
from threading import Thread
from boml import utils
import json
parser = argparse.ArgumentParser()

parser.add_argument(
    "-m",
    "--mode",
    type=str,
    default="train",
    metavar="STRING",
    help="mode, can be train or test",
)

parser.add_argument(
    '--name_of_args_json_file'
    , type=str,
    default="None")

parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    default="omniglot",
    metavar="STRING",
    help="omniglot or miniimagenet.",
)
parser.add_argument(
    "-nc",
    "--classes",
    type=int,
    default=5,
    metavar="NUMBER",
    help="number of classes used in classification (c for  c-way classification).",
)
parser.add_argument(
    "-etr",
    "--examples_train",
    type=int,
    default=1,
    metavar="NUMBER",
    help="number of examples used for inner gradient update (k for k-shot learning).",
)
parser.add_argument(
    "-etes",
    "--examples_test",
    type=int,
    default=15,
    metavar="NUMBER",
    help="number of examples used for test sets",
)

# Training options
parser.add_argument(
    "-s",
    "--seed",
    type=int,
    default=0,
    metavar="NUMBER",
    help="seed for random number generators",
)
parser.add_argument(
    "-mbs",
    "--meta_batch_size",
    type=int,
    default=2,
    metavar="NUMBER",
    help="number of tasks sampled per meta-update",
)
parser.add_argument(
    "-mti",
    "--meta_train_iterations",
    type=int,
    default=2000,
    metavar="NUMBER",
    help="number of metatraining iterations.",
)
parser.add_argument(
    "-T",
    "--T",
    type=int,
    default=3,
    metavar="NUMBER",
    help="number of inner updates during training.",
)
parser.add_argument(
    "-xi",
    "--xavier",
    type=bool,
    default=False,
    metavar="BOOLEAN",
    help="FFNN weights initializer",
)
parser.add_argument(
    "-bn",
    "--batch-norm",
    type=bool,
    default=False,
    metavar="BOOLEAN",
    help="Use batch normalization before classifier",
)
parser.add_argument(
    "-mlr",
    "--meta_lr",
    type=float,
    default=0.001,
    metavar="NUMBER",
    help="starting meta learning rate",
)
parser.add_argument(
    "-mlrdr",
    "--meta-lr-decay-rate",
    type=float,
    default=1.0e-5,
    metavar="NUMBER",
    help="meta lr  inverse time decay rate",
)

parser.add_argument(
    "-cv",
    "--clip_value",
    type=float,
    default=0.0,
    metavar="NUMBER",
    help="meta gradient clip value (0. for no clipping)",
)
parser.add_argument(
    "-lr",
    "--lr",
    type=float,
    default=0.1,
    metavar="NUMBER",
    help="starting learning rate",
)

parser.add_argument(
    "-tr_ir",
    "--truncate_iter",
    type=int,
    default=-1,
    metavar="NUMBER",
    help="truncated iterations ",
)
parser.add_argument(
    "-alpha_decay",
    "--alpha_decay",
    type=float,
    default=1.0e-5,
    metavar="NUMBER",
    help="alpha decay rate",
)
parser.add_argument(
    "-lrl",
    "--learn_lr",
    type=bool,
    default=False,
    metavar="BOOLEAN",
    help="True if learning rate is an hyperparameter",
)
parser.add_argument(
    "-lrst",
    "--learn_st",
    type=bool,
    default=False,
    metavar="BOOLEAN",
    help="True if s and t are outer parameters",
)
parser.add_argument(
    "-lralpha",
    "--learn_alpha",
    type=bool,
    default=False,
    metavar="BOOLEAN",
    help="True if alpha is an hyperparameter",
)
parser.add_argument(
    "-learn_alpha_itr",
    "--learn_alpha_itr",
    type=bool,
    default=False,
    metavar="BOOLEAN",
    help="learn alpha iteration wise",
)

parser.add_argument(
    "-regularization",
    "--regularization",
    type=str,
    default=None,
    metavar="STRING",
    help="L1 or L2 or no regularization measure to apply",
)
parser.add_argument(
    "-alpha",
    "--alpha",
    type=float,
    default=0.0,
    metavar="NUMBER",
    help="factor for controlling the ratio of gradients",
)

parser.add_argument(
    "-bs",
    "--bda_s",
    type=float,
    default=1.0,
    metavar="NUMBER",
    help="factor for controlling the ratio of UL objective",
)


parser.add_argument(
    "-bt",
    "--bda_t",
    type=float,
    default=1.0,
    metavar="NUMBER",
    help="factor for controlling the ratio of LL objective",
)

parser.add_argument(
    "-md",
    "--method",
    type=str,
    default="MetaFeat",
    metavar="STRING",
    help="choose which method to use,[MetaFeat,MetaInit]",
)
parser.add_argument(
    "-i_d",
    "--inner_method",
    type=str,
    default="Trad",
    metavar="STRING",
    help="choose which method to use,[Trad, Aggr,Simple]",
)
parser.add_argument(
    "-o_d",
    "--outer_method",
    type=str,
    default="Reverse",
    metavar="STRING",
    help="choose which method to use,[Reverse,Implicit,Forward,Simple]",
)
parser.add_argument(
    "-u_T",
    "--use_t",
    type=bool,
    default=False,
    metavar="BOOLEAN",
    help="whether to use T-Net",
)
parser.add_argument(
    "-u_W",
    "--use_warp",
    type=bool,
    default=False,
    metavar="BOOLEAN",
    help="whether to use Warp layer to implement WarpGrad",
)
parser.add_argument(
    "-fo",
    "--first_order",
    type=bool,
    default=False,
    metavar="BOOLEAN",
    help="whether to implement FOMAML, short for First Order MAML",
)
parser.add_argument(
    "-re",
    "--reptile",
    type=bool,
    default=False,
    metavar="BOOLEAN",
    help="whether to implement Reptile method",
)

parser.add_argument(
    "-io",
    "--inner_opt",
    type=str,
    default="SGD",
    metavar="STRING",
    help="the typer of inner optimizer, which should be listed in [SGD,Adam,Momentum]",
)
parser.add_argument(
    "-oo",
    "--outer_opt",
    type=str,
    default="Adam",
    metavar="STRING",
    help="the typer of outer optimizer, which should be listed in [SGD,Adam,Momentum]",
)


# Logging, saving, and testing options
parser.add_argument(
    "-log",
    "--log",
    type=bool,
    default=False,
    metavar="BOOLEAN",
    help="if false, do not log summaries, for debugging code.",
)
parser.add_argument(
    "-ld",
    "--logdir",
    type=str,
    default="logs/",
    metavar="STRING",
    help="directory for summaries and checkpoints.",
)
parser.add_argument(
    "-res",
    "--resume",
    type=bool,
    default=True,
    metavar="BOOLEAN",
    help="resume training if there is a model available",
)
parser.add_argument(
    "-pi",
    "--print-interval",
    type=int,
    default=1,
    metavar="NUMBER",
    help="number of meta-train iterations before print",
)
parser.add_argument(
    "-si",
    "--save_interval",
    type=int,
    default=1,
    metavar="NUMBER",
    help="number of meta-train iterations before save",
)
parser.add_argument(
    "-te",
    "--test_episodes",
    type=int,
    default=600,
    metavar="NUMBER",
    help="number of episodes for testing",
)


# Testing options (put parser.mode = 'test')
parser.add_argument(
    "-exd",
    "--expdir",
    type=str,
    default=None,
    metavar="STRING",
    help="directory of the experiment model files",
)
parser.add_argument(
    "-itt",
    "--iterations_to_test",
    type=str,
    default=[40000],
    metavar="STRING",
    help='meta_iteration to test (model file must be in "exp_dir")',
)
parser.add_argument(
    "-Notes", "--Notes", type=str, default="Notes", help="Something important"
)

class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)

def extract_args_from_json(json_file_path, args_dict):
    summary_filename = json_file_path
    with open(summary_filename) as f:
        summary_dict = json.load(fp=f)

    for key in summary_dict.keys():
        if "continue_from" in key:
            pass
        elif "gpu_to_use" in key:
            pass
        else:
            args_dict[key] = summary_dict[key]

    return args_dict


args = parser.parse_args()
args_dict = vars(args)
if args.name_of_args_json_file is not "None":
    args_dict = extract_args_from_json(args.name_of_args_json_file, args_dict)

for key in list(args_dict.keys()):

    if str(args_dict[key]).lower() == "true":
        args_dict[key] = True
    elif str(args_dict[key]).lower() == "false":
        args_dict[key] = False

    if key == "dataset_path":
        args_dict[key] = os.path.join(os.environ['DATASET_DIR'], args_dict[key])
        print(key, os.path.join(os.environ['DATASET_DIR'], args_dict[key]))

    print(key, args_dict[key], type(args_dict[key]))

args = Bunch(args_dict)


exp_string = (
    str(args.classes)
    + "way_"
    + str(args.examples_train)
    + "shot_"
    + str(args.meta_batch_size)
    + "mbs"
    + str(args.T)
    + "T"
    + str(args.method)
    + "Method"
    + str(args.inner_method)
    + "inner_method"
    + str(args.outer_method)
    + "outer_method"
    + str(args.meta_lr)
    + "meta_lr"
    + str(args.lr)
    + "lr"
    + str(args.Notes)
    + "Notes"
)


def meta_train(
    exp_dir,
    metasets,
    exs,
    pybml_ho,
    saver,
    sess,
    n_test_episodes,
    MBS,
    seed,
    resume,
    T,
    n_meta_iterations,
    print_interval,
    save_interval,
):
    # use workers to fill the batches queues (is it worth it?)

    result_path = os.path.join(exp_dir, "results.pickle")
    tf.global_variables_initializer().run(session=sess)
    n_test_batches = n_test_episodes // MBS
    rand = dl.get_rand_state(seed)

    results = {
        "train_train": {"mean": [], "std": []},
        "train_test": {"mean": [], "std": []},
        "test_test": {"mean": [], "std": []},
        "valid_test": {"mean": [], "std": []},
        "outer_losses": {"mean": [], "std": []},
        "learning_rate": [],
        "iterations": [],
        "episodes": [],
        "time": [],
        "alpha": [],
    }

    resume_itr = 0
    if resume:
        model_file = tf.train.latest_checkpoint(exp_dir)
        if model_file:
            print("Restoring results from " + result_path)
            results = load_obj(result_path)

            ind1 = model_file.index("model")
            resume_itr = int(model_file[ind1 + 5 :]) + 1
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    """ Meta-Train """
    train_batches = BatchQueueMock(metasets.train, 1, MBS, rand)
    valid_batches = BatchQueueMock(metasets.validation, n_test_batches, MBS, rand)
    test_batches = BatchQueueMock(metasets.test, n_test_batches, MBS, rand)

    start_time = time.time()
    print(
        "\nIteration quantities: train_train acc, train_test acc, valid_test, acc"
        " test_test acc mean(std) over %d episodes" % n_test_episodes
    )
    with sess.as_default():
        inner_losses = []
        for meta_it in range(resume_itr, n_meta_iterations):
            tr_fd, v_fd = utils.feed_dicts(train_batches.get_all_batches()[0], exs)
            pybml_ho.run(tr_fd, v_fd)

            duration = time.time() - start_time

            results["time"].append(duration)
            outer_losses = []
            for _, ex in enumerate(exs):
                outer_losses.append(
                    sess.run(
                        ex.errors["validation"], boml.utils.merge_dicts(tr_fd, v_fd)
                    )
                )
            outer_losses_moments = (np.mean(outer_losses), np.std(outer_losses))
            results["outer_losses"]["mean"].append(outer_losses_moments[0])
            results["outer_losses"]["std"].append(outer_losses_moments[1])

            if meta_it % print_interval == 0 or meta_it == n_meta_iterations - 1:
                results["iterations"].append(meta_it)
                results["episodes"].append(meta_it * MBS)
                if "alpha" in pybml_ho.param_dict.keys():
                    alpha_moment = pybml_ho.param_dict["alpha"].eval()
                    print("alpha_itr" + str(meta_it) + ": ", alpha_moment)
                    results["alpha"].append(alpha_moment)
                if "s" in pybml_ho.param_dict.keys():
                    s = sess.run(["s:0"])[0]
                    print("s: {}".format(s))
                if "t" in pybml_ho.param_dict.keys():
                    t = sess.run(["t:0"])[0]
                    print("t: {}".format(t))

                train_result = accuracy_on(train_batches, exs, pybml_ho, sess, T)
                test_result = accuracy_on(test_batches, exs, pybml_ho, sess, T)
                valid_result = accuracy_on(valid_batches, exs, pybml_ho, sess, T)
                train_train = (np.mean(train_result[0]), np.std(train_result[0]))
                train_test = (np.mean(train_result[1]), np.std(train_result[1]))
                valid_test = (np.mean(valid_result[1]), np.std(valid_result[1]))
                test_test = (np.mean(test_result[1]), np.std(test_result[1]))

                results["train_train"]["mean"].append(train_train[0])
                results["train_test"]["mean"].append(train_test[0])
                results["valid_test"]["mean"].append(valid_test[0])
                results["test_test"]["mean"].append(test_test[0])

                results["train_train"]["std"].append(train_train[1])
                results["train_test"]["std"].append(train_test[1])
                results["valid_test"]["std"].append(valid_test[1])
                results["test_test"]["std"].append(test_test[1])

                results["inner_losses"] = inner_losses

                print("mean outer losses: {}".format(outer_losses_moments[0]))

                print(
                    "it %d, ep %d (%.5fs): %.5f, %.5f, %.5f, %.5f"
                    % (
                        meta_it,
                        meta_it * MBS,
                        duration,
                        train_train[0],
                        train_test[0],
                        valid_test[0],
                        test_test[0],
                    )
                )

                lr = sess.run(["lr:0"])[0]
                print("lr: {}".format(lr))

                # do_plot(logdir, results)

            if meta_it % save_interval == 0 or meta_it == n_meta_iterations - 1:
                saver.save(sess, exp_dir + "/model" + str(meta_it))
                save_obj(result_path, results)

            start_time = time.time()

        return results


def meta_test(
    exp_dir,
    metasets,
    exs,
    pybml_ho,
    saver,
    sess,
    c_way,
    k_shot,
    lr,
    n_test_episodes,
    MBS,
    seed,
    T,
    iterations=list(range(10000)),
):
    meta_test_str = (
        str(c_way)
        + "way_"
        + str(k_shot)
        + "shot_"
        + str(T)
        + "T"
        + str(lr)
        + "lr"
        + str(n_test_episodes)
        + "ep"
    )

    n_test_batches = n_test_episodes // MBS
    rand = dl.get_rand_state(seed)

    valid_batches = BatchQueueMock(metasets.validation, n_test_batches, MBS, rand)
    test_batches = BatchQueueMock(metasets.test, n_test_batches, MBS, rand)

    print("\nMeta-testing {} (over {} eps)...".format(meta_test_str, n_test_episodes))

    test_results = {
        "test_test": {"mean": [], "std": []},
        "valid_test": {"mean": [], "std": []},
        "cp_numbers": [],
        "time": [],
        "n_test_episodes": n_test_episodes,
        "episodes": [],
        "iterations": [],
    }

    test_result_path = os.path.join(exp_dir, meta_test_str + "_results.pickle")

    start_time = time.time()
    for i in iterations:
        model_file = os.path.join(exp_dir, "model" + str(i))
        if tf.train.checkpoint_exists(model_file):
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

            test_results["iterations"].append(i)
            test_results["episodes"].append(i * MBS)

            valid_result = accuracy_on(valid_batches, exs, pybml_ho, sess, T)
            test_result = accuracy_on(test_batches, exs, pybml_ho, sess, T)

            duration = time.time() - start_time

            valid_test = (np.mean(valid_result[1]), np.std(valid_result[1]))
            test_test = (np.mean(test_result[1]), np.std(test_result[1]))

            test_results["time"].append(duration)

            test_results["valid_test"]["mean"].append(valid_test[0])
            test_results["test_test"]["mean"].append(test_test[0])

            test_results["valid_test"]["std"].append(valid_test[1])
            test_results["test_test"]["std"].append(test_test[1])

            print(
                "valid-test_test acc (%d meta_it)(%.2fs): %.3f (%.3f),  %.3f (%.3f)"
                % (
                    i,
                    duration,
                    valid_test[0],
                    valid_test[1],
                    test_test[0],
                    test_test[1],
                )
            )

            save_obj(test_result_path, test_results)

    return test_results


def meta_test_up_to_T(
    exp_dir,
    metasets,
    exs,
    pybml_ho,
    saver,
    sess,
    c_way,
    k_shot,
    lr,
    n_test_episodes,
    MBS,
    seed,
    T,
    iterations=list(range(10000)),
):
    meta_test_str = (
        str(c_way)
        + "way_"
        + str(k_shot)
        + "shot_"
        + str(lr)
        + "lr"
        + str(n_test_episodes)
        + "ep"
    )

    n_test_batches = n_test_episodes // MBS
    rand = dl.get_rand_state(seed)

    valid_batches = BatchQueueMock(metasets.validation, n_test_batches, MBS, rand)
    test_batches = BatchQueueMock(metasets.test, n_test_batches, MBS, rand)
    train_batches = BatchQueueMock(metasets.train, n_test_batches, MBS, rand)

    print("\nMeta-testing {} (over {} eps)...".format(meta_test_str, n_test_episodes))

    test_results = {
        "valid_test": [],
        "test_test": [],
        "train_test": [],
        "time": [],
        "n_test_episodes": n_test_episodes,
        "episodes": [],
        "iterations": [],
    }

    test_result_path = os.path.join(exp_dir, meta_test_str + "noTrain_results.pickle")

    start_time = time.time()
    for i in iterations:
        model_file = os.path.join(exp_dir, "model" + str(i))
        if tf.train.checkpoint_exists(model_file):
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

            test_results["iterations"].append(i)
            test_results["episodes"].append(i * MBS)

            valid_result = accuracy_on_up_to_T(valid_batches, exs, pybml_ho, sess, T)
            test_result = accuracy_on_up_to_T(test_batches, exs, pybml_ho, sess, T)
            train_result = accuracy_on_up_to_T(train_batches, exs, pybml_ho, sess, T)

            duration = time.time() - start_time

            test_results["time"].append(duration)

            for t in range(T):

                valid_test = (np.mean(valid_result[1][t]), np.std(valid_result[1][t]))
                test_test = (np.mean(test_result[1][t]), np.std(test_result[1][t]))
                train_test = (np.mean(train_result[1][t]), np.std(train_result[1][t]))

                if t >= len(test_results["valid_test"]):
                    test_results["valid_test"].append({"mean": [], "std": []})
                    test_results["test_test"].append({"mean": [], "std": []})
                    test_results["train_test"].append({"mean": [], "std": []})

                test_results["valid_test"][t]["mean"].append(valid_test[0])
                test_results["test_test"][t]["mean"].append(test_test[0])
                test_results["train_test"][t]["mean"].append(train_test[0])

                test_results["valid_test"][t]["std"].append(valid_test[1])
                test_results["test_test"][t]["std"].append(test_test[1])
                test_results["train_test"][t]["std"].append(train_test[1])

                print(
                    "valid-test_test acc T=%d (%d meta_it)(%.2fs): %.4f (%.4f), %.4f (%.4f),"
                    "  %.4f (%.4f)"
                    % (
                        t + 1,
                        i,
                        duration,
                        train_test[0],
                        train_test[1],
                        valid_test[0],
                        valid_test[1],
                        test_test[0],
                        test_test[1],
                    )
                )

                # print('valid-test_test acc T=%d (%d meta_it)(%.2fs): %.4f (%.4f),'
                #      '  %.4f (%.4f)' % (t+1, i, duration, valid_test[0], valid_test[1],
                #                         test_test[0], test_test[1]))

            save_obj(test_result_path, test_results)

    return test_results


def batch_producer(metadataset, batch_queue, n_batches, batch_size, rand=0):
    while True:
        batch_queue.put([d for d in metadataset.generate(n_batches, batch_size, rand)])


def start_batch_makers(
    number_of_workers, metadataset, batch_queue, n_batches, batch_size, rand=0
):
    for w in range(number_of_workers):
        worker = Thread(
            target=batch_producer,
            args=(metadataset, batch_queue, n_batches, batch_size, rand),
        )
        worker.setDaemon(True)
        worker.start()


# Class for debugging purposes for multi-thread issues
class BatchQueueMock:
    def __init__(self, metadataset, n_batches, batch_size, rand):
        self.metadataset = metadataset
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.rand = rand

    def get_all_batches(self):
        return [
            d
            for d in self.metadataset.generate(
                self.n_batches, self.batch_size, self.rand
            )
        ]

    def get_single_batch(self):
        return [d for d in self.metadataset.generate(self.n_batches, 1, self.rand)]


def save_obj(file_path, obj):
    with open(file_path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_obj(file_path):
    with open(file_path, "rb") as handle:
        b = pickle.load(handle)
    return b


""" Useful Functions """


def just_train_on_dataset(dat, exs, pybml_ho, sess, T):
    train_fd, valid_fd = utils.feed_dicts(dat, exs)
    # print('train_feed:', train_fd)  # DEBUG
    sess.run(pybml_ho.outergradient.initialization)
    tr_acc, v_acc = [], []
    for ex in exs:
        [
            sess.run(
                ex.optimizers["apply_updates"],
                feed_dict={ex.x: train_fd[ex.x], ex.y: train_fd[ex.y]},
            )
            for _ in range(T)
        ]
        tr_acc.append(
            sess.run(
                ex.scores["accuracy"],
                feed_dict={ex.x: train_fd[ex.x], ex.y: train_fd[ex.y]},
            )
        )
        v_acc.append(
            sess.run(
                ex.scores["accuracy"],
                feed_dict={ex.x: valid_fd[ex.x_], ex.y: valid_fd[ex.y_]},
            )
        )
    return tr_acc, v_acc


def accuracy_on(batch_queue, exs, pybml_ho, sess, T):
    tr_acc, v_acc = [], []
    for d in batch_queue.get_all_batches():
        result = just_train_on_dataset(d, exs, pybml_ho, sess, T)
        tr_acc.extend(result[0])
        v_acc.extend(result[1])
    return tr_acc, v_acc


def just_train_on_dataset_up_to_T(dat, exs, pybml_ho, sess, T):
    train_fd, valid_fd = utils.feed_dicts(dat, exs)
    # print('train_feed:', train_fd)  # DEBUG
    sess.run(pybml_ho.outergradient.initialization)
    tr_acc, v_acc = [[] for _ in range(T)], [[] for _ in range(T)]
    for ex in exs:
        # ex.model.initialize(session=sess)
        for t in range(T):
            sess.run(
                ex.optimizers["apply_updates"],
                feed_dict={ex.x: train_fd[ex.x], ex.y: train_fd[ex.y]},
            )
            tr_acc[t].append(
                sess.run(
                    ex.scores["accuracy"],
                    feed_dict={ex.x: train_fd[ex.x], ex.y: train_fd[ex.y]},
                )
            )
            v_acc[t].append(
                sess.run(
                    ex.scores["accuracy"],
                    feed_dict={ex.x: valid_fd[ex.x_], ex.y: valid_fd[ex.y_]},
                )
            )
    return tr_acc, v_acc


def accuracy_on_up_to_T(batch_queue, exs, pybml_ho, sess, T):
    tr_acc, v_acc = [[] for _ in range(T)], [[] for _ in range(T)]
    for d in batch_queue.get_all_batches():
        result = just_train_on_dataset_up_to_T(d, exs, pybml_ho, sess, T)
        [tr_acc[T].extend(r) for T, r in enumerate(result[0])]
        [v_acc[T].extend(r) for T, r in enumerate(result[1])]

    return tr_acc, v_acc


def get_default_session():

    return tf.get_default_session()


def get_rand_state(rand):
    """
    Utility methods for getting a `RandomState` object.

    :param rand: rand can be None (new State will be generated),
                    np.random.RandomState (it will be returned) or an integer (will be treated as seed).

    :return: a `RandomState` object
    """
    if isinstance(rand, np.random.RandomState):
        return rand
    elif isinstance(rand, (int, np.ndarray, list)) or rand is None:
        return np.random.RandomState(rand)
    else:
        raise ValueError("parameter rand {} has wrong type".format(rand))


def mean_std_ci(measures, mul=1.0, tex=False):
    """
    Computes mean, standard deviation and 95% half-confidence interval for a list of measures.

    :param measures: list
    :param mul: optional multiplication coefficient (e.g. for percentage)
    :param tex: if True returns mean +- half_conf_interval for latex
    :return: a list or a string in latex
    """
    half_int = lambda _m: 1.96 * np.std(_m) / np.sqrt(len(_m) - 1)
    measures = np.array(measures) * mul
    ms = np.mean(measures), np.std(measures), half_int(measures)
    return ms if not tex else r"${:.2f} \pm {:.2f}$".format(ms[0], ms[2])


def feed_dict(data_batch, ex):
    """
    Generate the feed_dicts for boml_optimizer.run() with data_batch and the instance of BOMLExperiment
    :param data_batch: each batch of data for exery iteration
    :param ex: instance of BOMLExperiment
    :return:
    """
    data_batch = data_batch[0]
    train_fd = {ex.x: data_batch.train.data, ex.y: data_batch.train.target}
    valid_fd = {ex.x_: data_batch.test.data, ex.y_: data_batch.test.target}
    return train_fd, valid_fd
