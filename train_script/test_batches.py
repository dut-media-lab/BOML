import argparse
import pickle
from threading import Thread
import os

import matplotlib
matplotlib.use('Agg')
import sys
import inspect, os
sys.path.append('../')
os.environ['DATASETS_FOLDER'] = '../'
os.environ['EXPERIMENTS_FOLDER'] = '../'
import tensorflow as tf
import numpy as np
import time
import boml
import boml.load_data as dl




from train_script.script_helper import *
from boml.setup_model import BMLNetOmniglotHO_v1, BMLNetMiniHO_v1


dl.DATASET_FOLDER = 'datasets'

map_dict = {'omniglot': {'data_loader': dl.datasets.load_full_dataset.meta_omniglot, 'model': BMLNetOmniglotHO_v1},
            'miniimagenet': {'data_loader': dl.datasets.load_full_dataset.meta_mini_imagenet, 'model': BMLNetMiniHO_v1}}

metasets = dl.datasets.load_full_dataset.meta_omniglot(std_num_classes=5, examples_train=
            1, examples_test=15)
train_batches = metasets.train.loaded_images
print(len(list(train_batches)))
mini_dataset = boml.sample_mini_dataset(train_batches, 5, 10)
mini_batches = boml.mini_batches(mini_dataset, 5, 10,
                                 False)
print(mini_batches)