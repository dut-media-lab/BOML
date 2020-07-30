try:

    from py_bml.DataLoader.experiment import BMLExperiment
    from py_bml.DataLoader import meta_mini_imagenet, meta_omniglot, OmniglotMetaDataset,\
        ImageNetMetaDataset, mnist, read_dataset, augment_dataset
    from py_bml.Networks.BMLNetHO_v1 import BMLNetMiniHO_v1, BMLNetOmniglotHO_v1
    from py_bml.Networks.BMLNetMetaRepr_v1 import BMLNetOmniglotMetaRepr_v1, BMLNetMiniMetaRepr_v1
    from py_bml.Networks.BMLNetHO_v2 import BMLNetOmniglotHO_v2, BMLNetMiniHO_v2
    from py_bml.Networks.BMLNetMetaRepr_v2 import BMLNetMiniMetaRepr_v2, BMLNetOmniglotMetaRepr_v2
    from py_bml.Networks.BMLNet import BMLNetFeedForward
    from py_bml.Optimizer import BMLOptSGD, BMLOptMomentum, BMLOptAdam
    from py_bml.Core import BMLHOptimizer
    from py_bml.extension import get_outerparameter
    from py_bml import utils
    from train_script.script_helper import *
    from py_bml import extension
except ImportError:
    print('py_bml package not complete')

try:
    import tensorflow as tf
except ImportError:
    print("tensorflow platform need be installed for further research")
