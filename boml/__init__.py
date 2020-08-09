try:
    from boml.data_loader.experiment import BMLExperiment
    from boml.data_loader import meta_mini_imagenet, meta_omniglot, OmniglotMetaDataset,\
        ImageNetMetaDataset, mnist, read_dataset, augment_dataset
    from boml.networks.BOMLNetMetaInitV1 import BOMLNetMetaInitV1, BOMLNetOmniglotMetaInitV1, BOMLNetMiniMetaInitV1
    from boml.networks.BOMLNetMetaReprV1 import BOMLNetMetaReprV1, BOMLNetMiniMetaReprV1, BOMLNetOmniglotMetaReprV1
    from boml.networks.BOMLNetMetaInitV2 import BOMLNetOmniglotMetaInitV2, BOMLNetMiniMetaInitV2
    from boml.networks.BOMLNetMetaReprV2 import BOMLNetOmniglotMetaReprV2, BOMLNetMiniMetaReprV2
    from boml.networks.BOMLNet import BOMLNetFeedForward
    from boml.optimizers import BOMLOptSGD, BOMLOptMomentum, BOMLOptAdam
    from boml.core import BOMLOptimizer
    from boml.extension import get_outerparameter
    from boml import utils
    from train_script.script_helper import *
    from boml import extension
except ImportError:
    print('py_bml package not complete')

try:
    import tensorflow as tf
except ImportError:
    print("tensorflow platform need be installed for further research")
