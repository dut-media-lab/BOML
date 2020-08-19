try:
    from boml.load_data.experiment import BMLExperiment
    from boml.load_data import meta_mini_imagenet, meta_omniglot, OmniglotMetaDataset,\
        ImageNetMetaDataset, mnist
    from boml.setup_model.BOMLNetMetaInitV1 import BOMLNetMetaInitV1, BOMLNetOmniglotMetaInitV1, BOMLNetMiniMetaInitV1
    from boml.setup_model.BOMLNetMetaReprV1 import BOMLNetMetaReprV1, BOMLNetMiniMetaReprV1, BOMLNetOmniglotMetaReprV1
    from boml.setup_model.BOMLNetMetaInitV2 import BOMLNetOmniglotMetaInitV2, BOMLNetMiniMetaInitV2
    from boml.setup_model.BOMLNetMetaReprV2 import BOMLNetOmniglotMetaReprV2, BOMLNetMiniMetaReprV2
    from boml.setup_model.BOMLNet import BOMLNetFeedForward,BOMLNet
    from boml.optimizer import BOMLOptSGD, BOMLOptMomentum, BOMLOptAdam
    from boml.boml_optimizer import BOMLOptimizer
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
