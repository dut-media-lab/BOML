try:

    from boml.load_data.experiment import BMLExperiment
    from boml.load_data import meta_mini_imagenet, meta_omniglot, OmniglotMetaDataset,\
        ImageNetMetaDataset, mnist, read_dataset, augment_dataset
    from boml.setup_model.BOMLNetMetaInitV1 import BOMLNetMiniMetaInitV1, BOMLNetOmniglotMetaInitV1
    from boml.setup_model.BOMLNetMetaReprV1 import BOMLNetMiniMetaReprV1, BOMLNetOmniglotMetaReprV1
    from boml.setup_model.BOMLNetMetaInitV2 import BOMLNetOmniglotMetaInitV2, BOMLNetMiniMetaInitV2
    from boml.setup_model.BOMLNetMetaReprV2 import BOMLNetMiniMetaReprV2, BOMLNetOmniglotMetaReprV2
    from boml.setup_model.BOMLNet import BOMLNetFeedForward, BOMLNet
    from boml.optimizer import BOMLOptAdam, BOMLOptMomentum, BOMLOptSGD
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
