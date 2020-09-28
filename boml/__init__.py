# MIT License

# Copyright (c) 2020 Yaohua Liu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
try:
    import tensorflow as tf
except ImportError:
    print("No compatible version of TensorFlow has been installed!")

from boml.load_data.experiment import BOMLExperiment
from boml.load_data import (
    meta_mini_imagenet,
    meta_omniglot,
    OmniglotMetaDataset,
    ImageNetMetaDataset,
    mnist,
)
from boml.setup_model.meta_init_v1 import (
    BOMLNetMetaInitV1,
    BOMLNetOmniglotMetaInitV1,
    BOMLNetMiniMetaInitV1,
)
from boml.setup_model.meta_feat_v1 import (
    BOMLNetMetaFeatV1,
    BOMLNetMiniMetaFeatV1,
    BOMLNetOmniglotMetaFeatV1,
)
from boml.setup_model.meta_init_v2 import (
    BOMLNetOmniglotMetaInitV2,
    BOMLNetMiniMetaInitV2,
)
from boml.setup_model.meta_feat_v2 import (
    BOMLNetOmniglotMetaFeatV2,
    BOMLNetMiniMetaFeatV2,
)
from boml.setup_model.network import BOMLNet
from boml.setup_model.feedforward import BOMLNetFeedForward
from boml.optimizer import BOMLOptSGD, BOMLOptMomentum, BOMLOptAdam
from boml.boml_optimizer import BOMLOptimizer
from boml.extension import get_outerparameter
from boml import utils
from boml import extension


