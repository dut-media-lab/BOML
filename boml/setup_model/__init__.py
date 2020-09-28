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
from boml.setup_model import network
from boml.setup_model.feedforward import BOMLNetFeedForward
from boml.setup_model.meta_init_v1 import (
    BOMLNetMetaInitV1,
    BOMLNetMiniMetaInitV1,
    BOMLNetOmniglotMetaInitV1,
)
from boml.setup_model.meta_init_v2 import (
    BOMLNetMiniMetaInitV2,
    BOMLNetOmniglotMetaInitV2,
)
from boml.setup_model.meta_feat_v2 import (
    BOMLNetMiniMetaFeatV2,
    BOMLNetOmniglotMetaFeatV2,
)
from boml.setup_model.meta_feat_v1 import (
    BOMLNetMetaFeatV1,
    BOMLNetMiniMetaFeatV1,
    BOMLNetOmniglotMetaFeatV1,
)
