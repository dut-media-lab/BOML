try:
    from boml.load_data.datasets.dl_utils import *
    from boml.load_data.datasets.structures import Dataset, MetaDataset
    from boml.load_data.datasets.load_full_dataset import meta_omniglot, meta_mini_imagenet, OmniglotMetaDataset, \
        ImageNetMetaDataset, mnist
    from boml.load_data.datasets.load_batches_dataset import augment_dataset,read_dataset
except ImportError:
    print('datasets package import error!')
