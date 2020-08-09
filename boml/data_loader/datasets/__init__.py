try:
    from boml.data_loader.datasets.dl_utils import *
    from boml.data_loader.datasets.structures import Dataset, MetaDataset
    from boml.data_loader.datasets.load_full_dataset import meta_omniglot, meta_mini_imagenet, OmniglotMetaDataset, \
        ImageNetMetaDataset, mnist
    from boml.data_loader.datasets.load_batches_dataset import augment_dataset,read_dataset
except ImportError:
    print('datasets package import error!')
