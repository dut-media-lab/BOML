import os
import random

import numpy as np
from PIL import Image


def read_dataset(data_dir):
    """
    Iterate over the characters in a data directory.

    Args:
      data_dir: a directory of alphabet directories.

    Returns:
      An iterable over images.

    The dataset is unaugmented and not split up into
    training and test sets.
    """
    if os.path.exists(os.path.join(data_dir, 'train')):
        return tuple(_read_classes(os.path.join(data_dir, x)) for x in ['train', 'val', 'test'])
    else:
        return split_dataset(read_omniglot_dataset(data_dir=data_dir))


def read_omniglot_dataset(data_dir):
    """
    Iterate over the characters in a data directory.

    Args:
      data_dir: a directory of alphabet directories.

    Returns:
      An iterable over Characters.

    The dataset is unaugmented and not split up into
    training and test sets.
    """
    for alphabet_name in sorted(os.listdir(data_dir)):
        alphabet_dir = os.path.join(data_dir, alphabet_name)
        if not os.path.isdir(alphabet_dir):
            continue
        for char_name in sorted(os.listdir(alphabet_dir)):
            if not char_name.startswith('character'):
                continue
            yield OmniglotCharacter(os.path.join(alphabet_dir, char_name), 0)


def _read_classes(dir_path):
    """
    Read the WNID directories in a directory.
    """
    return [ImageNetClass(os.path.join(dir_path, f)) for f in os.listdir(dir_path)
            if f.startswith('n')]


def split_dataset(dataset, num_train=1200):
    """
    Split the dataset into a training and test set.

    Args:
      dataset: an iterable of Characters.

    Returns:
      A tuple (train, test) of Character sequences.
    """
    all_data = list(dataset)
    random.shuffle(all_data)
    return all_data[:num_train], all_data[num_train:]


def augment_dataset(dataset):
    """
    Augment the dataset by adding 90 degree rotations.

    Args:
      dataset: an iterable of Characters.

    Returns:
      An iterable of augmented Characters.
    """
    for character in dataset:
        for rotation in [0, 90, 180, 270]:
            yield OmniglotCharacter(character.dir_path, rotation=rotation)


# pylint: disable=R0903
class ImageNetClass:
    """
    A single image class.
    """
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self._cache = {}

    def sample(self, num_images):
        """
        Sample images (as numpy arrays) from the class.

        Returns:
          A sequence of 84x84x3 numpy arrays.
          Each pixel ranges from 0 to 1.
        """
        names = [f for f in os.listdir(self.dir_path) if f.endswith('.JPEG')]
        random.shuffle(names)
        images = []
        for name in names[:num_images]:
            images.append(self._read_image(name))
        return images

    def _read_image(self, name):
        if name in self._cache:
            return self._cache[name].astype('float32') / 0xff
        with open(os.path.join(self.dir_path, name), 'rb') as in_file:
            img = Image.open(in_file).resize((84, 84)).convert('RGB')
            self._cache[name] = np.array(img)
            return self._read_image(name)


class OmniglotCharacter:
    """
    A single character class.
    """
    def __init__(self, dir_path, rotation=0):
        self.dir_path = dir_path
        self.rotation = rotation
        self._cache = {}

    def sample(self, num_images):
        """
        Sample images (as numpy arrays) from the class.

        Returns:
          A sequence of 28x28 numpy arrays.
          Each pixel ranges from 0 to 1.
        """
        names = [f for f in os.listdir(self.dir_path) if f.endswith('.png')]
        random.shuffle(names)
        images = []
        for name in names[:num_images]:
            images.append(self._read_image(os.path.join(self.dir_path, name)))
        return images

    def _read_image(self, path):
        from scipy.ndimage.interpolation import rotate
        if path in self._cache:
            return self._cache[path]
        with open(path, 'rb') as in_file:
            img = 1. - np.reshape(Image.open(in_file), (28, 28, 1)) / 255.
            img = rotate(img, self.rotation, reshape=False)
            self._cache[path] = np.array(img).astype('float32')
            return self._cache[path]
