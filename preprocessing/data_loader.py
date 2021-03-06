from __future__ import print_function, division, absolute_import

import glob
import random
import time

# Python 2/3 support
try:
    import queue
except ImportError:
    import Queue as queue

import cv2
import numpy as np
import torch as th
from joblib import Parallel, delayed
from torch.multiprocessing import Queue, Process


from .utils import preprocessInput
import torch.utils.data
from torch.utils.data import Sampler
try:
    from sklearn.model_selection import StratifiedShuffleSplit
except:
    print('Need scikit-learn for this functionality')


def sample_coordinates(coord_1, max_distance, percentage):
    """
    Sampling from a coordinate A, a second one B within a maximum distance [max_distance X percentage]

    :param coord_1: (int) sample first coordinate
    :param max_distance: (int) max value of coordinate in the axis
    :param percentage: (float) maximum occlusion as a percentage
    :return: (tuple of int)
    """
    min_coord_2 = max(0, coord_1 - max_distance * percentage)
    max_coord_2 = min(coord_1 + max_distance * percentage, max_distance)
    coord_2 = np.random.randint(low=min_coord_2, high=max_coord_2)
    return min(coord_1, coord_2), max(coord_1, coord_2)


def preprocessImage(image, img_reshape=None, convert_to_rgb=True, apply_occlusion=False, occlusion_percentage=0.5):
    """
    :param image: (np.ndarray) image (BGR or RGB)
    :param img_reshape: (None or tuple e.g. (3, 128, 128)) reshape image to (128, 128)
    :param convert_to_rgb: (bool) whether the conversion to rgb is needed or not
    :param apply_occlusion: (bool) whether to occludes part of the images or not
                            (used for training denoising autoencoder)
    :param occlusion_percentage: (float) max percentage of occlusion (in width and height)
    :return: (np.ndarray)
    """
    # Resize
    if img_reshape is not None:
        assert isinstance(img_reshape, tuple), "'img_reshape' should be a tuple like: (3,128,128)"
        assert img_reshape[0] < 10, "'img_reshape' should be a tuple like: (3,128,128)"
        im = cv2.resize(image, img_reshape[1:], interpolation=cv2.INTER_AREA)
    else:
        im = image
        img_reshape = (im.shape[-1],) + im.shape[:-1]
    # Convert BGR to RGB
    if convert_to_rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # Normalize
    im = preprocessInput(im.astype(np.float32))

    img_height, img_width = img_reshape[1:]
    if apply_occlusion:
        h_1 = np.random.randint(img_height)
        h_1, h_2 = sample_coordinates(h_1, img_height, percentage=occlusion_percentage)
        w_1 = np.random.randint(img_width)
        w_1, w_2 = sample_coordinates(w_1, img_width, percentage=occlusion_percentage)
        noisy_img = im
        # This mask is set by applying zero values to corresponding pixels.
        noisy_img[h_1:h_2, w_1:w_2, :] = 0.
        im = noisy_img

    return im


class BalancedLabelSampler(Sampler): ## TODO useless
    r"""Balanced classes sampler (batch-level)

    Arguments:
        data_source (Dataset): dataset to sample from
        class_labels (np.ndarray): 

    """
    def __init__(self, data_source, class_labels, subset=None, batch_size=32):
        self.data_source = data_source
        self.class_labels = class_labels
        self.batch_size = batch_size
        bin_count = np.bincount(self.class_labels)
        self.num_classes = len(bin_count)
        ## Statistic abount dataset
        print("Find {} classes: max/min samples per class: {}/{}; variation amplitude (std/max) {:.2%}".format(\
            self.num_classes, np.max(bin_count), np.min(bin_count), np.std(bin_count)/np.max(bin_count)))
        assert len(class_labels) == len(self.data_source), "class_labels should have same length as dataset."
        # class_dict = 
        num_batches = int(len(self.data_source)/self.batch_size)
        resampling = []
        replacement = not (self.num_classes > self.batch_size)
        for ind in range(num_batches):
            sample_classes = np.random.choice(np.arange(self.num_classes), self.batch_size, replace=replacement)
            for cls_label in sample_classes:
                return
            # resampling

    def __iter__(self):
        return 
    def __len__(self):
        return len(self.data_source)


class StratifiedSampler(Sampler): ## TODO useless
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size, subset=None):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector
        self.subset = subset

    def gen_sample_array(self):

        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = th.randn(self.class_vector.size(0), 2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)

class RobotEnvDataset(torch.utils.data.Dataset):
    r"""Robot Envinronment Dataset for DataLoader (Pytorch natively supported)

    A Custom dataloader to work with our datasets, and to prepare data for the different models
    (inverse, priors, autoencoder, ...)

    :param minibatchlist: ([np.array]) list of observations indices (grouped per minibatch)
    :param images_path: (np.array) Array of path to images
    :param n_workers: (int) number of preprocessing worker (load and preprocess each image)
    :param multi_view: (bool)
    :param use_triplets: (bool)
    :param infinite_loop: (bool) whether to have an iterator that can be resetted, set to False, it
    :param max_queue_len: (int) Max number of minibatches that can be preprocessed at the same time
    :param apply_occlusion: is the use of occlusion enabled - when using DAE (bool)
    :param occlusion_percentage: max percentage of occlusion when using DAE (float)
    :param mode: (int)
    :param img_shape: (tuple or None) if None, image will not be resize, else: resize image to new shape (channels first)) e.g. img_shape = (3, 128, 128).

        Set to True, the dataloader will output both `obs` and `next_obs` (a tuple of th.Tensor)
        Set to false, it will only output one th.Tensor.
    """

    def __init__(self, sample_indices, images_path, actions, rewards, episode_starts,
                 img_shape=None,
                 mode=1,
                 ground_truth=None,
                 multi_view=False,
                 use_triplets=False,
                 apply_occlusion=False,
                 occlusion_percentage=0.5,
                 dtype=np.float32,
                 img_extension="png"):
        super(RobotEnvDataset, self).__init__()
        # Initialization
        self.sample_indices = sample_indices
        self.images_path = images_path
        self.actions = actions
        self.rewards = rewards
        self.episode_starts = episode_starts
        self.ground_truth = ground_truth # only used for supervised learning: mode == 3
        if self.ground_truth is not None:
            assert mode == 3, "Ground truth is only used for mode=3 (supervised learning)"

        self.img_shape = img_shape
        self.mode = mode
        self.use_triplets = use_triplets
        self.multi_view = multi_view
        assert not self.multi_view
        # apply occlusion for training a DAE
        self.apply_occlusion = apply_occlusion
        self.occlusion_percentage = occlusion_percentage

        self.dtype = dtype
        self.img_extension = img_extension
        
        self.class_labels = np.array(list(map(lambda x:int(x.split("/")[-2].split("_")[-1]), images_path)))
        # self.random_target_balance = random_target_balance
        
    def __len__(self):
        ## 'Denotes the total number of samples'
        return len(self.sample_indices)

    def _get_one_img(self, image_path):
        # self.minibatchlist = minibatchlist
        image_path = 'data/' + image_path.split('.{}'.format(self.img_extension))[0]  # [TODO]

        img = cv2.imread("{}.{}".format(image_path, self.img_extension))
        if img is None:
            raise ValueError("tried to load {}.{}, but it was not found".format(image_path, self.img_extension))
        img = preprocessImage(img, img_reshape=self.img_shape,
                              apply_occlusion=self.apply_occlusion,
                              occlusion_percentage=self.occlusion_percentage)
        img = img.transpose(2, 0, 1)
        return img

    def __getitem__(self, index):
        # 'Generates one sample of data': (main)

        index = self.sample_indices[index]  # real index of samples
        if (index+1) >= len(self.actions) or self.episode_starts[index + 1]:
            # the case where 'index' is the end of episode, no next observation.
            index -= 1  # this may repeat some observations, but the proba is rare.

        image_path = self.images_path[index]
        # Load data and get label
        
        if self.mode == 1: ## main mode, for training
            img = self._get_one_img(image_path)
            img_next = self._get_one_img(self.images_path[index+1])
            action = self.actions[index]
            reward = self.rewards[index]
            cls_gt = self.class_labels[index]
            return index, img.astype(self.dtype), img_next.astype(self.dtype), action, reward, cls_gt
        elif self.mode == 0: # for evaluation
            img = self._get_one_img(image_path)
            return img.astype(self.dtype)
        elif self.mode == 2: # for evaluation
            img = self._get_one_img(image_path)
            img_next = self._get_one_img(self.images_path[index+1])
            reward = self.rewards[index]
            return img.astype(self.dtype), img_next.astype(self.dtype), reward
        elif self.mode == 3: # for supervised learning: learn ground truth
            img = self._get_one_img(image_path)
            ground_truth_state = self.ground_truth[index]
            return img.astype(self.dtype), ground_truth_state.astype(self.dtype)
        else:
            raise NotImplementedError
