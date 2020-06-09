from __future__ import print_function, division
import torch
import os
import numpy as np
from torch.utils.data import Dataset
import random


# %%
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        tensor_dict = {}
        if sample.get('audio') is not None:
            tensor_dict['audio'] = torch.from_numpy(sample['audio'].astype(np.float32))
        if sample.get('image') is not None:
            tensor_dict['image'] = torch.from_numpy(sample['image'].astype(np.float32))
        tensor_dict['label'] = int(sample['label'])

        return tensor_dict


class Normalize(object):
    """Input image cleaning."""

    def __init__(self, mean_vector, std_devs):
        self.mean_vector, self.std_devs = mean_vector, std_devs

    def __call__(self, sample):
        tensor_dict = {}
        if sample.get('audio') is not None:
            tensor_dict['audio'] = sample['audio']
        if sample.get('image') is not None:
            image = self._normalize(sample['image'], mean=self.mean_vector, std=self.std_devs)
            tensor_dict['image'] = image
        tensor_dict['label'] = int(sample['label'])

        return tensor_dict

    def _normalize(self, tensor, mean, std):
        """Normalize a tensor image with mean and standard deviation.
        See ``Normalize`` for more details.
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.
        Returns:
            Tensor: Normalized Tensor image.
        """
        if not self._is_tensor_image(tensor):
            raise TypeError('tensor is not a torch image.')
        # TODO: make efficient
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor

    def _is_tensor_image(self, img):
        return torch.is_tensor(img) and img.ndimension() == 3


class RandomModalityMuting(object):
    """Randomly turn a mode off."""

    def __init__(self, p_muting=0.1):
        self.p_muting = p_muting

    def __call__(self, sample):
        rval = random.random()

        im = sample['image']
        au = sample['audio']
        if rval <= self.p_muting:
            vval = random.random()

            if vval <= 0.5:
                im = sample['image'] * 0
            else:
                au = sample['audio'] * 0

        return {'image': im, 'audio': au, 'label': sample['label']}


# %%
class AVMnist(Dataset):

    def __init__(self, root_dir='./avmnist',  # args.datadir
                 transform=None,
                 stage='train',
                 modal_separate=False,
                 modal=None):
        """
        Args:
            root_dir (string): Directory where data is.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.modal_separate = modal_separate
        self.modal = modal
        if not modal_separate:
            if stage == 'train':
                self.audio_data = np.load(os.path.join(root_dir, 'audio', 'train_data.npy'))
                self.mnist_data = np.load(os.path.join(root_dir, 'image', 'train_data.npy'))
                self.labels = np.load(os.path.join(root_dir, 'train_labels.npy'))
            else:
                self.audio_data = np.load(os.path.join(root_dir, 'audio', 'test_data.npy'))
                self.mnist_data = np.load(os.path.join(root_dir, 'image', 'test_data.npy'))
                self.labels = np.load(os.path.join(root_dir, 'test_labels.npy'))

            self.audio_data = self.audio_data[:, np.newaxis, :, :]
            self.mnist_data = self.mnist_data.reshape(self.mnist_data.shape[0], 1, 28, 28)
        else:
            if modal:
                if modal not in ['audio', 'image']:
                    raise ValueError('the value of modal is allowed')

                if stage == 'train':
                    self.data = np.load(os.path.join(root_dir, modal, 'train_data.npy'))
                    self.labels = np.load(os.path.join(root_dir, 'train_labels.npy'))
                else:
                    self.data = np.load(os.path.join(root_dir, modal, 'test_data.npy'))
                    self.labels = np.load(os.path.join(root_dir, 'test_labels.npy'))

                if modal == 'audio':
                    self.data = self.data[:, np.newaxis, :, :]
                elif modal == 'image':
                    self.data = self.data.reshape(self.data.shape[0], 1, 28, 28)

            else:
                raise ValueError('the value of modal should be given')

    def __len__(self):
        return self.mnist_data.shape[0] if not self.modal_separate else self.data.shape[0]

    def __getitem__(self, idx):
        if not self.modal_separate:
            image = self.mnist_data[idx]
            audio = self.audio_data[idx]
            label = self.labels[idx]

            sample = {'image': image, 'audio': audio, 'label': label}
        else:
            data = self.data[idx]
            label = self.labels[idx]

            sample = {self.modal: data, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
