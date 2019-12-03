import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def _data_transforms_cifar10(cutout_size):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if cutout_size is not None:
        train_transform.transforms.append(Cutout(cutout_size))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def build_cifar10(data_path,
                cutout_size=16,
                num_worker=10,
                train_batch_size=32,
                eval_batch_size=32,
                split_train_for_valid:float=None, **kwargs):

    train_transform, valid_transform = _data_transforms_cifar10(cutout_size)
    if split_train_for_valid is None:
        train_data = dset.CIFAR10(
            root=data_path, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(
            root=data_path, train=False, download=True, transform=valid_transform)

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=num_worker)
        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=eval_batch_size, shuffle=False, pin_memory=True, num_workers=num_worker)
    else:
        train_data = dset.CIFAR10(
            root=data_path, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(
            root=data_path, train=True, download=True, transform=valid_transform)
        n = len(train_data)
        indices = list(range(n))
        split = int(np.floor(split_train_for_valid * n))
        np.random.shuffle(indices)
        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=train_batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                indices[:split]),
            pin_memory=True, num_workers=num_worker)
        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=eval_batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                indices[split:n]),
            pin_memory=True, num_workers=num_worker)

    return train_queue, valid_queue
