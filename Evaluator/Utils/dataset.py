import random
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np
import threading
import logging
import os
import gc
import shutil
from io import BytesIO
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def convert_to_pil(bytes_obj):
    img = Image.open(BytesIO(bytes_obj))
    #img = Image.open(bytes_obj)
    #img = bytes_obj 
    return img.convert('RGB')

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


class ReadImageThread(threading.Thread):
    def __init__(self, root, fnames, class_id, target_list):
        threading.Thread.__init__(self)
        self.root = root
        self.fnames = fnames
        self.class_id = class_id
        self.target_list = target_list

    def run(self):
        for fname in self.fnames:
            if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                path = os.path.join(self.root, fname)
                try:
                    image = Image.open(path)
                except(OSError, NameError):
                    os.remove(path)
                    continue
                image.close()
                if(random.random()<0.1):
                    continue
                with open(path, 'rb') as f:
                    image = f.read()
                item = (image, self.class_id)
                self.target_list.append(item)


class InMemoryDataset(data.Dataset):
    def __init__(self, path, transform=None, num_workers=1):
        super(InMemoryDataset, self).__init__()
        self.path = path
        self.transform = transform
        self.samples = []
        classes, class_to_idx = self.find_classes(self.path)
        dir = os.path.expanduser(self.path)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                if num_workers == 1:
                    for fname in sorted(fnames):
                        if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                            path = os.path.join(root, fname)
                            try: 
                               image = Image.open(path)
                            except(OSError, NameError):
                               os.remove(path)
                               continue
                            image.close()
                            with open(path, 'rb') as f:
                               image = f.read()
                            item = (image, class_to_idx[target])
                            self.samples.append(item)
                else:
                    fnames = sorted(fnames)
                    num_files = len(fnames)
                    threads = []
                    res = [[] for i in range(num_workers)]
                    num_per_worker = num_files // num_workers
                    for i in range(num_workers):
                        start_index = num_per_worker * i
                        end_index = num_files if i == num_workers - \
                            1 else num_per_worker * (i+1)
                        thread = ReadImageThread(
                            root, fnames[start_index:end_index], class_to_idx[target], res[i])
                        threads.append(thread)
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join()
                    for item in res:
                        self.samples += item
                    del res, threads
                    gc.collect()
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample, target = self.samples[index]
        sample = convert_to_pil(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.path)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    @staticmethod
    def find_classes(root):
        classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

class ZipDataset(data.Dataset):
    def __init__(self, path, transform=None):
        super(ZipDataset, self).__init__()
        self.path = os.path.expanduser(path)
        self.transform = transform
        self.samples = []
        with zipfile.ZipFile(self.path, 'r') as reader:
            classes, class_to_idx = self.find_classes(reader)
            fnames = sorted(reader.namelist())
        for fname in fnames:
            if self.is_directory(fname):
                continue
            target = self.get_target(fname)
            item = (fname, class_to_idx[target])
            self.samples.append(item)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample, target = self.samples[index]
        with zipfile.ZipFile(self.path, 'r') as reader:
            sample = reader.read(sample)
        sample = convert_to_pil(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.path)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
    @staticmethod
    def is_directory(fname):
        if fname.startswith('n') and fname.endswith('/'):
            return True
        return False
    
    @staticmethod
    def get_target(fname):
        assert fname.startswith('n')
        return fname.split('/')[0]
    
    @staticmethod
    def find_classes(reader):
        classes = [ZipDataset.get_target(name) for name in reader.namelist() if ZipDataset.is_directory(name)]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

def build_cifar10(data_path,
                  cutout_size=16,
                  num_worker=10,
                  train_batch_size=32,
                  eval_batch_size=32,
                  split_train_for_valid: float = None,
                  small_set=1, **kwargs):

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
        n = int(len(train_data)*small_set)
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


def build_cifar100(data_path,
                   cutout_size=16,
                   num_worker=10,
                   train_batch_size=32,
                   eval_batch_size=32,
                   split_train_for_valid: float = None, **kwargs):

    train_transform, valid_transform = _data_transforms_cifar10(cutout_size)
    if split_train_for_valid is None:
        train_data = dset.CIFAR100(
            root=data_path, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(
            root=data_path, train=False, download=True, transform=valid_transform)

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=num_worker)
        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=eval_batch_size, shuffle=False, pin_memory=True, num_workers=num_worker)
    else:
        train_data = dset.CIFAR100(
            root=data_path, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(
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


def build_imagenet(data_path,
                   load_num_work=32,
                   feed_num_work=16,
                   zip_file: bool = False,
                   lazy_load: bool = False,
                   train_batch_size: int = 128,
                   valid_batch_size: int = 128):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2),
        transforms.ToTensor(),
        normalize])

    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])
    if zip_file:
        logging.debug('Loading data from zip file')
        traindir = os.path.join(data_path, 'train.zip')
        validdir = os.path.join(data_path, 'valid.zip')
        if lazy_load:
            train_data = ZipDataset(traindir, train_transform)
            valid_data = ZipDataset(validdir, valid_transform)
        else:
            logging.debug('Loading data into memory')
            train_data = InMemoryZipDataset(
                traindir, train_transform, num_workers=load_num_work)
            valid_data = InMemoryZipDataset(
                validdir, valid_transform, num_workers=load_num_work)
    else:
        logging.debug('Loading data from directory')
        traindir = os.path.join(data_path, 'ILSVRC2012_img_train/')
        validdir = os.path.join(data_path, 'ILSVRC2012_img_val/')
        if lazy_load:
            train_data = dset.ImageFolder(traindir, train_transform)
            valid_data = dset.ImageFolder(validdir, valid_transform)
        else:
            logging.debug('Loading data into memory')
            train_data = InMemoryDataset(
                traindir, train_transform, num_workers=load_num_work)
            valid_data = InMemoryDataset(
                validdir, valid_transform, num_workers=load_num_work)

    logging.info('[ImageNet] Found %d in training data', len(train_data))
    logging.info('[ImageNet] Found %d in validation data', len(valid_data))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=feed_num_work)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=valid_batch_size, shuffle=True, pin_memory=True, num_workers=feed_num_work)
    return train_queue, valid_queue

class auto_data(object):
    @staticmethod
    def generate_data(value_range=(0,15), length_range=(10,20), num_samples=500000, data_parser=lambda x: ' '.join(['.'.join([str(j) for j in i]) for i in x.reshape(-1, 3)])):
        dataset =  list()
        data1 = list()
        data2 = list()
        for i in range(length_range[0], length_range[1], 1):
            data1.extend([i for i in np.random.randint(value_range[0], value_range[1], size=(np.ceil(num_samples/(length_range[1]-length_range[0])).astype('int'), i*3))])
            data2.extend([i for i in np.random.randint(value_range[0], value_range[1], size=(np.ceil(num_samples/(length_range[1]-length_range[0])).astype('int'), i*3))])
        np.random.shuffle(data1)
        np.random.shuffle(data2)
        for d0, d1 in zip(data1, data2):
            dataset.append('{0} <---> {1}\t{0} <---> {1}'.format(data_parser(d0), data_parser(d1)))
        return dataset
    
    @staticmethod
    def save_data(data, save_path='./Res/PretrainModel/', file_name='trainset.txt'):
        file = open(os.path.join(save_path, file_name), 'a')
        file.write("\n".join(data))
        file.flush()
        file.close()
        
