from re import A
from PIL import Image, ImageFile
import torch
import numpy as np
import os
import sys
import pickle
from numpy.testing import assert_array_almost_equal
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torch
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import ImageFilter

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Onehot(object):
    def __call__(self, sample, num_classes=31):
        target_onehot = torch.zeros(num_classes)
        target_onehot[sample] = 1

        return target_onehot

def train_transform():
    """
    Training images transform.

    Args
        None

    Returns
        transform(torchvision.transforms): transform
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

def train_aug_transform():
    """
    Training images transform.

    Args
        None

    Returns
        transform(torchvision.transforms): transform
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

def query_transform():
    """
    Query images transform.

    Args
        None

    Returns
        transform(torchvision.transforms): transform
    """
    # Data transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

def load_data(source_list, target_list, batch_size, num_workers, setting, task = 'cross'):

    office31.init(source_list, target_list, setting, task)
    query_dataset = office31('query', query_transform(),target_transform=Onehot())
    train_s_dataset = office31('train_s', train_transform(),target_transform=Onehot())
    train_t_dataset = office31('train_t', train_transform(),target_transform=Onehot())
    retrieval_dataset = office31('retrieval', query_transform(), target_transform=Onehot())
    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=num_workers,
    )
    train_s_dataloader = DataLoader(
        train_s_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=num_workers,
    )
    train_t_dataloader = DataLoader(
        train_t_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=num_workers,
    )
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=num_workers,
    )

    return query_dataloader, train_s_dataloader, train_t_dataloader, retrieval_dataloader


class office31(Dataset):
    def __init__(self, mode, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.aug = train_aug_transform()
        self.mode = mode

        if mode == 'train_s':
            self.data = office31.TRAIN_S_DATA
            self.targets = office31.TRAIN_S_TARGETS

        elif mode == 'train_t':
            self.data = office31.TRAIN_T_DATA
            self.targets = office31.TRAIN_T_TARGETS

        elif mode == 'query':
            self.data = office31.QUERY_DATA
            self.targets = office31.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = office31.RETRIEVAL_DATA
            self.targets = office31.RETRIEVAL_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')

    def __getitem__(self, index):
        img = Image.open(self.data[index]).convert('RGB')
        img_aug = self.aug(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, img_aug, self.target_transform(self.targets[index]), index

    def __len__(self):
        return self.data.shape[0]
    
    def get_targets(self):
        one_hot = torch.zeros((self.targets.shape[0],31))
        for i in range(self.targets.shape[0]):
            one_hot[i,:] = self.target_transform(self.targets[i])
        return  one_hot


    @staticmethod
    def init(source_list, target_list,setting, task):
        source_data = []
        source_label = []
        target_data = []
        target_label = []

        if setting == 'opda':
            common_id =[i for i in range(10)]
            source_id = [i for i in range(10,20)] + common_id
            target_id = [i for i in range(20,31)] + common_id
        elif setting == 'oda':
            common_id =[i for i in range(10)]
            source_id = common_id
            target_id = [i for i in range(10,21)] + common_id
        else:
            common_id =[i for i in range(10)]
            source_id = common_id + [i for i in range(10,31)]
            target_id = common_id
        
        with open(source_list, 'r') as f:
            for line in f.readlines():
                if int(line.split()[1]) in source_id:
                    source_data.append(line.split()[0].replace(\
                        '/data/office/domain_adaptation_images','/home/boot/whx/CDAN/data/office/office31'))
                    source_label.append(int(line.split()[1]))
        
        with open(target_list, 'r') as f:
            for line in f.readlines():
                if int(line.split()[1]) in target_id:
                    target_data.append(line.split()[0].replace(\
                        '/data/office/domain_adaptation_images','/home/boot/whx/CDAN/data/office/office31'))
                    target_label.append(int(line.split()[1]))

        source_data = np.array(source_data)
        source_label = np.array(source_label)
        target_data = np.array(target_data)
        target_label = np.array(target_label)

        if task == 'cross':
            
            
            cnt = np.count_nonzero(target_label < 10) 
            all = target_label.shape[0]
            ### perm_index是common class 
            perm_index = np.random.permutation(cnt)
            perm_index2 = np.random.permutation(all - cnt) + cnt
            ratio = 0.6
            threshold = int(cnt*ratio)
            threshold2 = int(0.1*all) - threshold
            query_index = perm_index[:threshold]
            train_t_index = np.concatenate((perm_index[threshold:],\
                            perm_index2),axis=0)
        
                                
            office31.QUERY_DATA = target_data[query_index]
            office31.QUERY_TARGETS = target_label[query_index]

            office31.TRAIN_S_DATA = source_data
            office31.TRAIN_S_TARGETS = source_label

            office31.TRAIN_T_DATA = target_data[train_t_index]
            office31.TRAIN_T_TARGETS = target_label[train_t_index]

            office31.RETRIEVAL_DATA = source_data
            office31.RETRIEVAL_TARGETS = source_label
