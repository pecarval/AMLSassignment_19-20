# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:12:35 2019
@author: Pedro
"""

import pandas as pd
import os
from matplotlib import image
import numpy as np
import torch
from torchvision import transforms, utils, datasets, models
from torch.utils.data import Dataset, DataLoader


def mainB2():
    '''
    Loads data into dataloaders for model training, validation and
    testing of task B2, performing data augmentation on training
    data and pre-processing all data (normalization and re-sizing)
    '''
    

    # Data normalization for training
    data_transforms = {
        'train': transforms.Compose([
            transforms.CenterCrop(178),
            transforms.RandomApply([transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(178),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop(178),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    data_dir = 'dataset/B2/CNN/'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    test_datasets = datasets.ImageFolder(os.path.join(data_dir, 'test'),data_transforms['test'])
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True, num_workers=4)

    return dataloaders, test_dataloader, dataset_sizes