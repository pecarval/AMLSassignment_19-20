import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models


def mainB2CNN():
    '''
    Loads train/validation/testing dataset into each separate dataloader
    Applies transformations to each of the datasets (Pre-processing + Augmentation)
    
    Returns:
        - dataloaders : PyTorch DataLoader with transformed train, val and test datasets
        - dataset_sizes : Size of training and validation dataset (Needed for accuracy computation in training)
    '''
    

    # Data pre-processing and Augmentation for each dataset
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

    data_dir = './Datasets/dataset/B2/'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val','test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val','test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return dataloaders, dataset_sizes