import os, time
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import image
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern

import torch
from torchvision import transforms, utils, datasets, models
from torch.utils.data import Dataset, DataLoader
import HelperFunctions.LandmarksMain.landmarksB1 as landmarks


# ======================================================================================================================
# Data Pre-Processing for SVM Model with Facial Landmarks
# ======================================================================================================================


def mainB1Landmarks():
    '''
    Extracts facial landmarks for each picture
    Performs train/test spliting (90% train, 10% test)
    Implements dimensionality reduction by scaling and performing PCA
    
    Returns:
        - pca_train : Train dataset of facial landmarks after PCA
        - pca_test : Test dataset of facial landmarks after PCA
        - lbs_train : Labels of training dataset
        - lbs_test : Labels of testing dataset
    '''
    
    # Extracting facial landmarks
    train_imgs, lbs = landmarks.extract_features_labels('./Datasets/dataset/B1/SVM/original/')
    addtest_imgs, lbs_addtest = landmarks.extract_features_labels('./Datasets/dataset/B1/SVM/addtest/')

    # Splitting data into 90% train and 10% test
    tr_data, te_data, lbs_train, lbs_test = train_test_split(train_imgs, lbs, test_size=0.1)
    data_train = tr_data.reshape(tr_data.shape[0], tr_data.shape[1]*tr_data.shape[2])
    data_test = te_data.reshape(te_data.shape[0], te_data.shape[1]*te_data.shape[2])
    data_addtest = addtest_imgs.reshape(addtest_imgs.shape[0], addtest_imgs.shape[1]*addtest_imgs.shape[2])

    # Applying dimensionality reduction
    pca_train, pca_test, pca_addtest = dimensionality_reduction(data_train, data_test, data_addtest)

    return pca_train, pca_test, pca_addtest, lbs_train, lbs_test, lbs_addtest
 

def dimensionality_reduction(train_data, test_data, addtest_data):
    '''
    Scales train and test datasets
    Implements Principal Component Analysis (PCA) on both datasets

    Keyword arguments:
        - train_data : Raw train dataset of facial landmarks
        - test_data : Raw test dataset of facial landmarks
        - addtest_data : Raw additional test dataset of facial landmarks

    Returns:
        - train_pca : Train dataset of facial landmarks after PCA
        - test_pca : Test dataset of facial landmarks after PCA
        - addtest_pca : Additional test dataset of facial landmarks after PCA
    '''

    # Scaling both datasets
    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    addtest_data = scaler.transform(addtest_data)

    # Applying PCA to both datasets
    pca = PCA(n_components = 'mle', svd_solver = 'full')
    pca.fit(train_data)
    train_pca = pca.transform(train_data)
    test_pca = pca.transform(test_data)
    addtest_pca = pca.transform(addtest_data)

    return train_pca, test_pca, addtest_pca


# ======================================================================================================================
# Data Pre-Processing for pre-trained CNN models
# ======================================================================================================================


def mainB1CNN():
    '''
    Loads train/validation/testing dataset into each separate dataloader
    Applies transformations to each of the datasets (Pre-processing + Augmentation)
    
    Returns:
        - dataloaders : PyTorch DataLoader with transformed train and validation datasets
        - test_dataloaders : PyTorch DataLoader with transformed test and additional test datasets
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

    data_dir = './Datasets/dataset/B1/CNN/'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    test_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms['test']) for x in ['test', 'addtest']}
    test_dataloaders = {x: torch.utils.data.DataLoader(test_datasets[x], batch_size=64, shuffle=True, num_workers=4) for x in ['test', 'addtest']}

    return dataloaders, test_dataloaders, dataset_sizes