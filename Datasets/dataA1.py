import os
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

import Datasets.LandmarksMain.landmarksA1 as landmarks


# ======================================================================================================================
# Data Pre-Processing for SVM Model with Facial Landmarks
# ======================================================================================================================

def mainA1Landmarks():
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
    
    # Extracting facil landmarks
    imgs, lbs = landmarks.extract_features_labels()

    # Splitting data into 90% train and 10% test
    tr_data, te_data, lbs_train, lbs_test = train_test_split(imgs, lbs, test_size=0.1)
    data_train = tr_data.reshape(tr_data.shape[0], tr_data.shape[1]*tr_data.shape[2])
    data_test = te_data.reshape(te_data.shape[0], te_data.shape[1]*te_data.shape[2])

    # Applying dimensionality reduction
    pca_train, pca_test = dimensionality_reduction(data_train, data_test)

    return pca_train, pca_test, lbs_train, lbs_test
 

def dimensionality_reduction(train_data, test_data):
    '''
    Scales train and test datasets
    Implements Principal Component Analysis (PCA) on both datasets

    Keyword arguments:
        - train_data : Raw train dataset of facial landmarks
        - test_data : Raw test dataset of facial landmarks

    Returns:
        - train_pca : Train dataset of facial landmarks after PCA
        - test_pca : Train dataset of facial landmarks after PCA
    '''

    # Scaling both datasets
    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    # Applying PCA to both datasets
    pca = PCA(n_components = 'mle', svd_solver = 'full')
    pca.fit(train_data)
    train_pca = pca.transform(train_data)
    test_pca = pca.transform(test_data)

    return train_pca, test_pca


# ======================================================================================================================
# Data Pre-Processing for SVM Model with Local Binary Patterns (LBP)
# ======================================================================================================================


def mainA1LBP():
    '''
    Extracts LBP histograms for each picture
    Performs train/test spliting (90% train, 10% test)
    Implements dimensionality reduction by scaling and performing PCA
    
    Returns:
        - pca_train : Train dataset of LBP after PCA
        - pca_test : Test dataset of LBP after PCA
        - lbs_train : Labels of training dataset
        - lbs_test : Labels of testing dataset
    '''

    # Extracting LBP histograms
    imgs, lbs = extract_lbp()

    # Splitting dataset into 90% train and 10% test
    data_train, data_test, lbs_train, lbs_test = train_test_split(imgs, lbs, test_size=0.1)

    # Applying dimensionality reduction to dataset
    #pca_train, pca_test = dimensionality_reductionLBP(data_train, data_test)

    #return pca_train, pca_test, lbs_train, lbs_test
    return data_train, data_test, lbs_train, lbs_test

def extract_lbp():
    '''
    Converts images to grayscale for LBP to be applied
    Computes LBP for each picture
    Implements histogram of LBP

    Returns:
        - hist_lbp : Dataset of images after LBP histogram computation
        - lbs : Labels of entire dataset
    '''

    # Obtaining grayscale images and respective labels
    imgs, lbs = grayscale()

    # Defining parameters for LBP computation
    # radius : Defines radius of circle of neighours
    # numPoints : Defines number of neighbours to be used in LBP
    numImgs = len(imgs)
    radius = 2
    numPoints = 24
    hist_lbp = np.ones((numImgs, numPoints+2))
    
    for i, img in enumerate(imgs):
        img = local_binary_pattern(img, numPoints, radius, "uniform")
        (hist, _) = np.histogram(img.ravel(), bins=np.arange(0, numPoints + 3),range=(0, numPoints + 2))
        hist = hist.astype("float")
        hist /= hist.sum()
        hist_lbp[i,:] = hist

    return hist_lbp, lbs

def grayscale():
    '''
    Converts all images into grayscale

    Returns:
        - imgs : Entire dataset of grayscale images
        - labels : Labels of entire dataset
    '''

    # Extracting labels
    basedir = './Datasets/dataset/A/'
    labels_file = open(os.path.join(basedir,'labels.csv'), 'r')
    lines = labels_file.readlines()
    gender_labels = {line.split(',')[0] : int(line.split(',')[2]) for line in lines[1:]}

    imgs = []
    all_labels = []
    dirA1 = os.path.join(basedir,'img/')

    # Iterating over each image and converting it to grayscale
    for filename in sorted(os.listdir(dirA1), key = lambda x : int(x[:-4])):

        img = np.array(Image.open(os.path.join(dirA1,filename)).convert('L'))
        imgs.append(img)
        all_labels.append(gender_labels[filename[:-4]])
    
    labels = np.array(all_labels)
    return imgs, labels


def dimensionality_reductionLBP(train_data, test_data):
    '''
    Scales train and test datasets
    Implements Principal Component Analysis (PCA) on both datasets

    Keyword arguments:
        - train_data : Raw train dataset of LBP
        - test_data : Raw test dataset of LBP

    Returns:
        - train_pca : Train dataset of LBP after PCA
        - test_pca : Train dataset of LBP after PCA
    '''

    # Scaling datasets
    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    # Applying PCA to datasets
    # 'mle' algorithm not used since n_components > n_features
    pca = PCA(n_components = 0.8, svd_solver = 'full')
    pca.fit(train_data)
    train_pca = pca.transform(train_data)
    test_pca = pca.transform(test_data)

    return train_pca, test_pca


# ======================================================================================================================
# Data Pre-Processing for pre-trained VGG model
# ======================================================================================================================

def mainA1VGG():
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

    data_dir = './Datasets/dataset/A1_CNN/'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val','test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val','test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return dataloaders, dataset_sizes