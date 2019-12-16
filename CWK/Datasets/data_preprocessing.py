# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:25:58 2019
@author: Pedro
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from PIL import Image
import glob, os
import matplotlib.pyplot as plt 
from os import listdir
from matplotlib import image
import numpy as np
#import cv2
import time
#import torch
#from torchvision import transforms, utils, datasets, models
#from torch.utils.data import Dataset, DataLoader
import Datasets.dataset.lab2_landmarks as import_data

def mainA1():
    tr_data, tr_lbs, te_data, te_lbs = landmark_computation()
    data_train = tr_data.reshape(tr_data.shape[0], tr_data.shape[1]*tr_data.shape[2])
    data_test = te_data.reshape(te_data.shape[0], te_data.shape[1]*te_data.shape[2])
    
    pca_train, pca_test = dimensionality_reduction(data_train, data_test)
    return pca_train, pca_test, tr_lbs, te_lbs

def landmark_computation():
    imgs, lbs = import_data.extract_features_labels()
    tr_data, te_data, tr_lbs, te_lbs = train_test_split(imgs, lbs, test_size=0.2)
    return tr_data, tr_lbs, te_data, te_lbs

def dimensionality_reduction(train_dataset, test_dataset):
    '''
    Scales the data and performs Principal Component 
    Analysis (PCA) on a given dataset
    '''

    print("Dimensionality reduction started!")
    time0 = time.time()

    scaler = StandardScaler()
    scaler.fit(train_dataset)
    train_dataset = scaler.transform(train_dataset)
    test_dataset = scaler.transform(test_dataset)

    pca = PCA(n_components = 'mle', svd_solver = 'full')

    pca.fit(train_dataset)
    train_dataset = pca.transform(train_dataset)
    test_dataset = pca.transform(test_dataset)

    time1 = time.time()
    print("PCA finished, it took: ", (time1-time0)/60, " min")

    return train_dataset, test_dataset

'''
def mainA2():
    
    #Loads data into dataloaders for model training, validation and
    testing of task A2, performing data augmentation on training
    data and pre-processing all data (normalization and re-sizing)
    

    # Data normalization for training
    data_transforms = {
        'train': transforms.Compose([
            transforms.CenterCrop(178),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(178),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    data_dir = 'Datasets/dataset/A2/70 20 10 Split/'
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
'''


'''
DELETED CODE
--
Might need until the end of the project

def pixels_differences(images):
    ''''''
    Computes the difference between each pixel in an image
    and the other pixels in that same image, thus building a feature
    ''''''

    numPixels = images[0].shape[0]*images[0].shape[1]
    feature = np.zeros(shape=(len(images), numPixels))
    mask = np.arange(numPixels) != np.arange(numPixels)[:,None]

    for i, image in enumerate(images):
        ret = image.ravel()
        ret = ret[:,None] - ret
        feature[i,:] = ret[np.where(mask)]

    return feature

def process_labelsA1():
    ''''''
    Processes and orders labels for task A1
    ''''''

    curDir = os.getcwd() + '/Datasets/dataset/Labels for Train Test/'

    df_train = pd.read_csv(curDir + 'trainlabelsA.csv', index_col=0)
    df_train.sort_index()
    lbs_train = df_train['gender'].to_numpy().reshape(-1,)

    df_test= pd.read_csv(curDir + 'testlabelsA.csv', index_col=0)
    df_test.sort_index()
    lbs_test = df_test['gender'].to_numpy().reshape(-1,)

    #df_val = pd.read_csv(curDir + 'vallabelsA.csv', index_col=0)
    #df_val.sort_index()
    #lbs_val = df_val['gender'].to_numpy().reshape(-1,)

    #return lbs_train, lbs_test, lbs_val
    return lbs_train, lbs_test

def color_to_gray(images):
    ''''''
    Converts image to grayscale
    ''''''

    num_pixels = images[0].shape[0]*images[1].shape[1]
    grey_imgs = np.zeros(shape=(len(images),num_pixels))

    for i, image in enumerate(images):
         # Shape: (218, 178)
         # Since gray image only has 1 channel
        grey_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        grey_imgs[i,:] = np.array(grey_img).reshape(1,num_pixels)

    return grey_imgs

def image_processing():
    ''''''
    Load all images of a directory into 3 datasets: train, test and val
    ''''''

    dirA1 = '/Datasets/dataset/A1/Train Test Split/'
    phases = ['train','test']
    datasets = dict()

    # Iterating over Train/Test/Val
    for folder in phases:
        curFolder = os.getcwd() + dirA1 + folder + '/'

        datasets[str(folder)] = []

        # Iterating over images in a sorted order
        for filename in sorted(os.listdir(curFolder), key = lambda x : int(x[:-4])):

            img = np.array(Image.open(curFolder + filename).convert('L'))
            datasets[str(folder)].append(img)

    return datasets

def reshape_images(images):

    numPixels = images[0].shape[0]*images[0].shape[1]
    dataset = np.zeros(shape=(len(images),numPixels))

    for i, image in enumerate(images):
        dataset[i,:] = image.reshape(1,-1)

    return dataset


def process_images():
    ''''''
    Load all images of a directory into 3 datasets: train, test and val
    ''''''

    dirA1 = '/Datasets/dataset/A1/Train Test Split/'
    #phases = ['train','test','val']
    phases = ['train','test']
    datasets = dict()

    # Iterating over Train/Test/Val
    for folder in phases:
        curFolder = os.getcwd() + dirA1 + folder + '/'

        datasets[str(folder)] = []

        # Iterating over images in a sorted order
        for filename in sorted(os.listdir(curFolder), key = lambda x : int(x[:-4])):

            # Load image as numpy array
            # Shape: (218, 178, 3)
            img_data = image.imread(curFolder + filename)
            datasets[str(folder)].append(img_data)

    return datasets

'''