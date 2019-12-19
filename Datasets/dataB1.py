# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:12:35 2019
@author: Pedro
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern
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
import Datasets.dataset.landmarks_celeba as landmarks_celeba
import Datasets.dataset.landmarks_cartoon as landmarks_cartoon


def mainB1():
    start = time.time()
    tr_data, tr_lbs, te_data, te_lbs = landmarksB1()
    print("Landmark computation took ", (time.time()-start)/60, "min")
    data_train = tr_data.reshape(tr_data.shape[0], tr_data.shape[1]*tr_data.shape[2])
    data_test = te_data.reshape(te_data.shape[0], te_data.shape[1]*te_data.shape[2])
    
    pca_train, pca_test = dimensionality_reduction(data_train, data_test)
    return pca_train, pca_test, tr_lbs, te_lbs

def dimensionality_reduction(train_dataset, test_dataset):
    '''
    Scales the data and performs Principal Component 
    Analysis (PCA) on a given dataset
    '''

    print("Pre-PCA Train shape: ", train_dataset.shape)
    print("Pre-PCA Test shape: ", test_dataset.shape)

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

    print("Post-PCA Train shape: ", train_dataset.shape)
    print("Post-PCA Test shape: ", test_dataset.shape)

    return train_dataset, test_dataset

def landmarksB1():
    imgs, lbs = landmarks_cartoon.extract_features_labels()
    print(imgs.shape)
    print(lbs.shape)
    tr_data, te_data, tr_lbs, te_lbs = train_test_split(imgs, lbs, test_size=0.2)
    return tr_data, tr_lbs, te_data, te_lbs

