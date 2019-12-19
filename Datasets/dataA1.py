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
import Datasets.dataset.landmarksA1 as landmarks


def mainA1Landmarks():
    tr_data, tr_lbs, te_data, te_lbs = landmarksA1()
    data_train = tr_data.reshape(tr_data.shape[0], tr_data.shape[1]*tr_data.shape[2])
    data_test = te_data.reshape(te_data.shape[0], te_data.shape[1]*te_data.shape[2])
    
    pca_train, pca_test = dimensionality_reduction(data_train, data_test)
    return pca_train, pca_test, tr_lbs, te_lbs

def landmarksA1():
    imgs, lbs = landmarks.extract_features_labels()
    tr_data, te_data, tr_lbs, te_lbs = train_test_split(imgs, lbs, test_size=0.2)
    return tr_data, tr_lbs, te_data, te_lbs

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

def mainA1LBP():
    imgs, lbs = extract_lbp()
    data_train, data_test, lbs_tr, lbs_te = train_test_split(imgs, lbs, test_size=0.2)
    pca_train, pca_test = dimensionality_reductionLBP(data_train, data_test)
    return pca_train, pca_test, lbs_tr, lbs_te

def extract_lbp():
    imgs, lbs = grayscale()

    numImgs = len(imgs)
    numPixels = imgs[0].shape[0] * imgs[0].shape[1]

    lbp_imgs = np.ones((numImgs, numPixels))

    for i, img in enumerate(imgs):
        img = local_binary_pattern(img, 8, 1, "uniform")
        lbp_imgs[i,:] = img.reshape(1,-1)

    return lbp_imgs, lbs

def dimensionality_reductionLBP(train_dataset, test_dataset):
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

    pca = PCA(n_components = 0.8, svd_solver = 'full')

    pca.fit(train_dataset)
    train_dataset = pca.transform(train_dataset)
    test_dataset = pca.transform(test_dataset)

    time1 = time.time()
    print("PCA finished, it took: ", (time1-time0)/60, " min")

    return train_dataset, test_dataset


def grayscale():
    '''
    Converts all images into grayscale
    '''

    basedir = '/Datasets/dataset/Original Datasets/celeba/'
    print(os.getcwd())
    labels_file = open(os.getcwd() + os.path.join(basedir,'labels.csv'), 'r')
    lines = labels_file.readlines()
    gender_labels = {line.split(',')[0] : int(line.split(',')[2]) for line in lines[1:]}

    imgs = []
    all_labels = []

    dirA1 = os.path.join(basedir,'img/')

    # Iterating over images in a sorted order
    for filename in sorted(os.listdir(os.getcwd() + dirA1), key = lambda x : int(x[:-4])):

        img = np.array(Image.open(os.getcwd() + os.path.join(dirA1,filename)).convert('L'))
        imgs.append(img)
        all_labels.append(gender_labels[filename[:-4]])
    
    labels = np.array(all_labels)
    return imgs, labels
