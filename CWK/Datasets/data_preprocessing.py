# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:25:58 2019
@author: Pedro
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import glob, os
import matplotlib.pyplot as plt 
from os import listdir
from matplotlib import image
import numpy as np
import cv2

def mainA1():
    images = process_images()
    gray_train = color_to_gray(images['train'])
    gray_test = color_to_gray(images['test'])
    gray_val = color_to_gray(images['val'])

    lbs_train, lbs_test, lbs_val = process_labelsA1()

    return gray_train, gray_test, gray_val, lbs_train, lbs_test, lbs_val

def process_images():
    '''
    Load all images of a directory into 3 datasets: train, test and val
    '''

    dirA1 = '/Datasets/dataset/A1/'
    phases = ['train','test','val']
    datasets = dict()

    # Iterating over Train/Test/Val
    for folder in phases:
        curFolder = os.getcwd() + dirA1 + folder + '/'
        os.listdir(curFolder).sort()

        datasets[str(folder)] = []

        # Iterating over images
        for filename in os.listdir(curFolder).sort():

            # Load image as numpy array
            # Shape: (218, 178, 3)
            print(filename)
            img_data = image.imread(curFolder + filename)
            datasets[str(folder)].append(img_data)

    return datasets


def color_to_gray(images):
    '''
    Converts image to grayscale
    '''

    num_pixels = images[0].shape[0]*images[1].shape[1]
    grey_imgs = np.zeros(shape=(len(images),num_pixels))


    i = 0
    for image in images:
         # Shape: (218, 178)
         # Since gray image only has 1 channel
        grey_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        grey_imgs[i,:] = np.array(grey_img).reshape(1,num_pixels)

    return grey_imgs

def process_labelsA1():
    '''
    Processes and orders labels for task A1
    '''

    curDir = os.getcwd() + '/Datasets/dataset/'

    df_train = pd.read_csv(curDir + 'trainlabelsA.csv')
    df_train.sort_values(by='img_name')
    lbs_train = df_train['gender'].to_numpy().reshape(-1,)

    df_test= pd.read_csv(curDir + 'testlabelsA.csv')
    df_test.sort_values(by='img_name')
    lbs_test = df_test['gender'].to_numpy().reshape(-1,)

    df_val = pd.read_csv(curDir + 'vallabelsA.csv')
    df_val.sort_values(by='img_name')
    lbs_val = df_val['gender'].to_numpy().reshape(-1,)

    return lbs_train, lbs_test, lbs_val

#def feature_extractor(datasets):
    '''
    Obtains dataset with different features extraction methods
    '''

