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

def main():
    images = process_images()
    print(images.size)

def process_images():
    '''
    Load all images of a directory into a dataset
    '''

    dirA1 = '/Datasets/dataset/A1/'
    datasets = dict()

    for folder in listdir(os.getcwd() + dirA1):
        datasets[str(folder)] = []
        for filename in folder:
            # load image
               img_data = image.imread(os.getcwd() + dirA1 + folder + filename)
               datasets[str(folder)].append(img_data)
        print(folder)
        print(datasets[str(folder)].size)

    return datasets

