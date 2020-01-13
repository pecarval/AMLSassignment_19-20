import sys, time
import numpy as np
import pandas as pd

from A2.A2SVM import A2 as A2_SVM
from A2.A2VGG import A2 as A2_VGG

from Datasets import dataA2

# ======================================================================================================================
# Data Pre-Processing

dataA2_train1, dataA2_test1, dataA2_addtest1, lbsA2_train1, lbsA2_test1, lbsA2_addtest1 = dataA2.mainA2Landmarks()
dataA2_CNN, testdataA2_CNN, dataset_sizes = dataA2.mainA2VGG()


# ======================================================================================================================
# Task A2 with Facial Landmarks as Feature

model1_A2 = A2_SVM()
acc1_A2_train = model1_A2.train(dataA2_train1, lbsA2_train1)
acc1_A2_test = model1_A2.test(dataA2_test1, lbsA2_test1)
acc1_A2_addtest = model1_A2.test(dataA2_addtest1, lbsA2_addtest1)

# Clean up memory
del model1_A2, dataA2_train1, dataA2_test1, dataA2_addtest1, lbsA2_train1, lbsA2_test1, lbsA2_addtest1

'''
acc1_A2_train = 'TBD'
acc1_A2_test = 'TBD'
acc1_A2_addtest = 'TBD'
'''

# ======================================================================================================================
# Task A2 with pre-trained VGG model

model2_A2 = A2_VGG()
acc2_A2_train = model2_A2.train(dataA2_CNN, dataset_sizes)
acc2_A2_test = model2_A2.test(testdataA2_CNN['test'])
acc2_A2_addtest = model2_A2.test(testdataA2_CNN['addtest'])

# Clean up memory
del model2_A2, dataA2_CNN, testdataA2_CNN, dataset_sizes

'''
acc2_A2_train = 'TBD'
acc2_A2_test = 'TBD'
acc2_A2_addtest = 'TBD'
'''

# ======================================================================================================================
## Printing results

print('TA2_1:{},{},{};TA2_2:{},{},{};'.format(acc1_A2_train, acc1_A2_test ,acc1_A2_addtest,
                                        acc2_A2_train, acc2_A2_test, acc2_A2_addtest))

