import sys, time
import numpy as np
import pandas as pd

from A1.A1FL import A1 as A1_FL
from A1.A1LBP import A1 as A1_LBP
from A1.A1VGG import A1 as A1_VGG

from Datasets import dataA1

# ======================================================================================================================
# Data Pre-Processing

dataA1_train1, dataA1_test1, dataA1_addtest1, lbsA1_train1, lbsA1_test1, lbsA1_addtest1 = dataA1.mainA1Landmarks()
dataA1_train2, dataA1_test2, dataA1_addtest2, lbsA1_train2, lbsA1_test2, lbsA1_addtest2 = dataA1.mainA1LBP()
dataA1_CNN, testdataA1_CNN, dataset_sizes = dataA1.mainA1VGG()

# ======================================================================================================================
# Task A1 with Facial Landmarks as Feature
model1_A1 = A1_FL()
acc1_A1_train = model1_A1.train(dataA1_train1, lbsA1_train1)
acc1_A1_test = model1_A1.test(dataA1_test1, lbsA1_test1)
acc1_A1_addtest = model1_A1.test(dataA1_addtest1, lbsA1_addtest1)

# Clean up memory
del model1_A1, dataA1_train1, dataA1_test1, dataA1_addtest1, lbsA1_train1, lbsA1_test1, lbsA1_addtest1

'''
acc1_A1_train = 'TBD'
acc1_A1_test = 'TBD'
acc1_A1_addtest = 'TBD'
'''

# ======================================================================================================================
# Task A1 with LBP as Feature

model2_A1 = A1_LBP()
acc2_A1_train = model2_A1.train(dataA1_train2, lbsA1_train2)
acc2_A1_test = model2_A1.test(dataA1_test2, lbsA1_test2)
acc2_A1_addtest = model2_A1.test(dataA1_addtest2, lbsA1_addtest2)

# Clean up memory
del model2_A1, dataA1_train2, dataA1_test2, dataA1_addtest2, lbsA1_train2, lbsA1_test2, lbsA1_addtest2

'''
acc2_A1_train = 'TBD'
acc2_A1_test = 'TBD'
acc2_A1_addtest = 'TBD'
'''

# ======================================================================================================================
# Task A1 with pre-trained VGG model

model3_A1 = A1_VGG()
acc3_A1_train = model3_A1.train(dataA1_CNN, dataset_sizes)
acc3_A1_test = model3_A1.test(testdataA1_CNN['test'])
acc3_A1_addtest = model3_A1.test(testdataA1_CNN['addtest'])

# Clean up memory
del model3_A1, dataA1_CNN, testdataA1_CNN, dataset_sizes

'''
acc3_A1_train = 'TBD'
acc3_A1_test = 'TBD'
acc3_A1_addtest = 'TBD'
'''

# ======================================================================================================================
## Printing results

print('TA1_1:{},{},{};TA1_2:{},{},{};TA1_3:{},{},{};'.format(acc1_A1_train, acc1_A1_test, acc1_A1_addtest,
                                                    acc2_A1_train, acc2_A1_test, acc2_A1_addtest,
                                                    acc3_A1_train, acc3_A1_test, acc3_A1_addtest))

