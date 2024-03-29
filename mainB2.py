import sys
import numpy as np
import pandas as pd

from B2.B2VGG import B2 as B2_VGG
from B2.B2ResNet import B2 as B2_ResNet

from Datasets import dataB2

# ======================================================================================================================
# Data Pre-Processing

dataB2_CNN, testdataB2_CNN, dataset_sizes = dataB2.mainB2CNN()

# ======================================================================================================================
# Task B2 with pre-trained VGG model

model1_B2 = B2_VGG()
acc1_B2_train = model1_B2.train(dataB2_CNN, dataset_sizes)
acc1_B2_test = model1_B2.test(testdataB2_CNN['test'])
acc1_B2_addtest = model1_B2.test(testdataB2_CNN['addtest'])

# Clean up memory
del model1_B2

'''
acc1_B2_train = 'TBD'
acc1_B2_test = 'TBD'
acc1_B2_addtest = 'TBD'
'''

# ======================================================================================================================
# Task B2 with pre-trained ResNet model

model2_B2 = B2_VGG()
acc2_B2_train = model2_B2.train(dataB2_CNN, dataset_sizes)
acc2_B2_test = model2_B2.test(testdataB2_CNN['test'])
acc2_B2_addtest = model2_B2.test(testdataB2_CNN['addtest'])

# Clean up memory
del model2_B2, dataB2_CNN, testdataB2_CNN, dataset_sizes

'''
acc2_B2_train = 'TBD'
acc2_B2_test = 'TBD'
acc2_B2_addtest = 'TBD'
'''

# ======================================================================================================================
## Printing results

print('TB2_1:{},{},{};TB2_2:{},{},{};'.format(acc1_B2_train, acc1_B2_test, acc1_B2_addtest,
                                        acc2_B2_train, acc2_B2_test, acc2_B2_addtest))

