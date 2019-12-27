import sys
import numpy as np
import pandas as pd

from B1.B1FL import B1 as B1_FL
from B1.B1VGG import B1 as B1_VGG
from B1.B1ResNet import B1 as B1_ResNet

from Datasets import dataB1

# ======================================================================================================================
# Data Pre-Processing

dataB1_train1, dataB1_test1, lbsB1_train1, lbsB1_test1 = dataB1.mainB1Landmarks()
dataB1_CNN, dataset_sizes = dataB1.mainB1CNN()


# ======================================================================================================================
# Task B1 with Facial Landmarks as Feature

model1_B1 = B1_FL()
acc1_B1_train = model1_B1.train(dataB1_train1, lbsB1_train1)
acc1_B1_test = model1_B1.test(dataB1_test1, lbsB1_test1)

# Clean up memory
del model1_B1, dataB1_train1, dataB1_test1, lbsB1_train1, lbsB1_test1

#acc1_B1_train = 'TBD'
#acc1_B1_test = 'TBD'


# ======================================================================================================================
# Task B1 with pre-trained VGG model
'''
model2_B1 = B1_VGG()
acc2_B1_train = model2_B1.train(dataB1_CNN, dataset_sizes)
acc2_B1_test = model2_B1.test(dataB1_CNN)
'''
# Clean up memory
#del model2_B1

acc2_B1_train = 'TBD'
acc2_B1_test = 'TBD'

# ======================================================================================================================
# Task B1 with pre-trained ResNet model
'''
model3_B1 = B1_ResNet()
acc3_B1_train = model3_B1.train(dataB1_CNN, dataset_sizes)
acc3_B1_test = model3_B1.test(dataB1_CNN)
'''
# Clean up memory
#del model3_B1, dataB1_3, dataset_sizes

acc3_B1_train = 'TBD'
acc3_B1_test = 'TBD'


# ======================================================================================================================
## Printing results

print('TB1_1:{},{};TB1_2:{},{};TB1_3:{},{};'.format(acc1_B1_train, acc1_B1_test,
                                                    acc2_B1_train, acc2_B1_test,
                                                    acc3_B1_train, acc3_B1_test))

