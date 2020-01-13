import pandas as pd
import numpy as np

from A1.A1VGG import A1
from A2.A2SVM import A2
from B1.B1VGG import B1
from B2.B2VGG import B2

from Datasets import dataA1, dataA2, dataB1, dataB2

# ======================================================================================================================
# Data Pre-Processing

dataA1, testdataA1, sizesA1 = dataA1.mainA1VGG()
dataA2_train, dataA2_test, dataA2_addtest, lbsA2_train, lbsA2_test, lbsA2_addtest = dataA2.mainA2Landmarks()
dataB1, testdataB1, sizesB1 = dataB1.mainB1CNN()
dataB2, testdataB2, sizesB2 = dataB2.mainB2CNN()

# ======================================================================================================================
# Task A1

modelA1 = A1()
acc_A1_train = modelA1.train(dataA1, sizesA1)
acc_A1_test = modelA1.test(testdataA1['test'])
acc_A1_addtest = modelA1.test(testdataA1['addtest'])

# Clean up memory
del modelA1, dataA1, testdataA1, sizesA1

'''
acc_A1_train = 'TBD'
acc_A1_test = 'TBD'
acc_A1_addtest = 'TBD'
'''

# ======================================================================================================================
# Task A2

modelA2 = A2()
acc_A2_train = modelA2.train(dataA2_train, lbsA2_train)
acc_A2_test = modelA2.test(dataA2_test, lbsA2_test)
acc_A2_addtest = modelA2.test(dataA2_addtest, lbsA2_addtest)

# Clean up memory
del modelA2, dataA2_train, dataA2_test, dataA2_addtest, lbsA2_train, lbsA2_test, lbsA2_addtest


'''
acc_A2_train = 'TBD'
acc_A2_test = 'TBD'
acc_A2_addtest = 'TBD'
'''

# ======================================================================================================================
# Task B1

modelB1 = B1()
acc_B1_train = modelB1.train(dataB1, sizesB1)
acc_B1_test = modelB1.test(testdataB1['test'])
acc_B1_addtest = modelB1.test(testdataB1['addtest'])

# Clean up memory
del modelB1, dataB1, testdataB1, sizesB1

'''
acc_B1_train = 'TBD'
acc_B1_test = 'TBD'
acc_B1_addtest = 'TBD'
'''

# ======================================================================================================================
# Task B2

modelB2 = B2()
acc_B2_train = modelB2.train(dataB2, sizesB2)
acc_B2_test = modelB2.test(testdataB2['test'])
acc_B2_addtest = modelB2.test(testdataB2['addtest'])

# Clean up memory
del modelB2, dataB2, testdataB2, sizesB2

'''
acc_B2_train = 'TBD'
acc_B2_test = 'TBD'
acc_B2_addtest = 'TBD'
'''

# ======================================================================================================================
## Print out your results with following format:
print('TA1:{},{},{};TA2:{},{},{};TB1:{},{},{};TB2:{},{},{};'.format(acc_A1_train, acc_A1_test, acc_A1_addtest,
                                                            acc_A2_train, acc_A2_test, acc_A2_addtest,
                                                            acc_B1_train, acc_B1_test, acc_B1_addtest,
                                                            acc_B2_train, acc_B2_test, acc_B2_addtest))
