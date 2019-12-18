from A1.A1 import A1
from Datasets import data_preprocessing
#from A2.A2 import A2
from B1.B1 import B1


import pandas as pd
import numpy as np

# ======================================================================================================================
# Data preprocessing
#data_train, data_val, data_test = data_preprocessing.split_dataset()

#gray_train, gray_test, gray_val, lbs_train, lbs_test, lbs_val = data_preprocessing.mainA1()
#gray_train, gray_test,lbs_train, lbs_test = data_preprocessing.mainA1()
#gray_train, gray_test,lbs_train, lbs_test = data_preprocessing.mainA1LBP()
#train_dl, test_dl, sizes = data_preprocessing.mainA2()

trainB1, testB1, trainlbsB1, testlbsB1 = data_preprocessing.mainB1()

# ======================================================================================================================
# Task A1
'''
model_A1 = A1()                 # Build model object.
acc_A1_train = model_A1.train(gray_train, lbs_train) # Train model based on the training set (you should fine-tune your model based on validation set.)
acc_A1_test = model_A1.test(gray_test, lbs_test)   # Test model based on the test set.
'''
acc_A1_train = 'TBD'
acc_A1_test = 'TBD'
#print("Traing and Testing finished!")
#print('TA1:{},{}'.format(acc_A1_train, acc_A1_test))
#Clean up memory/GPU etc...             # Some code to free memory if necessary.

# ======================================================================================================================
# Task A2
#model_A2 = A2()
#acc_A2_train = model_A2.train(train_dl, sizes)
#acc_A2_test = model_A2.test(test_dl)
#Clean up memory/GPU etc...
acc_A2_train = 'TBD'
acc_A2_test = 'TBD'

# ======================================================================================================================
# Task B1

model_B1 = B1()
acc_B1_train = model_B1.train(trainB1, trainlbsB1)
acc_B1_test = model_B1.test(testB1, testlbsB1)

#Clean up memory/GPU etc...
#acc_B1_train = 'TBD'
#acc_B1_test = 'TBD'

# ======================================================================================================================
# Task B2
'''
model_B2 = B2(args...)
acc_B2_train = model_B2.train(args...)
acc_B2_test = model_B2.test(args...)
'''
#Clean up memory/GPU etc...
acc_B2_train = 'TBD'
acc_B2_test = 'TBD'

# ======================================================================================================================
## Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A1_train = 'TBD'
# acc_A1_test = 'TBD'
