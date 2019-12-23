import sys,os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate

from Datasets.plot_curve import plot_learning_curve


class A1:
    '''
    Class responsible for initializing, training and testing the SVM model
    with facial landmarks as features.
    '''

    def __init__(self):
        '''
        Initializes the SVM to be used in this task, with
        parameters found in Grid Search CV (file: hyperparametersA1model1.ipynb)
        '''

        # Best parameters for Facial landmarks model 
        self.svm = SVC(C=30, kernel='rbf', gamma=0.001, probability=True, random_state=42)


    def train(self, data_train, lbs_train):
        '''
        Trains the model using training data, using a previously
        defined SVM (in init function)

        Keyword arguments:
            - data_train : Training dataset
            - lbs_train : Labels for training dataset

        Returns:
            - train_accuracy : Training accuracy of model in %
        '''

        # Fitting model
        self.svm.fit(data_train, lbs_train)

        ## To perform Cross-Validation : Uncomment this block
        '''
        cv_results = cross_validate(SVC(C=10, kernel='rbf', gamma=0.001, probability=True, random_state=42), data_train, lbs_train, cv=3, return_estimator=True)
        accuracies = cv_results['test_score']
        print("Mean Validation Accuracy: %0.2f (+/- %0.2f)" % (accuracies.mean(), accuracies.std() * 2))
        '''

        # Computing training accuracy
        predictions = self.svm.predict(data_train)
        train_accuracy = accuracy_score(lbs_train, predictions) * 100

        # Obtain learning curve (Saves it in main directory)
        plot_learning_curve(SVC(C=30, kernel='rbf', gamma=0.001, probability=True, random_state=42),"Learning Curve for A1 Task (SVM with facial landmarks)", data_train, lbs_train,cv=5)

        return round(train_accuracy,2)


    def test(self, data_test, lbs_test):
        '''
        Tests the built model using previously unseen testing data

        Keyword arguments:
            - data_test : Testing dataset
            - lbs_test : Labels for testing dataset

        Returns:
            - test_accuracy : Testing accuracy of model in %
        '''

        # Obtaining predicted labels from testing
        pred_test = self.svm.predict(data_test)

        # Calculate accuracy
        test_accuracy = accuracy_score(lbs_test, pred_test) * 100

        return round(test_accuracy,2)