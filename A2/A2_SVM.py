from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

import numpy as np

import time

class A2:

    def __init__(self):
        # Define support vector classifier
        self.svm = SVC(C=10, kernel='rbf', gamma=0.001, probability=True, random_state=42)

    def train(self, data_train, lbs_train):
        '''
        Method to train the model using training data, as well as
        fine-tuning the model using Cross-Validation
        '''

        t0 = time.time()

        # Fitting model
        self.svm.fit(data_train, lbs_train)

        '''
        ## Perform Cross-Validation
        cv_results = cross_validate(self.svm, data_train, lbs_train, cv=3, return_estimator=True)
        accuracies = cv_results['test_score']
        print("Mean Validation Accuracy: %0.2f (+/- %0.2f)" % (accuracies.mean(), accuracies.std() * 2))
        svms = cv_results['estimator']
        self.svm = svms[-1]
        '''

        predictions = self.svm.predict(data_train)
        train_accuracy = accuracy_score(lbs_train, predictions) * 100

        t1 = time.time()
        print("Time taken for training:  %.2f s" % round(t1-t0,2))
        print("Train Accuracy: %.2f " % round(train_accuracy,2))
        
        return round(train_accuracy,2)


    def test(self, data_test, lbs_test):
        '''
        Method to test the built model using testing data
        '''

        t0 = time.time()
        print("\nTesting started!")

        # Obtaining predicted labels from testing
        pred_test = self.svm.predict(data_test)

        # Calculate accuracy
        test_accuracy = accuracy_score(lbs_test, pred_test) * 100

        t1 = time.time()
        print("Time taken for testing:  %.2f s" % round(t1-t0,2))
        print("Test Accuracy: %.2f " % round(test_accuracy,2))

        return round(test_accuracy,2)