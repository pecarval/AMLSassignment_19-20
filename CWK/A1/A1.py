from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

import time

class A1:

    def __init__(self):
        # Define support vector classifier
        self.svm = SVC(kernel='linear', probability=True, random_state=42)


    def train(self, data_train, lbs_train):
        '''
        Method to train the model using training data, as well as
        fine-tuning the model using Cross-Validation
        '''

        t0 = time.time()
        print("\nTraining started!")

        # Fitting model
        accuracies = cross_val_score(self.svm, data_train, lbs_train, cv=5)
        print('Maximum accuracy achieved with CV: ', accuracies[4])

        t1 = time.time()
        print("Time taken for training: ", t1-t0, "s")


    def test(self, data_test, lbs_test):
        '''
        Method to test the built model using testing data
        '''

        t0 = time.time()
        print("\nTesting started!")

        # Obtaining predicted labels from testing
        pred_test = self.svm.predict(data_test)

        # Calculate accuracy
        accuracy = accuracy_score(lbs_test, pred_test)
        print('Model accuracy is: ', accuracy)

        t1 = time.time()
        print("Time taken for testing: ", t1-t0, "s")
