import torch
import torchvision
import torch.nn as nn
from torch.optim import lr_scheduler, SGD
from torchvision import utils, datasets, models

from HelperFunctions.cnnMethods import train_cnn, test_cnn
from HelperFunctions.curvesCNN import plot_acc_curve, plot_loss_curve


class B1:

    def __init__(self):
        '''
        Imports the pre-trained VGG network to be used in this task
        Makes changes to its fully-connected layer so that it can only return 2 classes
        '''

        # Define B1 hyperparameters to be used
        LEARNING_RATE = 0.00002
        STEP_SIZE = 5
        DECAYING_FACTOR = 0.9
        self.NUM_EPOCHS = 15

        # Importing & Changing pre-trained VGG model
        self.model = models.vgg16(pretrained=True)
        classifier_input = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(classifier_input, 2)

        # Defining loss function as Cross Entropy Loss
        self.criterion = nn.CrossEntropyLoss()

        # Defining Optimizer as Stochastic Gradient Descent (SGD)
        self.optimizer = SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=0.9)

        # Defining a Step Scheduler of the LR
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=STEP_SIZE, gamma=DECAYING_FACTOR)

        if torch.cuda.is_available():
            self.model.cuda()


    def train(self, dataloader, sizes):
        '''
        Trains the CNN model for a number of epochs
        Plots 2 learning curves showing the evolution of accuracy
        and loss at each epoch of training

        Keyword arguments:
            - dataloader : torch DataLoader with train, val and test datasets
            - sizes : Array of 2 elements with size of train and val datasets

        Returns:
            - best_train_acc : Training ccuracy of the best epoch in %
        '''

        self.model, best_train_acc, accs, losses = train_cnn(self.model, dataloader, sizes, self.NUM_EPOCHS, self.criterion, self.optimizer, self.scheduler)

        # Plotting learning curves
        plot_loss_curve(losses, self.NUM_EPOCHS, 'Train and Validation Losses in Task B1 (Pre-trained VGG)')
        plot_acc_curve(accs, self.NUM_EPOCHS, 'Train and Validation Accuracies in Task B1 (Pre-trained VGG)')

        return best_train_acc


    def test(self, dataloader):
        '''
        Test the CNN model on previously unseen data

        Keyword arguments:
            - dataloader : torch DataLoader with train, val and test datasets

        Returns:
            - test_acc : Testing accuracy of the model in %
        '''

        test_acc = test_cnn(self.model, dataloader, self.criterion)
        return test_acc