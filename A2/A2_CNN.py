import os, time, copy, sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
from torch.optim import lr_scheduler, SGD
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models

sys.path.append('../Datasets/')
from learning_curve_CNN import plot_acc_curve, plot_loss_curve


class A2:

    def __init__(self):
        '''
        Imports the pre-trained VGG network to be used in this task
        Makes changes to its fully-connected layer so that it can only return 2 classes
        '''

        # Define hyperparameters to be used
        LEARNING_RATE = 0.002
        STEP_SIZE = 10
        DECAYING_FACTOR = 0.1

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


    def train(self, dataloader, sizes, num_epochs=25):
        '''
        Trains the CNN model for a number of epochs
        Loads the final model as being the one with the highest validation accuracy
        Plots 2 learning curves showing the evolution of accuracy
        and loss at each epoch of training

        Keyword arguments:
            - dataloader : torch DataLoader with train, val and test datasets
            - sizes : Array of 2 elements with size of train and val datasets

        Returns:
            - best_train_acc : Training ccuracy of the best epoch in %
        '''

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        prev_acc = 0.0

        # Initializing arrays to keep track of loss 
        # and accuracy at each epoch
        losses = dict()
        losses['train'] = np.ones(num_epochs)
        losses['val'] = np.ones(num_epochs)
        accs = dict()
        accs['train'] = np.ones(num_epochs)
        accs['val'] = np.ones(num_epochs)

        train_on_gpu = torch.cuda.is_available()
        if train_on_gpu:
            print('using gpu')
            self.model.cuda()

        for epoch in range(num_epochs):

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloader[phase]:
                    if train_on_gpu:
                        inputs, labels = inputs.cuda(), labels.cuda()

                    # Zero parameter gradients
                    self.optimizer.zero_grad()

                    # Forward propagation
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # Implementing backward propagation + Optimization
                        # if in train phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # Computing statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    self.scheduler.step()

                # Computing loss and accuracy at the epoch
                epoch_loss = running_loss / sizes[phase]
                losses[phase][epoch] = epoch_loss
                epoch_acc = running_corrects.double() / sizes[phase]
                accs[phase][epoch] = epoch_acc

                if phase == 'train':
                    prev_acc = epoch_acc

                # Save the best mmodel so far
                if phase == 'val' and epoch_acc > best_acc:
                    best_train_acc = prev_acc
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

        # Load best model weights as the model
        # Best model weights assumed to be the ones with highest validation accuracy
        self.model.load_state_dict(best_model_wts)

        # Plotting learning curves
        plot_loss_curve(losses, num_epochs, 'Train and Validation Losses in Task A2 (Pre-trained VGG)')
        plot_acc_curve(accs, num_epochs, 'Train and Validation Accuracies in Task A2 (Pre-trained VGG)')

        return round(best_train_acc * 100.,2)


    def test(self, dataloader):
        '''
        Test the CNN model on previously unseen data

        Keyword arguments:
            - dataloader : torch DataLoader with train, val and test datasets

        Returns:
            - test_acc : Testing accuracy of the model in %
        '''

        test_loss = 0.
        correct = 0.
        total = 0.

        train_on_gpu = torch.cuda.is_available()
        if train_on_gpu:
            self.model.cuda()

        for batch_idx, (data, target) in enumerate(dataloader['test']):

                # Move to GPU
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()

                # Forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data)

                # Calculate loss
                loss = self.criterion(output, target)

                # Update average test loss 
                test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))

                # Convert output probabilities to predicted class
                pred = output.data.max(1, keepdim=True)[1]

                # Compare predictions to true label
                correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
                total += data.size(0)

        test_acc = 100. * correct / total
        return round(test_acc,2)