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

'''
CNN training and testing methods are adapted from:

"Transfer Learning for Computer Vision Tutorial — PyTorch Tutorials 1.3.1 documentation", 
Pytorch.org, 2020. [Online]. Available: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html. 
[Accessed: 13- Jan- 2020].
'''

def train_cnn(model, dataloader, sizes, num_epochs, criterion, optimizer, scheduler):
    '''
    Trains the CNN model for a number of epochs
    Loads the final model as being the one with the highest validation accuracy

    Keyword arguments:
        - model : CNN model to-be-trained
        - dataloader : torch DataLoader with train, val and test datasets
        - sizes : Array of 2 elements with size of train and val datasets
        - num_epochs : Number of epochs for training
        - criterion : Loss function to be used in model training
        - optimizer : Optimizer function to be used in model training
        - scheduler : LR scheduler to be used in  model training

    Returns:
        - model : Trained CNN model
        - best_train_acc : Training accuracy of the best epoch in %
        - accs : Dictionary of 2 arrays (for training & validation) of accuracy value at each epoch
        - losses : Dictionary of 2 arrays (for training & validation) of loss value at each epoch
    '''

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
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
        model.cuda()

    for epoch in range(num_epochs):

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader[phase]:
                if train_on_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                # Zero parameter gradients
                optimizer.zero_grad()

                # Forward propagation
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Implementing backward propagation + Optimization
                    # if in train phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Computing statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            # Computing loss and accuracy at the epoch
            epoch_loss = running_loss / sizes[phase]
            losses[phase][epoch] = epoch_loss
            epoch_acc = running_corrects.double() / sizes[phase]
            accs[phase][epoch] = epoch_acc

            if phase == 'train':
                scheduler.step()
                prev_acc = epoch_acc

            # Save the best mmodel so far
            if phase == 'val' and epoch_acc > best_val_acc:
                best_train_acc = prev_acc
                best_val_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    # Load best model weights as the model
    # Best model weights assumed to be the ones with highest validation accuracy
    model.load_state_dict(best_model_wts)

    return model, round(best_train_acc * 100.,2), accs, losses


def test_cnn(model, dataloader, criterion):
    '''
    Test the CNN model on previously unseen data

    Keyword arguments:
        - model : Trained CNN model
        - dataloader : torch DataLoader with train, val and test datasets
        - criterion : Criterion function to be used in testing

    Returns:
        - test_acc : Testing accuracy of the model in %
    '''

    test_loss = 0.
    correct = 0.
    total = 0.

    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        model.cuda()

    for batch_idx, (data, target) in enumerate(dataloader):

            # Move to GPU
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)

            # Calculate loss
            loss = criterion(output, target)

            # Update average test loss 
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))

            # Convert output probabilities to predicted class
            pred = output.data.max(1, keepdim=True)[1]

            # Compare predictions to true label
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            total += data.size(0)

    test_acc = 100. * correct / total
    return round(test_acc,2)