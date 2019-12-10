from __future__ import print_function, division
import torchvision.models as models
import torch.nn as nn
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils, datasets, models
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
plt.ion()   # interactive mode


class A2:

    def __init__(self):
        # Current Approach: Transfer Learning using pre-trained VGG network

        self.model = models.vgg16(pretrained=True)
        # Replace default classifier with new classifier
        classifier_input = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(classifier_input, 2)
        self.criterion = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            self.model.cuda()

    def train(self, train_dl, sizes):

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(self.model.parameters(), lr=0.002, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
        num_workers = 0
        self.model, acc = self.train_model(train_dl, sizes, optimizer_ft, exp_lr_scheduler, num_epochs=25)
        return acc

    def train_model(self, train_dl, sizes, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        best_epoch = -1

        train_on_gpu = torch.cuda.is_available()
        if train_on_gpu:
            print('using gpu')
            self.model.cuda()

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in train_dl[phase]:
                    if train_on_gpu:
                        inputs, labels = inputs.cuda(), labels.cuda()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / sizes[phase]
                epoch_acc = running_corrects.double() / sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        print('Best epoch: {:4f}'.format(best_epoch))

        # Load best model weights as the model
        # Best model weights assumed to be the ones with highest validation accuracy
        self.model.load_state_dict(best_model_wts)
        return self.model, best_acc * 100


    def test(self, test_dl):

        test_loss = 0.
        correct = 0.
        total = 0.

        train_on_gpu = torch.cuda.is_available()
        if train_on_gpu:
            print('using gpu')
            self.model.cuda()

        for batch_idx, (data, target) in enumerate(test_dl):

                # move to GPU
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
                
        print('Test Loss: {:.6f}\n'.format(test_loss))

        test_accuracy = 100. * correct / total
        print('\nTest Accuracy: %2d%% (%2d/%2d)' % (test_accuracy, correct, total))
        return test_accuracy