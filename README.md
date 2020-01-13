# ELEC0134 Applied Machine Learning Systems Assignment

This repository implements the ELEC0134 Applied Machine Learning Systems assignment. The assignment consists of four tasks:
* Task A1: Gender Detection
* Task A2: Emotion Detection
* Task B1: Face Shape Recognition
* Task B2: Eye Colour Recognition

Tasks A1 and A2 are binary classification problems.
Tasks B1 and B2 are multi-class classification problems, each with 5 different classes.

## Software Installation

To install all libraries and packages necessary to run this project, run: 

```
pip install requirements.txt
```

## Dataset Installation

Please download the dataset necessary for this project [here](https://drive.google.com/open?id=1LOivxHk_6kZ8LpOWvdqo4DZzAIfF1cVP). Unzip it and add the folder inside the *Datasets* folder.

The whole project (featuring all the files in this GitHub repository plus the dataset folder) is also available [here](https://drive.google.com/open?id=1ZIhLUtEeXBAVrjwTi2pyk0dBrG7Ra-rC).

## Usage

To run the project's main program, move into the project's directory and once in it, run:

```
python main.py
```

This script will output the train, test and additional test sets accuracies of the final model selected for each task.


To run all developed models for each task, move into the project's directory and once in it, run:

```
python main<task>.py
```

where *task* should be substituted by the task name i.e. A1, A2, B1 or B2.

This script will output the train, test and additional test sets accuracies for each of the models tested for the selected task.

## Project Structure

The project has the following structure:

**A1** (folder) : Contains models for A1 Task, and Jupyter notebooks showing how hyperparameters were fine-tuned
* **A1FL.py** : Trains and tests a SVM model with facial landmarks as features
* **A1LBP.py** : Trains and tests a SVM model with Local Binary Patterns (LBP) as features
* **A1VGG.py** : Trains, validates and tests a pre-trained VGG model with raw pixels as input
* **hyperparametersA1model1.ipynb** : Jupyter notebook that performs Grid-Search CV to find best parameters for SVM model with facial landmarks
* **hyperparametersA1model2.ipynb** : Jupyter notebook that performs Grid-Search CV to find best parameters for SVM model with LBP

**A2** (folder) : Contains models for A2 Task, and Jupyter notebook showing how hyperparameters were fine-tuned
* **A2FL.py** : Trains and tests a SVM model with facial landmarks as features
* **A2VGG.py** : Trains, validates and tests a pre-trained VGG model with raw pixels as input
* **hyperparametersA2model1.ipynb** : Jupyter notebook that performs Grid-Search CV to find best parameters for SVM model with facial landmarks

**B1** (folder) : Contains models for B1 Task, and Jupyter notebook showing how hyperparameters were fine-tuned
* **B1FL.py** : Trains and tests a SVM model with facial landmarks as features
* **B1VGG.py** : Trains, validates and tests a pre-trained VGG model with raw pixels as input
* **B1ResNet.py** : Trains, validates and tests a pre-trained ResNet model with raw pixels as input
* **hyperparametersB1model1.ipynb** : Jupyter notebook that performs Grid-Search CV to find best parameters for SVM model with facial landmarks

**B2** (folder) : Contains models for B2 Task
* **B2VGG.py** : Trains, validates and tests a pre-trained VGG model with raw pixels as input
* **B2ResNet.py** : Trains, validates and tests a pre-trained ResNet model with raw pixels as input

**Datasets** (folder) : Contains data pre-processing files and original dataset
* **dataset** (folder) : Folder downloadable from [here](https://drive.google.com/open?id=1LOivxHk_6kZ8LpOWvdqo4DZzAIfF1cVP), where all the data used in the project is.
* **dataA1.py** : Pre-processes and augments data for each model implemented for Task A1
* **dataA2.py** : Pre-processes and augments data for each model implemented for Task A2
* **dataB1.py** : Pre-processes and augments data for each model implemented for Task B1
* **dataB2.py** : Pre-processes and augments data for each model implemented for Task B2

**HelperFunctions** (folder) : Contains CNN methods, curve plotting and landmark computation scripts
* **cnnMethods.py** : Script containing functions to train and test a CNN
* **curvesCNN.py** : Functions that plot learning curves for CNN models
* **curvesSVM.py** : Function that plots learning curve for SVM models
* **landmarksA1.py** : Script to compute facial landmarks for task A1
* **landmarksA2.py** : Script to compute facial landmarks for task A2
* **landmarksB1.py** : Script to compute facial landmarks for task B1

**main.py** : Trains and tests model chosen for each task, printing the train and test accuracy for each

**mainA1.py** : Trains and tests each model implemented for Task A1, printing the train and test accuracy for each

**mainA2.py** : Trains and tests each model implemented for Task A2, printing the train and test accuracy for each

**mainB1.py** : Trains and tests each model implemented for Task B1, printing the train and test accuracy for each

**mainB2.py** : Trains and tests each model implemented for Task B2, printing the train and test accuracy for each