# ELEC0134 Applied Machine Learning Systems Assignment

This repository implements the ELEC0134 Applied Machine Learning Systems assignment. This assignment consists of fours tasks:
* Task A1: Gender Detection
* Task A2: Emotion Detection
* Task B1: Face Shape Recognition
* Task B2: Eye Color Recognition

Tasks A1 and A2 are binary classification problems. Task A1 classifies for "male" or "female". Task A2 classifies for "smile" or "no smile".
Tasks B1 and B2 are multi-class classification problems. Both B classes classify for 5 different classes.

## Software Installation

To install all libraries and packages necessary to run this project, run: 

```
pip install requirements.txt
```

## Dataset Installation

Please download the dataset necessary for this project here. Unzip it and add the folder inside the *Datasets* folder.

## Usage

To run the project's main program, move into the directory of the project and once in it, run:

```
python main.py
```

This script will output the training and testing accuracies for each of the tasks, as required in the assigment description.


To run all the possible models for each task, move into the directory of the project and once in it, run:

```
python main<task>.py
```

where *<task>* should be substituted by the task name i.e. A1, A2, B1 or B2.

This script will output the training and testing accuracies for each of the models tested by the tasks you selected.

## Project Structure

The project has the following structure,

**A1** (folder) : Contains models for A1 Task, and Jupyter notebooks showing how hyperparameters were fine-tuned
* **A1FL.py** : Trains and tests a SVM model with facial landmarks as features
* **A1LBP.py** : Trains and tests a SVM model with Local Binary Patterns (LBP) as features
* **A1VGG.py** : Trains, validates and tests a pre-trained VGG model with raw pixels as input
* **hyperparametersA1model1.ipynb** : Jupyter notebook that performs Grid-Search CV to find best parameters for SVM model with facial landmarks
* **hyperparametersA1model1.ipynb** : Jupyter notebook that performs Grid-Search CV to find best parameters for SVM model with LBP

**A2** (folder) : Contains models for A2 Task, and Jupyter notebook showing how hyperparameters were fine-tuned
* **A2FL.py** : Trains and tests a SVM model with facial landmarks as features
* **A2VGG.py** : Trains, validates and tests a pre-trained VGG model with raw pixels as input
* **hyperparametersA2model1.ipynb** : Jupyter notebook that performs Grid-Search CV to find best parameters for SVM model with facial landmarks

**B1** (folder) : Contains models for B1 Task, and Jupyter notebook showing how hyperparameters were fine-tuned
* **B1FL.py** : Trains and tests a SVM model with facial landmarks as features
* **B1VGG.py** : Trains, validates and tests a pre-trained VGG model with raw pixels as input
* **hypeparametersA2model1.ipynb** : Jupyter notebook that performs Grid-Search CV to find best parameters for SVM model with facial landmarks

**B2** (folder) : Contains models for B2 Task
* **B2VGG.py** : Trains, validates and tests a pre-trained VGG model with raw pixels as input

**Datasets** (folder) : Contains data pre-processing files, curve plotting scripts and original dataset
* **dataset** (folder) : Folder downloaded from here, where all the data used in the project is.
* **LandmarksFT** (folder) : Folder that contains landmark computation scripts that are used in the hyper-parameter tuning files.
* **LandmarksMain** (folder) : Folder that contains landmark computation scripts that are used in data pre-processing files.
* **dataA1.py** : Pre-processes and augments data for each model implemented for Task A1
* **dataA2.py** : Pre-processes and augments data for each model implemented for Task A2
* **dataB1.py** : Pre-processes and augments data for each model implemented for Task B1
* **dataB2.py** : Pre-processes and augments data for each model implemented for Task B2
* **curvesCNN.py** : Plots accuracy vs. epoch & loss vs. epoch curves for CNN models
* **curvesSVM.py** : Plots accuracy vs. training examples curve for SVM models

**main.py** : Trains and tests model chosen for each task, printing the train and test accuracy for each

**mainA1.py** : Trains and tests each model implemented for Task A1, printing the train and test accuracy for each

**mainA2.py** : Trains and tests each model implemented for Task A2, printing the train and test accuracy for each

**mainB1.py** : Trains and tests each model implemented for Task B1, printing the train and test accuracy for each

**mainB2.py** : Trains and tests each model implemented for Task B2, printing the train and test accuracy for each

## License
[MIT](https://choosealicense.com/licenses/mit/)