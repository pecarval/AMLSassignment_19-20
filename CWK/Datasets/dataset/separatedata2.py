import shutil
import pandas as pd
import random
import os

labels = pd.read_csv('A1/labels.csv')

source = 'A1/img/'
source_files = os.listdir(source)

test = 'A1/test/'
train = 'A1/train/'
val = 'A1/val/'

files = os.listdir(source)

source_files.sort()  # make sure that the filenames have a fixed order before shuffling
random.seed(230)
random.shuffle(source_files) # shuffles the ordering of filenames (deterministic given the chosen seed)


split1_1 = int(0.64 * len(source_files))
split1_2 = int(0.8 * len(source_files))
train_filenames = source_files[:split1_1]
val_filenames = source_files[split1_1:split1_2]
test_filenames = source_files[split1_2:]

train_csv = 'A1_trainlabels.csv'
test_csv = 'A1_testlabels.csv'
val_csv = 'A1_vallabels.csv'

for f in train_filenames:
   shutil.move(source + f, train)
   with open(train_csv,'a') as fd:
      fd.write(labels.iloc[[str(f[:-4])]].to_csv(header=False,index=False))

for f in test_filenames:
   shutil.move(source + f, test)
   with open(test_csv,'a') as fd:
      fd.write(labels.iloc[[str(f[:-4])]].to_csv(header=False,index=False))

for f in val_filenames:
   shutil.move(source + f, val)
   with open(val_csv,'a') as fd:
      fd.write(labels.iloc[[str(f[:-4])]].to_csv(header=False,index=False))
