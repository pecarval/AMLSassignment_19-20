import shutil
import pandas as pd
import random
import os


source = 'A2/img/'
dest1 = 'A2/smile/'
dest2 = 'A2/nosmile/'
labels = pd.read_csv('A1/labels.csv')

files = os.listdir(source)

for f in files:
    num = int(f[:-4])
    if (int(labels.loc[num,"smiling"]) == 1):
       shutil.move(source + f, dest1)
    else:
       shutil.move(source + f, dest2)


label1 = 'smile/'
label2 = 'nosmile/'

label1dir = dest1
label2dir = dest2

label1files = os.listdir(label1dir)
label2files = os.listdir(label2dir)

trainLabel1 = 'A2/train/' + label1
trainLabel2 = 'A2/train/' + label2

testLabel1 = 'A2/test/' + label1
testLabel2 = 'A2/test/' + label2

valLabel1 = 'A2/val/' + label1
valLabel2 = 'A2/val/' + label2

label1files.sort()  # make sure that the filenames have a fixed order before shuffling
random.seed(230)
random.shuffle(label1files) # shuffles the ordering of filenames (deterministic given the chosen seed)

split1_1 = int(0.7 * len(label1files))
split1_2 = int(0.9 * len(label1files))
train1_filenames = label1files[:split1_1]
val1_filenames = label1files[split1_1:split1_2]
test1_filenames = label1files[split1_2:]

label2files.sort()  # make sure that the filenames have a fixed order before shuffling
random.seed(230)
random.shuffle(label2files) # shuffles the ordering of filenames (deterministic given the chosen seed)

split2_1 = int(0.7 * len(label2files))
split2_2 = int(0.9 * len(label2files))
train2_filenames = label2files[:split2_1]
val2_filenames = label2files[split2_1:split2_2]
test2_filenames = label2files[split2_2:]


for f in train1_filenames:
   shutil.move(label1dir + f, trainLabel1)

for f in train2_filenames:
   shutil.move(label2dir + f, trainLabel2)

for f in test1_filenames:
   shutil.move(label1dir + f, testLabel1)

for f in test2_filenames:
   shutil.move(label2dir + f, testLabel2)

for f in val1_filenames:
   shutil.move(label1dir + f, valLabel1)

for f in val2_filenames:
   shutil.move(label2dir + f, valLabel2)
