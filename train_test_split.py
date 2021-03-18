"""
Simple script to do the train-test split we need.
We use each Pokemon's pokedex ID as it's unique ID.
Additionally, we split based on these IDs so as to not contaminate our testing 
or validation sets. 
Thus there may be an uneven split since some Pokemon have female sprites.
It automatically creates the train/val/test folders and moves the images there.
"""

import os
import sys
import random

data_dir = sys.argv[1]
num_test = int(sys.argv[2])
num_valid = int(sys.argv[3])

files = os.listdir(data_dir)
for folder in ["train", "test", "val"]:
    if not os.path.exists(os.path.join(data_dir, folder)):
        os.mkdir(os.path.join(data_dir, folder))

unique = []
for file in files:
    file = file.split("_")[0]
    if file not in unique and file not in ["train", "test", "val"]:
        unique.append(file)

random.shuffle(unique)
test = set(unique[-num_test:])
unique = unique[:-num_test]
val = set(unique[-num_valid:])
train = set(unique[:-num_valid])

print(f"Train: {len(train)}\nVal: {len(val)}\nTest: {len(test)}")

for file in files:
    index = file.split("_")[0]
    filepath = os.path.join(data_dir, file)
    if index in train:
        os.rename(filepath, os.path.join(data_dir, "train", file))
    elif index in test:
        os.rename(filepath, os.path.join(data_dir, "test", file))
    elif index in val:
        os.rename(filepath, os.path.join(data_dir, "val", file))