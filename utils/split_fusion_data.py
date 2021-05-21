import os
import random

def remove_3_digit(num):
    if '-' in num:
        num, *addon = num.split('-')
        stuff = [str(int(num))] + [x for x in addon]
        return "-".join(stuff)
    return str(int(num))

fusions_dir = "data\\japeal"
original_data_dir = "data\\training"
output_dir = "data\\fusions"

# Remove New Gen Fusions
for folder in os.listdir(fusions_dir):
    if int(folder) >= 650:
        os.remove(os.path.join(fusions_dir, folder))

# Get original train-test-val IDs.
train = []
for file in os.listdir(os.path.join(original_data_dir, "train")):
    file = file.split('_')[0]
    if file not in train:
        train.append(file)
val = []
for file in os.listdir(os.path.join(original_data_dir, "val")):
    file = file.split('_')[0]
    if file not in val:
        val.append(file)
test = []
for file in os.listdir(os.path.join(original_data_dir, "test")):
    file = file.split('_')[0]
    if file not in test:
        test.append(file)

# 001 -> 1
train = [remove_3_digit(x) for x in train]
test = [remove_3_digit(x) for x in test]
val = [remove_3_digit(x) for x in val]

cache = []
for folder in os.listdir(fusions_dir):
    for file in os.listdir(os.path.join(fusions_dir, folder)):
        first, second, _ = file.split('.')
        second = second.split('-')[0]
        file_path = os.path.join(fusions_dir, folder, file)
        if first in train and second in train:
            target_dir = os.path.join(output_dir, "train", first)
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
            new_file = os.path.join(target_dir, file)
        elif first in test or second in test:
            target_dir = os.path.join(output_dir, "test", first)
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
            new_file = os.path.join(target_dir, file)
        elif first in val or second in val:
            target_dir = os.path.join(output_dir, "val", first)
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
            new_file = os.path.join(target_dir, file)
        if new_file not in cache:
            cache.append(new_file)
            os.rename(file_path, new_file)
        else:
            print(file_path, new_file)