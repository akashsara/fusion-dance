# Make a small subset of the training data
# This is removed entirely and kept aside for testing purposes.
# We use the paritioned training data for this as it is the largest.
# Note that this must be done before training any models.
# This separates out 5 sets of Pokemon each for:
# Category 1: A Pokemon whose base is not known.
# Category 2: A Pokemon whose fusee is not known.
# Category 3: A Pokemon whose base & fusee are not known.
# The idea is to see if the model can learn to fuse Pokemon which are not 
# involved in the training process at all (category 3). 
# The first two categories are smaller tests for this purpose.
import os
import random

normal_train_dir = "data\\training\\train"
fusion_train_dir = "data\\fusions\\train"
fusion_test_dir = "data\\fusions\\fusion_test"

# Get valid training IDs
train = []
for file in os.listdir(normal_train_dir):
    file = file.split("_")[0]
    if file not in train:
        train.append(file)
train = [x.split("-")[0] for x in train]
train = list(set(train))

train_fusions = os.listdir(fusion_train_dir)
both, base, fusee = [], [], []
while len(both) < 5 or len(base) < 5 or len(fusee) < 5:
    temp = random.choice(train)
    if temp in train_fusions:
        if temp in both or temp in base or temp in fusee:
            continue
        if len(both) < 5:
            both.append(temp)
        elif len(base) < 5:
            base.append(temp)
        elif len(fusee) < 5:
            fusee.append(temp)

for folder in os.listdir(fusion_train_dir):
    # Base
    if folder in base:
        src = os.path.join(fusion_train_dir, folder)
        dest = os.path.join(fusion_test_dir, "base", folder)
        os.rename(src, dest)
    # Both - Base
    elif folder in both:
        src = os.path.join(fusion_train_dir, folder)
        dest = os.path.join(fusion_test_dir, "both", folder)
        os.rename(src, dest)
    else:
        for file in os.listdir(os.path.join(fusion_train_dir, folder)):
            first, second, _ = file.split(".")
            second = second.split("-")[0]
            # Fusee
            if first in fusee or second in fusee:
                if not os.path.exists(os.path.join(fusion_test_dir, "fusee", folder)):
                    os.mkdir(os.path.join(fusion_test_dir, "fusee", folder))
                src = os.path.join(fusion_train_dir, folder, file)
                dest = os.path.join(fusion_test_dir, "fusee", folder, file)
                os.rename(src, dest)
            # Both - Fusee
            elif first in fusee or second in fusee:
                if not os.path.exists(os.path.join(fusion_test_dir, "both", folder)):
                    os.mkdir(os.path.join(fusion_test_dir, "both", folder))
                src = os.path.join(fusion_train_dir, folder, file)
                dest = os.path.join(fusion_test_dir, "both", folder, file)
                os.rename(src, dest)