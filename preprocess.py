"""
Train & Test Split:
We use each Pokemon's pokedex ID as it's unique ID.
Additionally, we split based on these IDs so as to not contaminate our testing 
or validation sets. 
Thus there may be an uneven split since some Pokemon have female sprites.
It automatically creates the train/val/test folders and moves the images there.

Preprocessing:
This script is reponsible for preprocessing the original data folder.
Since the images are in RGBA, we first need to convert them into RGB images.
Now we need to decide what color the background should be. 
White or black are standard, but they may impact how the model sees certain 
Pokemon that have very dark or very light shades.
We follow Gonzalez et. al's approach and use both. 
In the case of training images, they also used 2 additional noisy backgrounds.
We repeat the same.
In terms of image augmentation, we perform horizontal flips as well as rotations in the 15 and 30 degree angles in two directions. We do not rotate the horizontally flipped image though.
"""

import os
from PIL import Image
import sys
import random
import numpy as np

def change_background_color(image, color):
    if color == "none":
        return image.convert("RGB")
    if color == "black":
        background = Image.new("RGBA", image.size, (0, 0, 0))
    elif color == "white":
        background = Image.new("RGBA", image.size, (255, 255, 255))
    else:
        image_size = image.size + (3,)
        background = Image.fromarray(np.random.normal(0, 1, image_size), mode="RGBA")
    return Image.alpha_composite(background, image).convert("RGB")

input_dir = sys.argv[1]
output_dir = sys.argv[2]
num_test = int(sys.argv[3])
num_valid = int(sys.argv[4])
rotations = [15, 30]
use_noise = True
do_a_flip = True
colors = ["white", "black"]

# Create Folders
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for folder in ["train", "test", "val"]:
    if not os.path.exists(os.path.join(output_dir, folder)):
        os.mkdir(os.path.join(output_dir, folder))

# Do a Train-Val-Test Split
unique = []
for file in os.listdir(input_dir):
    file = file.split('.')[0].split("_")[0].split("-")[0]
    if file not in unique and file not in ["train", "test", "val"]:
        unique.append(file)

random.shuffle(unique)
test = set(unique[-num_test:])
unique = unique[:-num_test]
val = set(unique[-num_valid:])
train = set(unique[:-num_valid])
print(f"Train: {len(train)}\nVal: {len(val)}\nTest: {len(test)}")

all_train = []
all_val = []
all_test = []
for input_file in os.listdir(input_dir):
    print(input_file)
    # Load file
    sprite = Image.open(os.path.join(input_dir, input_file)).convert("RGBA")

    # Find directory to save to
    input_file_name = input_file.split(".")[0]
    image_id = input_file_name.split("_")[0].split("-")[0]
    
    if image_id in test:
        save_dir = os.path.join(output_dir, "test")
        all_test.append(image_id)
    elif image_id in val:
        save_dir = os.path.join(output_dir, "val")
        all_val.append(image_id)
    else:
        save_dir = os.path.join(output_dir, "train")
        all_train.append(image_id)
        if use_noise:
            colors.extend(["noise1", "noise2"])

    # Augment data
    for color in colors:
        rotation = 0
        # Change background color & save base image
        new_sprite = change_background_color(sprite, color)
        output_file = f"{input_file_name}_{color}BG_0rotation.png"
        new_sprite.save(os.path.join(save_dir, output_file))

        # Rotation
        prev_angle = 0
        for angle in rotations:
            # Counter Clockwise Rotation
            new_angle = np.random.randint(prev_angle, angle)
            rotated = sprite.rotate(angle)
            rotated = change_background_color(rotated, color)
            output_file = f"{input_file_name}_{color}BG_{new_angle}rotation.png"
            rotated.save(os.path.join(save_dir, output_file))
            # Clockwise Rotation
            new_angle = 360 - np.random.randint(prev_angle, angle)
            rotated = sprite.rotate(angle)
            rotated = change_background_color(rotated, color)
            output_file = f"{input_file_name}_{color}BG_{new_angle}rotation.png"
            rotated.save(os.path.join(save_dir, output_file))
            # Housekeeping
            prev_angle = angle

        # Horizontal Flip
        if do_a_flip:
            if "noise" in color:
                new_sprite = change_background_color(sprite, color)
            new_sprite = new_sprite.transpose(Image.FLIP_LEFT_RIGHT)
            output_file = f"{input_file_name}_{color}BG_flipped.png"
            new_sprite.save(os.path.join(save_dir, output_file))
print(f"Train: {len(all_train)}\nVal: {len(all_val)}\nTest: {len(all_test)}")
