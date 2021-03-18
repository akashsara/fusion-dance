"""
This script is reponsible for preprocessing the original data folder.
Since the images are in RGBA, we first need to convert them into RGB images.
Now we need to decide what color the background should be. 
White or black are standard, but they may impact how the model sees certain 
Pokemon that have very dark or very light shades.
We follow Gonzalez et. al's approach and use both. 
They also used 2 additional noisy backgrounds, but we leave that for the future.
Additionally, we also perform horizontal flipping for every Pokemon.
"""

import os
from PIL import Image
import sys

input_dir = sys.argv[1]
output_dir = sys.argv[2]

def change_background_color(image, color):
    colors = {
        "black": (0, 0, 0),
        "white": (255, 255, 255)
    }
    background = Image.new("RGBA", image.size, colors[color])
    return Image.alpha_composite(background, image).convert("RGB")


if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for input_file in os.listdir(input_dir):
    print(input_file)
    # Load file
    sprite = Image.open(os.path.join(input_dir, input_file)).convert("RGBA")
    
    for color in ["white", "black"]:
        # Change background color
        new_sprite = change_background_color(sprite, color)
        # Horizontally flipped copy of the image
        flipped = new_sprite.transpose(Image.FLIP_LEFT_RIGHT)
        # Create output file names
        output_file = input_file.split(".")[0] + f"_{color}BG.png"
        flipped_output_file = output_file.split(".")[0] + "_flipped.png"
        # Save image
        new_sprite.save(os.path.join(output_dir, output_file))
        flipped.save(os.path.join(output_dir, flipped_output_file))