import glob
import os
import pickle
import sys
from typing import Dict, List

import numpy as np
from PIL import Image


def get_markov_counts(
    sprites: List[Dict[int, List[int]]]
) -> Dict[str, Dict[str, float]]:
    """
    Computes the markov counts.
    Function to get all counts for each combination of key -> token.
        We use a key of (west, southwest, south).
        I.E. we consider the tokens west, sw & south to produce the next token.
        E.G. {(0, 0, 0): {0: 20, 255: 15}}
    """
    markov_counts = {}
    for sprite in sprites:
        max_y = len(sprite) - 1
        for y in range(max_y, -1, -1):
            for x in range(0, len(sprite[y])):
                west = sprite[y][x - 1] if x > 0 else (255, 255, 255)
                south = sprite[y + 1][x - 1] if y < max_y else (255, 255, 255)
                southwest = sprite[y + 1][x] if x > 0 and y < max_y else (255, 255, 255)
                key = (west, southwest, south)

                if key not in markov_counts:
                    markov_counts[key] = {}
                markov_counts[key][sprite[y][x]] = (
                    markov_counts[key].get(sprite[y][x], 0.0) + 1.0
                )
    return markov_counts


def get_markov_probabilities(
    markov_counts: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Normalizes the counts in the markov_counts dictionary.
    """
    for key in markov_counts:
        total = 0
        for token in markov_counts[key]:
            total += markov_counts[key][token]
        for token in markov_counts[key]:
            markov_counts[key][token] /= total
    return markov_counts


def load_images(image_dir: str) -> List[Dict[int, List[int]]]:
    sprites = []
    for filename in glob.glob(os.path.join(image_dir, "*.png")):
        sprite = {}
        img = Image.open(filename).convert("RGBA")
        # Add white background
        background = Image.new('RGBA', img.size, (255,255,255))
        img = Image.alpha_composite(background, img)
        # Resize
        img = img.resize((64, 64), resample=Image.BICUBIC)
        # Convert to RGB
        img = img.convert("RGB")
        # im = im.convert("L")  # Grayscale
        img = np.array(img)  # Convert into an array where each pixel is [0,255]
        y = 0
        for row in img:
            sprite[y] = [tuple(x) for x in row]
            y += 1
        sprites.append(sprite)
    return sprites


if __name__ == "__main__":
    data_path = sys.argv[1]  # Path to pokemon directory
    outfile = sys.argv[2]  # Path to save probabilities to

    images = load_images(data_path)
    markov_counts = get_markov_counts(images)
    markov_probabilities = get_markov_probabilities(markov_counts)
    pickle.dump(markov_probabilities, open(outfile, "wb"))
