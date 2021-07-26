import sys
import os
import random
import pickle
from datetime import datetime
import numpy as np
from PIL import Image

probabilities_file = sys.argv[1]
out_dir = probabilities_file.split("\\")[:-1]
outfile = os.path.join(*out_dir, str(int(datetime.utcnow().timestamp())) + ".png")
print(outfile)

markov_probabilities = pickle.load(open(probabilities_file, "rb"))
max_height = 64
max_length = 64
sprite = {}

for y in range(max_height, 0, -1):
    sprite[y] = []
    for x in range(0, max_length):
        west = sprite[y][x - 1] if x > 0 else (255, 255, 255)
        south = sprite[y + 1][x - 1] if y < max_height else (255, 255, 255)
        southwest = sprite[y + 1][x] if x > 0 and y < max_height else (255, 255, 255)
        key = (west, southwest, south)

        if key in markov_probabilities.keys():
            # Pick Probabilistically
            if random.choice([True]):
                random_sample = random.uniform(0, 1)
                possibilities = [x for x in markov_probabilities[key].keys()]
                probs = [float(x) for x in markov_probabilities[key].values()]
                current_value = 0.0
                for i, proba in enumerate(probs):
                    current_value += proba
                    if random_sample <= current_value:
                        sprite[y].append(possibilities[i])
                        break
            # Pick Randomly
            else:
                sprite[y].append(random.choice(list(markov_probabilities[key].keys())))
        else:
            sprite[y].append((255, 255, 255))

sprite = np.array([sprite[x] for x in sprite], dtype="uint8")
im = Image.fromarray(sprite)
im.save(outfile)