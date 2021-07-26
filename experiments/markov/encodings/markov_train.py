import glob
import os
import pickle
import sys
from typing import Dict, List

import numpy as np
from PIL import Image


def get_markov_counts(
    encodings: List[Dict[int, List[int]]]
) -> Dict[str, Dict[str, float]]:
    """
    Computes the markov counts.
    Function to get all counts for each combination of key -> token.
        We use a key of (west, southwest, south).
        I.E. we consider the tokens west, sw & south to produce the next token.
        E.G. {(0, 0, 0): {0: 20, 255: 15}}
    """
    markov_counts = {}
    for encoding in encodings:
        max_y = len(encoding) - 1
        for y in range(max_y, -1, -1):
            for x in range(0, len(encoding[y])):
                west = encoding[y][x - 1] if x > 0 else 112
                south = encoding[y + 1][x - 1] if y < max_y else 112
                southwest = encoding[y + 1][x] if x > 0 and y < max_y else 112
                key = (west, southwest, south)

                if key not in markov_counts:
                    markov_counts[key] = {}
                markov_counts[key][encoding[y][x]] = (
                    markov_counts[key].get(encoding[y][x], 0.0) + 1.0
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


if __name__ == "__main__":
    data_path = sys.argv[1]  # Path to pokemon directory
    outfile = sys.argv[2]  # Path to save probabilities to
    default_class = int(sys.argv[3]) # The background token class. 112 for 5.10

    encodings = np.load(data_path)
    encodings = encodings.reshape(encodings.shape[0], 64, 64)
    markov_counts = get_markov_counts(encodings)
    markov_probabilities = get_markov_probabilities(markov_counts)
    pickle.dump(markov_probabilities, open(outfile, "wb"))
