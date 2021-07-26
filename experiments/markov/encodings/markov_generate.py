import sys
import os
import random
import pickle
from datetime import datetime
import numpy as np
from PIL import Image
import torch

sys.path.append("./")
import models

probabilities_file = sys.argv[1]
default_class = int(sys.argv[2])  # The background token class. 112 for 5.10
out_dir = probabilities_file.split("\\")[:-1]
outfile = os.path.join(*out_dir, str(int(datetime.utcnow().timestamp())) + ".png")
print(outfile)

image_size = 64
vq_vae_num_layers = 0
vq_vae_small_conv = True
vq_vae_num_embeddings = 256
vq_vae_embedding_dim = 32
vq_vae_commitment_cost = 0.25
vq_vae_use_max_filters = True
vq_vae_max_filters = 512
vq_vae_model_path = "outputs/vq_vae_v5.10/model.pt"
vq_vae_encoded_image_size = image_size // np.power(2, vq_vae_num_layers)

markov_probabilities = pickle.load(open(probabilities_file, "rb"))
max_height = 64
max_length = 64
sprite = {}

for y in range(max_height, 0, -1):
    sprite[y] = []
    for x in range(0, max_length):
        west = sprite[y][x - 1] if x > 0 else default_class
        south = sprite[y + 1][x - 1] if y < max_height else default_class
        southwest = sprite[y + 1][x] if x > 0 and y < max_height else default_class
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
            sprite[y].append(default_class)

sprite = np.array([sprite[x] for x in sprite], dtype="uint8")

# Setup Device
gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")
print(gpu, device)

# Load Model
vq_vae = models.VQVAE(
    num_layers=vq_vae_num_layers,
    input_image_dimensions=image_size,
    small_conv=vq_vae_small_conv,
    embedding_dim=vq_vae_embedding_dim,
    num_embeddings=vq_vae_num_embeddings,
    commitment_cost=vq_vae_commitment_cost,
    use_max_filters=vq_vae_use_max_filters,
    max_filters=vq_vae_max_filters,
)
vq_vae.load_state_dict(torch.load(vq_vae_model_path, map_location=device))
vq_vae.eval()
vq_vae.to(device)

sprite = torch.Tensor(sprite).view(-1, 1).long()
target_shape = (
    1,
    vq_vae_encoded_image_size,
    vq_vae_encoded_image_size,
    vq_vae_embedding_dim,
)
sprite = vq_vae.quantize_and_decode(
    sprite,
    target_shape,
    device,
).squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
im = Image.fromarray((sprite * 255).astype(np.uint8))
im.save(outfile)