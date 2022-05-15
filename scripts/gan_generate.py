"""
Generate images using a GAN.
"""
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append("./")
from models import gan

seed = 42
np.random.seed(seed)
_ = torch.manual_seed(seed)

# GAN Config
experiment_name = "sprites_gan_v1"
epoch_to_load = 24
latent_dim = 100
generator_num_filters = 64
num_output_channels = 3

# Generation Config
num_generations = 10000
batch_size = 32
num_sample_batches = (num_generations // batch_size) + 1

# Data Config
model_prefix = f"outputs\\{experiment_name}"
model_path = os.path.join(model_prefix, "models", f"epoch_{epoch_to_load}_model.pt")
generated_dir = os.path.join("outputs", experiment_name, "generated_samples")

if not os.path.exists(generated_dir):
    os.makedirs(generated_dir)

# Setup Device
gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")
print(gpu, device)

# Create Model
model = gan.DCGANGenerator(
    latent_dim=latent_dim,
    num_filters=generator_num_filters,
    num_output_channels=num_output_channels,
)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["generator_model_state_dict"])
model.eval()
model.to(device)

with torch.no_grad():
    for iteration in tqdm(range(num_sample_batches)):
        # Generate batch of latent vectors
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        # Generate fake image batch with G
        generated = model(noise).clamp(min=0, max=1)
        generated = generated.permute(0, 2, 3, 1).detach().cpu().numpy()
        for j, image in enumerate(generated):
            filename = f"{(batch_size * iteration) + j}.png"
            plt.imsave(os.path.join(generated_dir, filename), image)
