"""
Generate images using a VAE.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append("./")
import utils.data as data
from models import vae

seed = 42
np.random.seed(seed)
_ = torch.manual_seed(seed)

# VAE Config
experiment_name = "base_convolutional_vae_v1"
num_layers = 4
max_filters = 512
image_size = 64
latent_dim = 256
small_conv = False  # To use the 1x1 convolution layer

# Generation Config
num_generations = 10000
batch_size = 32
num_sample_batches = (num_generations // batch_size) + 1

# Data Config
batch_size = 32
num_dataloader_workers = 0
model_prefix = f"outputs\\{experiment_name}"
model_path = os.path.join(model_prefix, "model.pt")
generated_dir = os.path.join("data", "model_outputs_for_fid", experiment_name)

if not os.path.exists(generated_dir):
    os.makedirs(generated_dir)

# Setup Device
gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")
print(gpu, device)

# Preprocess & Create Data Loaders
transform = data.image2tensor_resize(image_size)

# Create Model
model = vae.ConvolutionalVAE(
    max_filters=max_filters,
    num_layers=num_layers,
    input_image_dimensions=image_size,
    latent_dim=latent_dim,
    small_conv=small_conv,
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

with torch.no_grad():
    for iteration in tqdm(range(num_sample_batches)):
        embeddings = embeddings = torch.rand(
            (batch_size, latent_dim), device=device
        )
        generated = model.decoder(embeddings)

        generated = generated.permute(0, 2, 3, 1).detach().cpu().numpy()
        for j, image in enumerate(generated):
            filename = f"{(batch_size * iteration) + j}.png"
            plt.imsave(os.path.join(generated_dir, filename), image)
