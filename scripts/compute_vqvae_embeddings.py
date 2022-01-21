import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append("./")
import utils.data as data
from models import vqvae

seed = 42
np.random.seed(seed)
_ = torch.manual_seed(seed)

################################################################################
#################################### Config ####################################
################################################################################

# VQ-VAE Config
vq_vae_image_size = 64
vq_vae_experiment_name = f"sprites_vq_vae_v2"
vq_vae_num_layers = 1
vq_vae_max_filters = 512
vq_vae_use_max_filters = True
vq_vae_num_embeddings = 256
vq_vae_embedding_dim = 32
vq_vae_commitment_cost = 0.25
vq_vae_small_conv = True  # To use the 1x1 convolution layer
vq_vae_embedding_size = vq_vae_image_size // (2 ** vq_vae_num_layers)

# Pokedex
pokedex_url = "https://raw.githubusercontent.com/lgreski/pokemonData/master/Pokemon.csv"
pokedex = pd.read_csv(pokedex_url).drop_duplicates(["ID", "Name"])
pokedex.index = pokedex.ID
pokemon_types = pokedex["Type1"].to_dict()
pokedex = pokedex["Name"].to_dict()

# Data Config
batch_size = 32
num_dataloader_workers = 0
background = (255, 255, 255)
dataset = "sprites" # "pokemon", "tinyhero", "sprites"
data_folder_lookup = {
    "pokemon": "data\\pokemon\\original_data",
    "tinyhero": "data\\TinyHero\\processed",
    "sprites": "data\Sprites\processed"
}
data_folder = data_folder_lookup[dataset]

model_prefix = f"outputs\\{vq_vae_experiment_name}"
model_path = os.path.join(model_prefix, "model.pt")

# Setup Device
gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")
print(gpu, device)

# Preprocess & Create Data Loaders
transform = data.image2tensor_resize(vq_vae_image_size)

# Load Data
data_in = data.CustomDatasetNoMemoryAddBackground(
    data_folder, dataset, transform, background
)

input_dataloader = torch.utils.data.DataLoader(
    data_in,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_dataloader_workers,
    pin_memory=gpu,
)

# Create & Load VQVAE Model
model = vqvae.VQVAE(
    num_layers=vq_vae_num_layers,
    input_image_dimensions=vq_vae_image_size,
    small_conv=vq_vae_small_conv,
    embedding_dim=vq_vae_embedding_dim,
    num_embeddings=vq_vae_num_embeddings,
    commitment_cost=vq_vae_commitment_cost,
    use_max_filters=vq_vae_use_max_filters,
    max_filters=vq_vae_max_filters,
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

all_embeddings = []
all_encodings = []
all_filenames = []
all_color_ids = []
with torch.no_grad():
    for iteration, batch in enumerate(tqdm(input_dataloader)):
        # Move batch to device
        filenames, batch = batch  # Returns key, value for each Pokemon
        batch = batch.to(device)

        # Get Encodings from vq_vae
        current_batch_size = len(batch)
        _, _, _, encodings = model(batch)
        target_shape = (
            current_batch_size,
            vq_vae_embedding_size,
            vq_vae_embedding_size,
            vq_vae_embedding_dim,
        )
        embeddings = model.vq_vae.quantize_encoding_indices(
            encodings, target_shape, device
        )
        # Add embeddings to list
        all_embeddings.append(embeddings)
        all_encodings.append(encodings.reshape(current_batch_size, -1))
        # Add filenames (with Pokemon name instead of ID) to list
        for filename in filenames:
            if "pokemon" in dataset:
                filename = filename.split("_")[0]
                pokemon_id = filename.split("-")[0].split(".")[0]
                filename = filename.replace(pokemon_id, pokedex[int(pokemon_id)])
                all_color_ids.append(pokemon_types[int(pokemon_id)])
            elif dataset == "tinyhero":
                color = filename.split('.')[0].split('_')[1]
                all_color_ids.append(color)
            elif dataset == "sprites":
                id, pose, anim = filename.split('.')[0].split('_')
                all_color_ids.append([id, pose, anim])
            all_filenames.append(filename)
# Save to file
all_embeddings = torch.cat(all_embeddings).detach().cpu()
all_encodings = all_encodings.float() / vq_vae_num_embeddings

torch.save(
    {"embeddings": all_embeddings, "filenames": all_filenames, "color": all_color_ids},
    os.path.join(model_prefix, f"embeddings.pt"),
)
torch.save(
    {"embeddings": all_encodings, "filenames": all_filenames, "color": all_color_ids},
    os.path.join(model_prefix, f"encodings.pt"),
)

