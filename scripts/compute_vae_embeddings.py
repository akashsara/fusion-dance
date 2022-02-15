import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append("./")
import utils.data as data
from models import vae

seed = 42
np.random.seed(seed)
_ = torch.manual_seed(seed)

################################################################################
#################################### Config ####################################
################################################################################

# VAE Config
experiment_name = "base_convolutional_vae_v1"
num_layers = 4
max_filters = 512
image_size = 64
latent_dim = 256
small_conv = False  # To use the 1x1 convolution layer

# Data Config
batch_size = 32
num_dataloader_workers = 0
background = (255, 255, 255)
dataset = "pokemon" # "pokemon", "tinyhero", "sprites"
data_folder_lookup = {
    "pokemon": "data\\pokemon\\original_data",
    "tinyhero": "data\\TinyHero\\processed",
    "sprites": "data\Sprites\processed"
}
data_folder = data_folder_lookup[dataset]

# Pokedex
pokedex_url = "data\\Pokemon\\pokedex_(Update_04.21).csv"

# Pokedex
if "pokemon" in dataset:
    pokedex = pd.read_csv(pokedex_url)
    for forbidden in ["Mega ", "Partner ", "Alolan ", "Galarian "]:
        pokedex.drop(pokedex[pokedex['name'].str.contains(forbidden)].index, inplace=True)
    pokedex.drop_duplicates(["pokedex_number", "name"], inplace=True)
    pokedex.index = pokedex.pokedex_number
    names = pokedex["name"].to_dict()
    height = pokedex["height_m"].to_dict()
    weight = pokedex["weight_kg"].to_dict()
    type1 = pokedex["type_1"].to_dict()
    type2 = pokedex["type_2"].to_dict()
    egg1 = pokedex["egg_type_1"].to_dict()
    egg2 = pokedex["egg_type_2"].to_dict()

model_prefix = f"outputs\\{experiment_name}"
model_path = os.path.join(model_prefix, "model.pt")

# Setup Device
gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")
print(gpu, device)

# Preprocess & Create Data Loaders
transform = data.image2tensor_resize(image_size)

# Load Data
data_in = data.CustomDatasetNoMemoryAddBackground(data_folder, dataset, transform, background)

input_dataloader = torch.utils.data.DataLoader(
    data_in,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_dataloader_workers,
    pin_memory=gpu,
)

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

all_embeddings = []
all_filenames = []
all_color_ids = []
with torch.no_grad():
    for iteration, batch in enumerate(tqdm(input_dataloader)):
        # Move batch to device
        filenames, batch = batch  # Returns key, value for each Pokemon
        batch = batch.to(device)

        # Get Encodings from vae
        current_batch_size = len(batch)
        mu, log_var = model.get_latent_variables(batch)
        embeddings = model.reparameterize(mu, log_var)

        # Add embeddings to list
        all_embeddings.append(embeddings)

        # Add filenames (with Pokemon name instead of ID) to list
        for filename in filenames:
            if "pokemon" in dataset:
                filename = filename.split("_")[0]
                pokemon_id = filename.split("-")[0].split(".")[0]
                pokemon_id_int = int(pokemon_id)
                filename = filename.replace(pokemon_id, names[pokemon_id_int])
                colors = [
                    height[pokemon_id_int], 
                    weight[pokemon_id_int], 
                    type1[pokemon_id_int],
                    type2[pokemon_id_int],
                    egg1[pokemon_id_int],
                    egg2[pokemon_id_int],
                ]
                all_color_ids.append(colors)
            elif dataset == "tinyhero":
                color = filename.split('.')[0].split('_')[1]
                all_color_ids.append(color)
            elif dataset == "sprites":
                id, pose, anim = filename.split('.')[0].split('_')
                all_color_ids.append([id, pose, anim])
            all_filenames.append(filename)

# Save to file
all_embeddings = torch.cat(all_embeddings).detach().cpu()
torch.save(
    {"embeddings": all_embeddings, "filenames": all_filenames, "color": all_color_ids},
    os.path.join(model_prefix, f"embeddings.pt"),
)
