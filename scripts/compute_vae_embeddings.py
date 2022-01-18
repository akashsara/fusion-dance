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
experiment_name = "convolutional_vae_v16.5"
num_layers = 4
max_filters = 512
image_size = 64
latent_dim = 256
small_conv = True  # To use the 1x1 convolution layer

# Pokedex
pokedex_url = "https://raw.githubusercontent.com/lgreski/pokemonData/master/Pokemon.csv"
pokedex = pd.read_csv(pokedex_url).drop_duplicates(['ID', 'Name'])
pokedex.index = pokedex.ID
pokemon_types = pokedex['Type1'].to_dict()
pokedex = pokedex['Name'].to_dict()

# Data Config
batch_size = 32
num_dataloader_workers = 0
background = (255, 255, 255)
data_folder = "data\\pokemon\\original_data"

model_prefix = f"outputs\\{experiment_name}"
model_path = os.path.join(model_prefix, "model.pt")

# Setup Device
gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")
print(gpu, device)

# Preprocess & Create Data Loaders
transform = data.image2tensor_resize(image_size)

# Load Data
data_in = data.CustomDatasetNoMemoryAddBackground(data_folder, transform, background)

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
all_pokemon_types = []
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
            filename = filename.split("_")[0]
            pokemon_id = filename.split("-")[0].split(".")[0]
            filename = filename.replace(pokemon_id, pokedex[int(pokemon_id)])
            all_filenames.append(filename)
            all_pokemon_types.append(pokemon_types[int(pokemon_id)])

# Save to file
all_embeddings = torch.cat(all_embeddings).detach().cpu()
torch.save(
    {"embeddings": all_embeddings, "filenames": all_filenames, "pokemon_types": all_pokemon_types},
    os.path.join(model_prefix, f"embeddings.pt"),
)






