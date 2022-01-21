import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import utils.data as data
import utils.graphics as graphics
from models import vqvae
from transformers import GPT2LMHeadModel, GPT2Config

seed = 42
np.random.seed(seed)
_ = torch.manual_seed(seed)

################################################################################
#################################### Config ####################################
################################################################################

experiment_name = f"transformer_prior_v1"

# VQ-VAE Config
vq_vae_image_size = 64
vq_vae_experiment_name = f"vq_vae_v5.8"
vq_vae_num_layers = 1
vq_vae_max_filters = 512
vq_vae_use_max_filters = True
vq_vae_num_embeddings = 256
vq_vae_embedding_dim = 32
vq_vae_commitment_cost = 0.25
vq_vae_small_conv = True  # To use the 1x1 convolution layer

# Transformer Prior Config
image_size = vq_vae_image_size // (2 ** vq_vae_num_layers)
max_seq_length = image_size * image_size
vocab_size = vq_vae_num_embeddings
embedding_size = 768  # Default: 768
num_hidden_layers = 12  # Default: 12
num_attention_heads = 12  # Default: 12
resid_pdrop = 0.1 # Default: 0.1

# Generation Config
generation_batches = 5
generation_batch_size = 32
generation_temperature = 1
# If set to int > 0, all ngrams of that size can only occur once.
no_repeat_ngram_size = 0 
num_beams = 1 # Number of beam searches

# Hyperparameters
learning_rate = 1e-4
epochs = 10
batch_size = 32

# Data Config
num_dataloader_workers = 0
use_noise_images = False
data_prefix = "data\\pokemon\\final\\standard"
output_prefix = f"data\\{experiment_name}"
vq_vae_model_prefix = f"outputs\\{vq_vae_experiment_name}"
vq_vae_model_path = os.path.join(vq_vae_model_prefix, "model.pt")

train_data_folder = os.path.join(data_prefix, "train")
val_data_folder = os.path.join(data_prefix, "val")
test_data_folder = os.path.join(data_prefix, "test")

generated_dir = os.path.join(output_prefix, "generated")
model_output_path = os.path.join(output_prefix, "model.pt")

# Setup Device
gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")
print(gpu, device)

# Create Output Paths
if not os.path.exists(generated_dir):
    os.makedirs(generated_dir)

# Preprocessing Function
transform = data.image2tensor_resize(vq_vae_image_size)

# Creating a sample set to visualize the model's training
val_data = data.CustomDatasetNoMemory(val_data_folder, transform, use_noise_images)
sample = data.get_samples_from_data(val_data, 16)

# Load model
checkpoint = torch.load(model_output_path, map_location=device)

# Create & Load VQVAE Model
vq_vae = vqvae.VQVAE(
    num_layers=vq_vae_num_layers,
    input_image_dimensions=vq_vae_image_size,
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

# Create Model
configuration = GPT2Config(
    vocab_size=vocab_size,
    n_positions=max_seq_length,
    n_embd=embedding_size,
    n_layer=num_hidden_layers,
    n_head=num_attention_heads,
    resid_pdrop=resid_pdrop,
)
model = GPT2LMHeadModel(configuration)
model.eval()
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
print(model)

with torch.no_grad():
    # Get most common pixel values to feed into generation script
    _, _, _, encodings = vq_vae(sample.to(device))
    encodings, counts = encodings.unique(return_counts=True)
    bg1, bg2 = encodings[counts.topk(k=2, largest=True).indices].cpu().numpy()

    # Generate New Images
    for i in range(generation_batches):
        bg1_inputs = torch.zeros((generation_batch_size // 2, 1)).int() + bg1
        bg2_inputs = torch.zeros((generation_batch_size // 2, 1)).int() + bg2
        input_ids = torch.cat((bg1_inputs, bg2_inputs)).to(device)
        generated = model.generate(
            input_ids=input_ids,
            max_length=max_seq_length,
            temperature=generation_temperature,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=num_beams,
            do_sample=True,
        )
        target_shape = (
            generated.shape[0],
            image_size,
            image_size,
            vq_vae_embedding_dim,
        )
        generated = vq_vae.quantize_and_decode(
            generated.view(-1, 1), target_shape, device
        )
        generated = generated.permute(0, 2, 3, 1).detach().cpu().numpy()
        for j, image in enumerate(generated):
            filename = f"{(generation_batch_size * i) + j}.png"
            plt.imsave(os.path.join(generated_dir, filename), image)
