"""
Generates an arbitrary number of outputs from the given conditional Pixel CNN
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import sys

sys.path.append("./")
import utils.data as data
from models import vqvae, gated_pixelcnn

seed = 42
np.random.seed(seed)
_ = torch.manual_seed(seed)

################################################################################
#################################### Config ####################################
################################################################################

# VQ-VAE Config
vq_vae_model_name = f"vq_vae_v5.8"
vq_vae_model_config = {
    "num_layers": 1,
    "input_image_dimensions": 64,
    "small_conv": True,
    "embedding_dim": 32,
    "num_embeddings": 256,
    "commitment_cost": 0.25,
    "use_max_filters": True,
    "max_filters": 512,
}
vq_vae_model_path = os.path.join(f"outputs\\{vq_vae_model_name}", "model.pt")

# Pixel CNN Config
model_name = f"conditional_gated_pixelcnn_v1"
model_config = {
    "c_in": 1,
    "c_hidden": 256,
    "num_classes": vq_vae_model_config["num_embeddings"],
    "kernel_size": 3,
    "use_dilation": True,
}
model_path = os.path.join(f"outputs\\{model_name}", "model.pt")

# Generation Config
image_size = 64
num_generations = 10000
batch_size = 32
num_sample_batches = (num_generations // batch_size) + 1
input_dim = image_size // (2 ** vq_vae_model_config["num_layers"])
conditioning_info_file = "data\\Pokemon\\metadata.joblib"
conditioning_info_columns = ["type1", "type2", "shape"]
sample_conditioning_dict = {"type1": 1, "type2": 2, "shape": 1}
output_dir = ""

################################################################################
#################################### Setup $####################################
################################################################################

# Setup Device
gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")
print(gpu, device)

# Create Output Paths
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create & Load VQVAE Model
vq_vae = vqvae.VQVAE(**vq_vae_model_config)
vq_vae.load_state_dict(torch.load(vq_vae_model_path, map_location=device))
vq_vae.eval()
vq_vae.to(device)

checkpoint = torch.load(model_path, map_location=device)
label_handler = data.ConditioningLabelsHandlerFromSaved(conditioning_info_file, conditioning_info_columns, checkpoint["encoding_dict"])
model_config["conditioning_size"] = label_handler.get_size()

# Create Model
model = gated_pixelcnn.ConditionalPixelCNN(**model_config)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

################################################################################
################################### Generate ###################################
################################################################################

# Generate samples
target_shape = (batch_size, input_dim, input_dim, vq_vae_model_config["embedding_dim"])
image_shape = (batch_size, model_config["c_in"], input_dim, input_dim)
for i in tqdm(range(num_sample_batches)):
    with torch.no_grad():
        # Pick some random conditioning info
        conditioning_info = label_handler.sample_conditions(batch_size, sample_conditioning_dict)
        conditioning_info = torch.as_tensor(conditioning_info).float().to(device)
        # Sample from model
        sample = model.sample(image_shape, device, conditioning_info)
        # Feed into VQ-VAE
        sample = sample.flatten(start_dim=1).view(-1, 1)
        sample = vq_vae.quantize_and_decode(sample, target_shape, device)
        # Convert to image
        sample = sample.permute(0, 2, 3, 1).detach().cpu().numpy()
    # Save
    for filename, (image, condition) in enumerate(zip(sample, conditioning_info)):
        conditions = (condition == condition.max()).nonzero().flatten()
        conditions = "-".join([label_handler.reverse_transform(int(condition)) for condition in conditions])
        filename = f"{(i*batch_size)+filename}_{conditions}.png"
        plt.imsave(os.path.join(output_dir, filename), image)
