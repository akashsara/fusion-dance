import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pytorch_msssim
from tqdm import tqdm

import utils.data as data
import utils.graphics as graphics
import utils.loss as loss
from models import vae, vqvae

seed = 42
np.random.seed(seed)
_ = torch.manual_seed(seed)

################################################################################
#################################### Config ####################################
################################################################################
learning_rate = 1e-4
epochs = 25
batch_size = 32
num_dataloader_workers = 0
use_noise_images = False

experiment_name = f"convolutional_vae_v10"

# VQ-VAE Config
mode = "discrete"
vq_vae_image_size = 64
vq_vae_experiment_name = f"vq_vae_v5.5"
vq_vae_num_layers = 2
vq_vae_max_filters = 512
vq_vae_use_max_filters = True
vq_vae_num_embeddings = 256
vq_vae_embedding_dim = 64
vq_vae_commitment_cost = 0.25
vq_vae_small_conv = True  # To use the 1x1 convolution layer

# VAE Prior Config
image_size = vq_vae_image_size // (2 ** vq_vae_num_layers)
num_layers = 2
max_filters = 512
latent_dim = 256
num_classes = vq_vae_num_embeddings
small_conv = False  # To use the 1x1 convolution layer
kl_d_weight = 1  # equivalent to beta in a Beta-VAE
use_sum = False

# Data Config
data_prefix = "data\\pokemon\\final\\standard"
output_prefix = f"data\\{experiment_name}"
vq_vae_model_prefix = f"outputs\\{vq_vae_experiment_name}"
vq_vae_model_path = os.path.join(vq_vae_model_prefix, "model.pt")

train_data_folder = os.path.join(data_prefix, "train")
val_data_folder = os.path.join(data_prefix, "val")
test_data_folder = os.path.join(data_prefix, "test")

output_dir = os.path.join(output_prefix, "generated")
loss_output_path = output_prefix
model_output_path = os.path.join(output_prefix, "model.pt")

animation_output_path = os.path.join(output_prefix, "animation.mp4")
animation_sample_image_name = os.path.join(output_prefix, "animation_base.jpg")

test_sample_input_name = os.path.join(output_prefix, "test_sample_input.jpg")
test_sample_output_name = os.path.join(output_prefix, "test_sample_output.jpg")
################################################################################
##################################### Setup ####################################
################################################################################

# Setup Device
gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")
print(gpu, device)

# Create Output Paths
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

################################################################################
################################## Data Setup ##################################
################################################################################

# Load Data
train = data.load_images_from_folder(train_data_folder, use_noise_images)
val = data.load_images_from_folder(val_data_folder, use_noise_images)
test = data.load_images_from_folder(test_data_folder, use_noise_images)

# Preprocess & Create Data Loaders
transform = data.image2tensor_resize(vq_vae_image_size)

train_data = data.CustomDataset(train, transform)
val_data = data.CustomDataset(val, transform)
test_data = data.CustomDataset(test, transform)

train_dataloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_dataloader_workers,
    pin_memory=gpu,
)
val_dataloader = torch.utils.data.DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_dataloader_workers,
    pin_memory=gpu,
)
test_dataloader = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_dataloader_workers,
    pin_memory=gpu,
)

# Creating a sample set to visualize the model's training
sample = data.get_samples_from_data(val_data, 16)

################################################################################
##################################### Model ####################################
################################################################################

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
model = vae.VAEPrior(
    input_dimensions=image_size,
    num_classes=num_classes,
    num_layers=num_layers,
    max_filters=max_filters,
    latent_dim=latent_dim,
    small_conv=small_conv,
)
model.to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
if use_sum:
    criterion = nn.CrossEntropyLoss(reduction='sum')
else:
    criterion = nn.CrossEntropyLoss()
################################################################################
################################### Training ###################################
################################################################################

# Train
all_samples = []
all_train_loss = []
all_val_loss = []

# Get an initial "epoch 0" sample
model.eval()
target_shape = (sample.shape[0], image_size, image_size, vq_vae_embedding_dim)
with torch.no_grad():
    _, _, _, encodings = vq_vae(sample.to(device))
    epoch_sample = encodings.reshape(-1, image_size, image_size)
    epoch_sample, _, _, _ = model(epoch_sample)
    epoch_sample = epoch_sample.argmax(dim=1, keepdim=True)
    epoch_sample = epoch_sample.flatten(start_dim=1).view(-1, 1)
    epoch_sample = vq_vae.quantize_and_decode(epoch_sample, target_shape, device)

# Add sample reconstruction to our list
all_samples.append(epoch_sample.detach().cpu())

for epoch in range(epochs):
    train_loss = 0
    train_recon_loss = 0
    train_kl_d = 0
    val_loss = 0
    val_recon_loss = 0
    val_kl_d = 0

    # Training Loop
    model.train()
    for iteration, batch in enumerate(tqdm(train_dataloader)):
        # Reset gradients back to zero for this iteration
        optimizer.zero_grad()

        # Move batch to device
        _, batch = batch  # Returns key, value for each Pokemon
        batch = batch.to(device)

        with torch.no_grad():
            # Get Encodings from vq_vae
            _, _, _, encodings = vq_vae(batch)
            original = encodings.reshape(-1, image_size, image_size)

        # Run our model & get outputs
        reconstructed, mu, log_var, _ = model(original)

        # Calculate loss
        reconstruction_loss = criterion(reconstructed, original)
        kl_d_loss = loss.kl_divergence(mu, log_var, use_sum=False)
        final_loss = reconstruction_loss + (kl_d_weight * kl_d_loss)

        # Backprop
        final_loss.backward()

        # Update our optimizer parameters
        optimizer.step()

        # Add the batch's loss to the total loss for the epoch
        train_loss += final_loss.item()
        train_recon_loss += reconstruction_loss.item()
        train_kl_d += kl_d_loss.item()

    # Validation Loop
    model.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(tqdm(val_dataloader)):
            # Move batch to device
            _, batch = batch  # Returns key, value for each Pokemon
            batch = batch.to(device)

            # Get Encodings from vq_vae
            _, _, _, encodings = vq_vae(batch)
            original = encodings.reshape(-1, image_size, image_size)

            # Run our model & get outputs
            reconstructed, mu, log_var, _ = model(original)

            # Calculate loss
            reconstruction_loss = criterion(reconstructed, original)
            kl_d_loss = loss.kl_divergence(mu, log_var, use_sum=False)
            final_loss = reconstruction_loss + (kl_d_weight * kl_d_loss)

            # Add the batch's loss to the total loss for the epoch
            val_loss += final_loss.item()
            val_recon_loss += reconstruction_loss.item()
            val_kl_d += kl_d_loss.item()

        # Get reconstruction of our sample
        target_shape = (sample.shape[0], image_size, image_size, vq_vae_embedding_dim)
        _, _, _, encodings = vq_vae(sample.to(device))
        epoch_sample = encodings.reshape(-1, image_size, image_size)
        epoch_sample, _, _, _ = model(epoch_sample)
        epoch_sample = epoch_sample.argmax(dim=1, keepdim=True)
        epoch_sample = epoch_sample.flatten(start_dim=1).view(-1, 1)
        epoch_sample = vq_vae.quantize_and_decode(epoch_sample, target_shape, device)

    # Add sample reconstruction to our list
    all_samples.append(epoch_sample.detach().cpu())

    # Compute the average losses for this epoch
    train_loss = train_loss / len(train_dataloader)
    train_recon_loss = train_recon_loss / len(train_dataloader)
    train_kl_d = train_kl_d / len(train_dataloader)
    all_train_loss.append((train_loss, train_recon_loss, train_kl_d))

    val_loss = val_loss / len(val_dataloader)
    val_recon_loss = val_recon_loss / len(val_dataloader)
    val_kl_d = val_kl_d / len(val_dataloader)
    all_val_loss.append((val_loss, val_recon_loss, val_kl_d))

    # Print Metrics
    print(
        f"\nEpoch: {epoch+1}/{epochs}:\
        \nTrain Loss = {train_loss}\
        \nTrain Reconstruction Loss = {train_recon_loss}\
        \nTrain KL Divergence = {train_kl_d}\
        \nVal Loss = {val_loss}\
        \nVal Reconstruction Loss = {val_recon_loss}\
        \nVal KL Divergence = {val_kl_d}"
    )

################################################################################
################################## Save & Test #################################
################################################################################
# Generate Loss Graph
graphics.draw_loss(all_train_loss, all_val_loss, loss_output_path, mode="vae")

# Save Model
torch.save(
    {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": all_train_loss,
        "val_loss": all_val_loss,
    },
    model_output_path,
)

# Plot Animation Sample
fig, axis = graphics.make_grid(("Sample", sample), 4, 4)
plt.savefig(animation_sample_image_name)

# Create & Save Animation
anim = graphics.make_animation(graphics.make_grid, all_samples)
anim.save(animation_output_path)

model.eval()

# Evaluate on Test Images
# Save Generated Images & Calculate Metrics
# Testing Loop - Standard
test_loss = 0
test_reconstruction = 0
test_kl_d = 0
with torch.no_grad():
    for iteration, batch in enumerate(tqdm(test_dataloader)):
        # Move batch to device
        filenames, batch = batch  # Returns key, value for each Pokemon
        batch = batch.to(device)

        # Get Encodings from vq_vae
        _, _, _, encodings = vq_vae(batch)
        original = encodings.reshape(-1, image_size, image_size)

        # Run our model & get outputs
        reconstructed, mu, log_var, _ = model(original)

        # Calculate loss
        reconstruction_loss = criterion(reconstructed, original)
        kl_d_loss = loss.kl_divergence(mu, log_var, use_sum=False)
        final_loss = reconstruction_loss + (kl_d_weight * kl_d_loss)

        # Add the batch's loss to the total loss for the epoch
        test_reconstruction += reconstruction_loss.item()
        test_kl_d += kl_d_loss.item()
        test_loss += final_loss.item()

        # Save
        target_shape = (
            reconstructed.shape[0],
            image_size,
            image_size,
            vq_vae_embedding_dim,
        )
        reconstructed = reconstructed.argmax(dim=1, keepdim=True)
        reconstructed = reconstructed.flatten(start_dim=1).view(-1, 1)
        reconstructed = vq_vae.quantize_and_decode(reconstructed, target_shape, device)
        reconstructed = reconstructed.permute(0, 2, 3, 1).detach().cpu().numpy()
        for image, filename in zip(reconstructed, filenames):
            plt.imsave(os.path.join(output_dir, filename), image)

# Print Metrics
test_loss = test_loss / len(test_dataloader)
test_reconstruction = test_reconstruction / len(test_dataloader)
test_kl_d = test_kl_d / len(test_dataloader)
print(
    f"\nTest Loss:\
        \nTest Loss = {test_loss}\
        \nTest Reconstruction Loss = {test_reconstruction}\
        \nTest KL Divergence = {test_kl_d}"
)

# Pick a couple of sample images for an Input v Output comparison
test_sample = data.get_samples_from_data(test_data, 16)

# Plot A Set of Test Images
fig, axis = graphics.make_grid(("Test Sample", test_sample), 4, 4)
plt.savefig(test_sample_input_name)

target_shape = (test_sample.shape[0], image_size, image_size, vq_vae_embedding_dim)
with torch.no_grad():
    _, _, _, encodings = vq_vae(test_sample.to(device))
    test_sample = encodings.reshape(-1, image_size, image_size)
    test_sample, _, _, _ = model(test_sample)
    test_sample = test_sample.argmax(dim=1, keepdim=True)
    test_sample = test_sample.flatten(start_dim=1).view(-1, 1)
    test_sample = vq_vae.quantize_and_decode(test_sample, target_shape, device)
    reconstructed = test_sample.detach().cpu()

# Plot A Set of Reconstructed Test Images
fig, axis = graphics.make_grid(("Test Sample", reconstructed), 4, 4)
plt.savefig(test_sample_output_name)
