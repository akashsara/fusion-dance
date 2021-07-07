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
import models

seed = 42
np.random.seed(seed)
_ = torch.manual_seed(seed)

################################################################################
#################################### Config ####################################
################################################################################
learning_rate = 1e-4
epochs = 10
batch_size = 64
num_dataloader_workers = 0

experiment_name = f"fusion_cnn_prior_v1"

vq_vae_experiment_name = f"vq_vae_v3.6"
vq_vae_num_layers = 1
vq_vae_max_filters = 512
vq_vae_use_max_filters = False
vq_vae_num_embeddings = 128
vq_vae_embedding_dim = 32
vq_vae_commitment_cost = 0.25
vq_vae_small_conv = True  # To use the 1x1 convolution layer

image_size = 64
use_noise_images = True
input_channels = 6  # Two Images
output_dim = image_size // np.power(2, vq_vae_num_layers)

data_prefix = "data\\final\\standard"
fusion_data_prefix = "data\\final\\fusions"
output_prefix = f"data\\{experiment_name}"
vq_vae_model_prefix = f"outputs\\tbd\\{vq_vae_experiment_name}"

vq_vae_model_path = os.path.join(vq_vae_model_prefix, "model.pt")

train_data_folder = os.path.join(data_prefix, "train")
val_data_folder = os.path.join(data_prefix, "val")
test_data_folder = os.path.join(data_prefix, "test")

fusion_train_data_folder = os.path.join(fusion_data_prefix, "train")
fusion_val_data_folder = os.path.join(fusion_data_prefix, "val")
fusion_test_data_folder = os.path.join(fusion_data_prefix, "test")

output_dir = output_prefix
loss_output_path = os.path.join(output_prefix, "loss.jpg")
accuracy_output_path = os.path.join(output_prefix, "accuracy.jpg")
model_output_path = os.path.join(output_prefix, "model.pt")
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
####################################################################

# Preprocess & Create Data Loaders
transform = data.image2tensor_resize(image_size)

train_data = data.FusionDatasetV2(
    train_data_folder,
    fusion_train_data_folder,
    data_prefix,
    transform,
    use_noise_images,
)
val_data = data.FusionDatasetV2(
    val_data_folder, fusion_val_data_folder, data_prefix, transform, use_noise_images
)
test_data = data.FusionDatasetV2(
    test_data_folder, fusion_test_data_folder, data_prefix, transform, use_noise_images
)

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

################################################################################
##################################### Model ####################################
################################################################################

# Create & Load VQVAE Model
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

# Create Model
model = models.CNNPrior(
    input_channels=input_channels,
    output_channels=vq_vae_num_embeddings,
    input_dim=image_size,
    output_dim=output_dim,
)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

################################################################################
################################### Training ###################################
################################################################################

# Train
all_train_loss = []
all_val_loss = []
train_accuracy = []
val_accuracy = []

for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    epoch_train_accuracy = []
    epoch_val_accuracy = []

    # Training Loop
    model.train()
    for iteration, batch in enumerate(tqdm(train_dataloader)):
        # Reset gradients back to zero for this iteration
        optimizer.zero_grad()

        # Move batch to device
        _, (base, fusee, fusion) = batch  # (names), (images)
        base = base.to(device)
        fusee = fusee.to(device)
        fusion = fusion.to(device)
        current_batch_size = base.shape[0]

        with torch.no_grad():
            # Get Encodings from vq_vae
            _, _, _, encodings = vq_vae(fusion)
            # Reshape encodings
            y = encodings.reshape(current_batch_size, output_dim, output_dim)

        # Run our model & get outputs
        y_hat = model(torch.cat([base, fusee], dim=1))

        # Calculate reconstruction loss
        batch_loss = criterion(y_hat, y)

        # Backprop
        batch_loss.backward()

        # Update our optimizer parameters
        optimizer.step()

        # Calculate Accuracy
        y_hat = y_hat.argmax(dim=1)
        epoch_train_accuracy.extend(
            (y == y_hat).detach().flatten(start_dim=1).float().mean(dim=1).cpu()
        )

        # Add the batch's loss to the total loss for the epoch
        train_loss += batch_loss.item()

    # Validation Loop
    model.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(tqdm(val_dataloader)):
            # Move batch to device
            _, (base, fusee, fusion) = batch  # (names), (images)
            base = base.to(device)
            fusee = fusee.to(device)
            fusion = fusion.to(device)
            current_batch_size = base.shape[0]

            # Get Encodings from vq_vae
            _, _, _, encodings = vq_vae(fusion)
            # Reshape encodings
            y = encodings.reshape(current_batch_size, output_dim, output_dim)

            # Run our model & get outputs
            y_hat = model(torch.cat([base, fusee], dim=1))

            # Calculate reconstruction loss
            batch_loss = criterion(y_hat, y)

            # Calculate Accuracy
            y_hat = y_hat.argmax(dim=1)
            epoch_val_accuracy.extend(
                (y == y_hat).detach().flatten(start_dim=1).float().mean(dim=1).cpu()
            )

            # Add the batch's loss to the total loss for the epoch
            val_loss += batch_loss.item()

    # Compute the average losses for this epoch
    train_loss = train_loss / len(train_dataloader)
    all_train_loss.append(train_loss)

    val_loss = val_loss / len(val_dataloader)
    all_val_loss.append(val_loss)

    # Compute the average accuracy for this epoch
    epoch_train_accuracy = torch.stack(epoch_train_accuracy).mean()
    train_accuracy.append(epoch_train_accuracy)
    epoch_val_accuracy = torch.stack(epoch_val_accuracy).mean()
    val_accuracy.append(epoch_val_accuracy)

    # Print Metrics
    print(
        f"\nEpoch: {epoch+1}/{epochs}:\
        \nTrain Loss = {train_loss}\
        \nTrain Accuracy = {epoch_train_accuracy}\
        \nVal Loss = {val_loss}\
        \nVal Accuracy = {epoch_val_accuracy}"
    )

################################################################################
################################## Save & Test #################################
################################################################################
# Generate Loss Graph
graphics.draw_loss(all_train_loss, all_val_loss, loss_output_path, mode="autoencoder")
graphics.plot_and_save_loss(
    train_accuracy, "Train Accuracy", val_accuracy, "Val Accuracy", accuracy_output_path
)

# Save Model
torch.save(model.state_dict(), model_output_path)

model.eval()

# Evaluate on Test Images
# Testing Loop - Standard
all_image_accuracy = []
with torch.no_grad():
    for iteration, batch in enumerate(tqdm(test_dataloader)):
        # Move batch to device
        _, (base, fusee, fusion) = batch  # (names), (images)
        base = base.to(device)
        fusee = fusee.to(device)
        fusion = fusion.to(device)
        current_batch_size = base.shape[0]

        # Get Encodings from vq_vae
        _, _, _, encodings = vq_vae(fusion)
        # Reshape encodings
        y = encodings.reshape(current_batch_size, output_dim, output_dim)

        # Run our model & get outputs
        y_hat = model(torch.cat([base, fusee], dim=1))

        # Calculate Metrics
        y_hat = y_hat.argmax(dim=1)
        all_image_accuracy.extend(
            (y == y_hat).detach().flatten(start_dim=1).float().mean(dim=1).cpu()
        )

# Print Metrics
test_accuracy = torch.stack(all_image_accuracy).mean()
print(f"Test Accuracy = {test_accuracy}")