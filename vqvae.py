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
epochs = 25
batch_size = 64
num_dataloader_workers = 0

experiment_name = f"vq_vae_v1"

num_layers = 2
num_embeddings = 512
embedding_dim = 64
commitment_cost = 0.25
image_size = 64
use_noise_images = True
small_conv = True  # To use the 1x1 convolution layer
use_sum = False  # Use a sum instead of a mean for our loss function
use_ssim_loss = False
mse_weight = 1
ssim_weight = 1
num_mse = 0  # Each increment halves the image sizes and takes the MSE

data_prefix = "data\\final\\standard"
train_data_folder = os.path.join(data_prefix, "train")
val_data_folder = os.path.join(data_prefix, "val")
test_data_folder = os.path.join(data_prefix, "test")

output_prefix = f"data\\{experiment_name}"

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
transform = data.image2tensor_resize(image_size)

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

# Create Model
model = models.VQVAE(
    num_layers=num_layers,
    input_image_dimensions=image_size,
    small_conv=small_conv,
    embedding_dim=embedding_dim,
    num_embeddings=num_embeddings,
    commitment_cost=commitment_cost,
)
print(model)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

ssim_module = None
if use_ssim_loss:
    ssim_module = pytorch_msssim.SSIM(
        data_range=1.0, win_size=11, win_sigma=1.5, K=(0.01, 0.03)
    )

################################################################################
################################### Training ###################################
################################################################################

# Train
all_samples = []
all_train_loss = []
train_perplexity = []
all_val_loss = []
val_perplexity = []

# Get an initial "epoch 0" sample
model.eval()
with torch.no_grad():
    _, epoch_sample, _, _ = model(sample.to(device))

# Add sample reconstruction to our list
all_samples.append(epoch_sample.detach().cpu())

for epoch in range(epochs):
    train_loss = 0
    train_recon_loss = 0
    train_vq_loss = 0
    train_epoch_perplexity = []
    val_loss = 0
    val_recon_loss = 0
    val_vq_loss = 0
    val_epoch_perplexity = []

    # Training Loop
    model.train()
    for iteration, batch in enumerate(tqdm(train_dataloader)):
        # Reset gradients back to zero for this iteration
        optimizer.zero_grad()

        # Move batch to device
        _, batch = batch  # Returns key, value for each Pokemon
        batch = batch.to(device)

        # Run our model & get outputs
        vq_loss, reconstructed, perplexity, _ = model(batch)

        # Calculate reconstruction loss
        batch_loss, loss_dict = loss.mse_ssim_loss(
            reconstructed,
            batch,
            use_sum=use_sum,
            ssim_module=ssim_module,
            mse_weight=mse_weight,
            ssim_weight=ssim_weight,
        )

        # Add VQ-Loss to Overall Loss
        batch_loss += vq_loss
        loss_dict["Commitment Loss"] = vq_loss.item()

        # For multiple MSE
        # For every MSE, we halve the image size
        # And take the MSE between the resulting images
        for i in range(num_mse):
            new_size = image_size // pow(2, i + 1)
            with torch.no_grad():
                resized_batch = nn.functional.interpolate(
                    batch, size=new_size, mode="bilinear"
                )
            resized_output = nn.functional.interpolate(
                reconstructed, size=new_size, mode="bilinear"
            )
            mse = loss.mse_loss(resized_output, resized_batch, use_sum)
            batch_loss += mse
            loss_dict["MSE"] += mse.item()

        # Backprop
        batch_loss.backward()

        # Update our optimizer parameters
        optimizer.step()

        # Add the batch's loss to the total loss for the epoch
        train_loss += batch_loss.item()
        train_recon_loss += loss_dict["MSE"] + loss_dict["SSIM"]
        train_vq_loss += loss_dict["Commitment Loss"]
        train_epoch_perplexity.append(perplexity.item())

    # Validation Loop
    model.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(tqdm(val_dataloader)):
            # Move batch to device
            _, batch = batch  # Returns key, value for each Pokemon
            batch = batch.to(device)

            # Run our model & get outputs
            vq_loss, reconstructed, perplexity, _ = model(batch)

            # Calculate reconstruction loss
            batch_loss, loss_dict = loss.mse_ssim_loss(
                reconstructed,
                batch,
                use_sum=use_sum,
                ssim_module=ssim_module,
                mse_weight=mse_weight,
                ssim_weight=ssim_weight,
            )

            # Add VQ-Loss to Overall Loss
            batch_loss += vq_loss
            loss_dict["Commitment Loss"] = vq_loss.item()

            # For multiple MSE
            # For every MSE, we halve the image size
            # And take the MSE between the resulting images
            for i in range(num_mse):
                new_size = image_size // pow(2, i + 1)
                resized_batch = nn.functional.interpolate(
                    batch, size=new_size, mode="bilinear"
                )
                resized_output = nn.functional.interpolate(
                    reconstructed, size=new_size, mode="bilinear"
                )
                mse = loss.mse_loss(resized_output, resized_batch, use_sum)
                batch_loss += mse
                loss_dict["MSE"] += mse.item()
            # Add the batch's loss to the total loss for the epoch
            val_loss += batch_loss.item()
            val_recon_loss += loss_dict["MSE"] + loss_dict["SSIM"]
            val_vq_loss += loss_dict["Commitment Loss"]
            val_epoch_perplexity.append(perplexity.item())

        # Get reconstruction of our sample
        _, epoch_sample, _, _ = model(sample.to(device))

    # Add sample reconstruction to our list
    all_samples.append(epoch_sample.detach().cpu())

    # Compute the average losses for this epoch
    train_loss = train_loss / len(train_dataloader)
    train_recon_loss = train_recon_loss / len(train_dataloader)
    train_vq_loss = train_vq_loss / len(train_dataloader)
    all_train_loss.append((train_loss, train_recon_loss, train_vq_loss))
    train_epoch_perplexity = np.mean(train_epoch_perplexity)
    train_perplexity.append(train_epoch_perplexity)

    val_loss = val_loss / len(val_dataloader)
    val_recon_loss = val_recon_loss / len(val_dataloader)
    val_vq_loss = val_vq_loss / len(val_dataloader)
    all_val_loss.append((val_loss, val_recon_loss, val_vq_loss))
    val_epoch_perplexity = np.mean(val_epoch_perplexity)
    val_perplexity.append(val_epoch_perplexity)

    # Print Metrics
    print(
        f"\nEpoch: {epoch+1}/{epochs}:\
        \nTrain Loss = {train_loss}\
        \nTrain Reconstruction Loss = {train_recon_loss}\
        \nTrain Commitment Loss = {train_vq_loss}\
        \nTrain Perplexity = {train_epoch_perplexity}\
        \nVal Loss = {val_loss}\
        \nVal Reconstruction Loss = {val_recon_loss}\
        \nVal Commitment Loss = {val_vq_loss}\
        \nVal Perplexity = {val_epoch_perplexity}"
    )

################################################################################
################################## Save & Test #################################
################################################################################
# Generate Loss Graph
graphics.draw_loss(all_train_loss, all_val_loss, loss_output_path, mode="vqvae")
graphics.plot_and_save_loss(
    train_perplexity,
    "Train Perplexity",
    val_perplexity,
    "Validation Perplexity",
    os.path.join(loss_output_path, "perplexity.jpg"),
)

# Save Model
torch.save(model.state_dict(), model_output_path)

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
all_mse = []
all_ssim = []
with torch.no_grad():
    for iteration, batch in enumerate(tqdm(test_dataloader)):
        # Move batch to device
        filenames, image = batch
        image = image.to(device)

        # Run our model & get outputs
        _, reconstructed, _, _ = model(image)

        # Calculate Metrics
        mse = nn.functional.mse_loss(reconstructed, image)
        ssim_score = pytorch_msssim.ssim(
            reconstructed,
            image,
            data_range=1.0,
            win_size=11,
            win_sigma=1.5,
            K=(0.01, 0.03),
        )

        # Add metrics to tracking list
        all_mse.append(mse.detach().cpu().numpy())
        all_ssim.append(ssim_score.detach().cpu().numpy())

        # Save
        reconstructed = reconstructed.permute(0, 2, 3, 1).detach().cpu().numpy()
        for image, filename in zip(reconstructed, filenames):
            plt.imsave(os.path.join(output_dir, filename), image)

# Print Metrics
mse = np.asarray(all_mse).mean()
ssim_score = np.asarray(all_ssim).mean()
print(f"\nMSE = {mse}, SSIM = {ssim_score}")

# Pick a couple of sample images for an Input v Output comparison
test_sample = data.get_samples_from_data(test_data, 16)

# Plot A Set of Test Images
fig, axis = graphics.make_grid(("Test Sample", test_sample), 4, 4)
plt.savefig(test_sample_input_name)

with torch.no_grad():
    reconstructed = model(test_sample.to(device))[1].detach().cpu()

# Plot A Set of Reconstructed Test Images
fig, axis = graphics.make_grid(("Test Sample", reconstructed), 4, 4)
plt.savefig(test_sample_output_name)