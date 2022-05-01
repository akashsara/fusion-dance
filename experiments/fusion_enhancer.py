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
from models import cnn_enhancer
seed = 42
np.random.seed(seed)
_ = torch.manual_seed(seed)

################################################################################
#################################### Config ####################################
################################################################################
learning_rate = 1e-4
epochs = 1
batch_size = 64
num_dataloader_workers = 0

image_size = 64
use_noise_images = True

use_ssim_loss = False
mse_weight = 1
ssim_weight = 1

# NOTE:
# Input: (image_size, image_size, 3)
# Output: (image_size, image_size, 3)
# So to ensure that this is possible, there have to be a minimum of 2 layers

experiment_name = f"fusion_enhancer_v1"
input_channels = 3
num_filters = 128
num_layers = 2  # Must be at least 2
use_4by4_conv = True

features_prefix = "data\\pokemon\\experiments\\cnn_multirnn_v2_outputs\\fake"
labels_prefix = "data\\pokemon\\experiments\\cnn_multirnn_v2_outputs\\real"
output_prefix = f"data\\{experiment_name}"

train_features = os.path.join(features_prefix, "train")
val_features = os.path.join(features_prefix, "val")
test_features = os.path.join(features_prefix, "test")

train_labels = os.path.join(labels_prefix, "train")
val_labels = os.path.join(labels_prefix, "val")
test_labels = os.path.join(labels_prefix, "test")

output_dir = os.path.join(output_prefix, "generated")
loss_output_path = os.path.join(output_prefix, "loss.jpg")
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

# Preprocess & Create Data Loaders
transform = data.image2tensor_resize(image_size)

train_data = data.CustomImage2ImageDatasetWithLabels(train_features, train_labels, transform)
val_data = data.CustomImage2ImageDatasetWithLabels(val_features, val_labels, transform)
test_data = data.CustomImage2ImageDatasetWithLabels(test_features, test_labels, transform)

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

# Create Enhancer
model = cnn_enhancer.ImageEnhancerCNN(input_channels, num_filters, num_layers, use_4by4_conv)
model.to(device)
print(model)

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
all_train_loss = []
all_val_loss = []

for epoch in range(epochs):
    train_loss = 0
    val_loss = 0

    # Training Loop
    model.train()
    for iteration, batch in enumerate(tqdm(train_dataloader)):
        # Reset gradients back to zero for this iteration
        optimizer.zero_grad()

        # Move batch to device
        _, features, labels = batch  # (names), (images)
        features = features.to(device)
        labels = labels.to(device)
        current_batch_size = features.shape[0]

        # Run Model & Get Output
        predictions = model(features)

        # Calculate Loss
        batch_loss, loss_dict = loss.mse_ssim_loss(
            predictions,
            labels,
            use_sum=False,
            ssim_module=ssim_module,
            mse_weight=mse_weight,
            ssim_weight=ssim_weight,
        )

        # Backprop
        batch_loss.backward()

        # Update our optimizer parameters
        optimizer.step()

        # Add the batch's loss to the total loss for the epoch
        train_loss += loss_dict["MSE"] + loss_dict["SSIM"]

    # Validation Loop
    model.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(tqdm(val_dataloader)):
            # Move batch to device
            _, features, labels = batch  # (names), (images)
            features = features.to(device)
            labels = labels.to(device)
            current_batch_size = features.shape[0]

            # Run Model & Get Output
            predictions = model(features)

            # Calculate Loss
            batch_loss, loss_dict = loss.mse_ssim_loss(
                predictions,
                labels,
                use_sum=False,
                ssim_module=ssim_module,
                mse_weight=mse_weight,
                ssim_weight=ssim_weight,
            )

            # Add the batch's loss to the total loss for the epoch
            val_loss += loss_dict["MSE"] + loss_dict["SSIM"]

    # Compute the average losses for this epoch
    train_loss = train_loss / len(train_dataloader)
    all_train_loss.append(train_loss)

    val_loss = val_loss / len(val_dataloader)
    all_val_loss.append(val_loss)

    # Print Metrics
    print(
        f"\nEpoch: {epoch+1}/{epochs}:\
        \nTrain Loss = {train_loss}\
        \nVal Loss = {val_loss}"
    )

################################################################################
################################## Save & Test #################################
################################################################################
# Generate Loss Graph
graphics.draw_loss(all_train_loss, all_val_loss, loss_output_path, mode="autoencoder")

# Save Model
torch.save(model.state_dict(), model_output_path)

model.eval()

# Evaluate on Test Images
# Testing Loop - Standard
mse_loss = torch.nn.MSELoss(reduction="none")
ssim_module = pytorch_msssim.SSIM(
    data_range=1.0, win_size=11, win_sigma=1.5, K=(0.01, 0.03), size_average=False
)

all_mse = []
all_ssim = []

with torch.no_grad():
    for iteration, batch in enumerate(tqdm(test_dataloader)):
        # Move batch to device
        filenames, features, labels = batch  # (names), (images)
        features = features.to(device)
        labels = labels.to(device)
        current_batch_size = features.shape[0]

        # Run Model & Get Output
        predictions = model(features)

        # Calculate MSE
        mse = mse_loss(predictions, labels).flatten(start_dim=1).mean(dim=1)
        all_mse.extend(mse.detach().cpu())

        # Calculate SSIM
        ssim = ssim_module(predictions, labels)
        all_ssim.extend(ssim.detach().cpu())

        # Save
        predictions = predictions.permute(0, 2, 3, 1).detach().cpu().numpy()
        for image, filename in zip(predictions, filenames):
            plt.imsave(os.path.join(output_dir, filename), image)

all_mse = torch.stack(all_mse).mean()
all_ssim = torch.stack(all_ssim).mean()
print(f"Test MSE = {all_mse}")
print(f"Test SSIM = {all_ssim}")