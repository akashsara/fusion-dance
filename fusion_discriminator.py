import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pytorch_msssim
from tqdm import tqdm
from sklearn.metrics import classification_report

import utils.data as data
import utils.graphics as graphics
import utils.loss as loss
from models import cnn_discriminator

seed = 42
np.random.seed(seed)
_ = torch.manual_seed(seed)

################################################################################
#################################### Config ####################################
################################################################################
learning_rate = 1e-3
epochs = 25
batch_size = 64
num_dataloader_workers = 0

experiment_name = f"fusion_discriminator_v1"

image_size = 64
use_noise_images = True

input_channels = 3
input_dim = image_size
num_filters = 512
num_layers = 4

data_prefix = "data\\\cnn_prior_predictions"
output_prefix = f"data\\{experiment_name}"

train_data_folder = os.path.join(data_prefix, "train")
val_data_folder = os.path.join(data_prefix, "val")
test_data_folder = os.path.join(data_prefix, "test")

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
####################################################################

# Preprocess & Create Data Loaders
transform = data.image2tensor_resize(image_size)

train_data = data.CustomDatasetV2(train_data_folder, transform)
val_data = data.CustomDatasetV2(val_data_folder, transform)
test_data = data.CustomDatasetV2(test_data_folder, transform)

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

# Create Model
model = cnn_discriminator.CNNDiscriminator(
    input_channels=input_channels,
    input_dim=input_dim,
    num_filters=num_filters,
    num_layers=num_layers,
)
model.to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.BCEWithLogitsLoss()

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

        # Get data
        labels, batch = batch
        # Convert filenames into labels
        labels = torch.Tensor(["generated" in x for x in labels])
        # 1 = Real; 0 = Generated
        labels = torch.logical_not(labels).float().unsqueeze(1)
        # Move to device
        labels = labels.to(device)
        batch = batch.to(device)
        current_batch_size = batch.shape[0]

        # Run our model & get outputs
        predictions = model(batch)

        # Calculate Loss
        batch_loss = criterion(predictions, labels)

        # Backprop
        batch_loss.backward()

        # Update our optimizer parameters
        optimizer.step()

        # Add the batch's loss to the total loss for the epoch
        train_loss += batch_loss.item()

    # Validation Loop
    model.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(tqdm(val_dataloader)):
            # Get data
            labels, batch = batch
            # Convert filenames into labels
            labels = torch.Tensor(["generated" in x for x in labels])
            # 1 = Real; 0 = Generated
            labels = torch.logical_not(labels).float().unsqueeze(1)
            # Move to device
            labels = labels.to(device)
            batch = batch.to(device)
            current_batch_size = batch.shape[0]

            # Run our model & get outputs
            predictions = model(batch)

            # Calculate Loss
            batch_loss = criterion(predictions, labels)

            # Add the batch's loss to the total loss for the epoch
            val_loss += batch_loss.item()

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
y = []
y_hat = []

with torch.no_grad():
    for iteration, batch in enumerate(tqdm(test_dataloader)):
        # Get data
        labels, batch = batch
        # Convert filenames into labels
        labels = torch.Tensor(["generated" in x for x in labels])
        # 1 = Real; 0 = Generated
        labels = torch.logical_not(labels).float().unsqueeze(1)
        # Move to device
        labels = labels.to(device)
        batch = batch.to(device)
        current_batch_size = batch.shape[0]

        # Run our model & get outputs
        predictions = model(batch).sigmoid()

        # Store in Variable
        y.extend(labels.detach().cpu())
        y_hat.extend(predictions.detach().cpu())

# Precision/Recall/F-Score
y = torch.stack(y).numpy()
y_hat = (torch.stack(y_hat) > 0.5).float().numpy()
print(classification_report(y, y_hat, labels=[0, 1], target_names=["fake", "real"]))