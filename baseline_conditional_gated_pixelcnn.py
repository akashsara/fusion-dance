import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import utils.data as data
import utils.graphics as graphics
import utils.loss as loss
from models import vqvae, gated_pixelcnn

seed = 42
np.random.seed(seed)
_ = torch.manual_seed(seed)

################################################################################
#################################### Config ####################################
################################################################################
learning_rate = 1e-4
epochs = 10
batch_size = 32
num_dataloader_workers = 0
image_size = 64
use_noise_images = False
load_data_to_memory = False

experiment_name = f"conditional_gated_pixelcnn_v1"

# Pixel CNN Config
input_dim = image_size
input_channels = 3
hidden_channels = 128
num_classes = 256 # RGB = 0-255
kernel_size = 3
use_bits_per_dimension_loss = False
use_dilation = True

conditioning_info_columns = ["type1", "type2", "shape"]
sample_batch_size = 4
num_sample_batches = 1
sample_conditioning_dict = {"type1": 1, "type2": 2, "shape": 1}

# Data Config
conditioning_info_file = "data\\Pokemon\\metadata.joblib"
data_prefix = "data\\Pokemon\\final\\standard"
output_prefix = f"data\\{experiment_name}"

train_data_folder = os.path.join(data_prefix, "train")
val_data_folder = os.path.join(data_prefix, "val")
test_data_folder = os.path.join(data_prefix, "test")

output_dir = os.path.join(output_prefix, "generated")
loss_output_path = output_prefix
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

if load_data_to_memory:
    # Load Data
    train = data.load_images_from_folder(train_data_folder, use_noise_images)
    val = data.load_images_from_folder(val_data_folder, use_noise_images)
    test = data.load_images_from_folder(test_data_folder, use_noise_images)

    train_data = data.CustomDataset(train, transform)
    val_data = data.CustomDataset(val, transform)
    test_data = data.CustomDataset(test, transform)
else:
    train_data = data.CustomDatasetNoMemory(
        train_data_folder, transform, use_noise_images
    )
    val_data = data.CustomDatasetNoMemory(
        val_data_folder, transform, use_noise_images
    )
    test_data = data.CustomDatasetNoMemory(
        test_data_folder, transform, use_noise_images
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

# Creating a sample set to visualize the model's training
sample = data.get_samples_from_data(val_data, 16)

label_handler = data.ConditioningLabelsHandler(
    conditioning_info_file, conditioning_info_columns
)
conditioning_classes = label_handler.get_size()

################################################################################
##################################### Model ####################################
################################################################################

# Create Model
model = gated_pixelcnn.ConditionalPixelCNN(
    c_in=input_channels,
    c_hidden=hidden_channels,
    num_classes=num_classes,
    conditioning_size=conditioning_classes,
    kernel_size=kernel_size,
    use_dilation=use_dilation,
)
model.to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
if use_bits_per_dimension_loss:
    criterion = loss.bits_per_dimension_loss
else:
    criterion = nn.CrossEntropyLoss()

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
        key, batch = batch  # (names), (images)
        batch = batch.to(device)
        conditioning = torch.as_tensor(label_handler(key)).float().to(device)
        current_batch_size = batch.shape[0]

        # Run our model & get outputs
        y_hat = model(batch, conditioning)
        with torch.no_grad():
            batch = (batch * 255).clamp(0, 1).long()
        batch_loss = criterion(y_hat, batch)

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
            # Move batch to device
            key, batch = batch  # (names), (images)
            batch = batch.to(device)
            conditioning = torch.as_tensor(label_handler(key)).float().to(device)
            current_batch_size = batch.shape[0]

            # Run our model & get outputs
            y_hat = model(batch, conditioning)
            batch = (batch * 255).clamp(0, 1).long()
            batch_loss = criterion(y_hat, batch)

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
torch.save(
    {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": all_train_loss,
        "val_loss": all_val_loss,
        "encoding_dict": label_handler.encoding_dict,
    },
    model_output_path,
)

# Evaluation Time
model.eval()

# Compute bpd loss on test images
test_loss = 0
with torch.no_grad():
    for iteration, batch in enumerate(tqdm(test_dataloader)):
        # Move batch to device
        key, batch = batch  # (names), (images)
        batch = batch.to(device)
        conditioning = torch.as_tensor(label_handler(key)).float().to(device)
        current_batch_size = batch.shape[0]

        # Run our model & get outputs
        y_hat = model(batch, conditioning)
        batch = (batch * 255).clamp(0, 1).long()
        batch_loss = criterion(y_hat, batch)

        # Add the batch's loss to the total loss for the epoch
        test_loss += batch_loss.item()
test_loss = test_loss / len(test_dataloader)
print(f"Test Loss: {test_loss}")

# Generate samples
image_shape = (sample_batch_size, input_channels, input_dim, input_dim)
for i in range(num_sample_batches):
    # Pick some random conditioning info
    conditioning_info = label_handler.sample_conditions(sample_batch_size, sample_conditioning_dict)
    conditioning_info = torch.as_tensor(conditioning_info).float().to(device)
    with torch.no_grad():
        # Sample from model
        sample = model.sample(image_shape, device, conditioning_info) / 255.0
    # Convert to image
    sample = sample.permute(0, 2, 3, 1).detach().cpu().numpy()
    # Save
    for filename, (image, condition) in enumerate(zip(sample, conditioning_info)):
        conditions = (condition == condition.max()).nonzero().flatten()
        conditions = "-".join([label_handler.reverse_transform(int(condition)) for condition in conditions])
        filename = f"{(i*sample_batch_size)+filename}_{conditions}.png"
        plt.imsave(os.path.join(output_dir, filename), image)
