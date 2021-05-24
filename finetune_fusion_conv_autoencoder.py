import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from IPython.display import HTML
from matplotlib import animation, colors
from PIL import Image
from pytorch_msssim import ssim
from torchvision import transforms
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
epochs = 2
batch_size = 64
num_dataloader_workers = 0

experiment_name = f"test596"

num_layers = 4
max_filters = 512
image_size = 64
latent_dim = 256
use_noise_images = True
small_conv = True  # To use the 1x1 convolution layer

# Fusion Parameters
fusion_mode = "encoder"  # encoder, decoder, both
pretrained_model_path = "outputs\\convolutional_autoencoder_v9\\model.pt"
freeze_conv_for_fusions = True

# Path to Data
data_prefix = "data\\final\\standard"
fusion_data_prefix = "data\\final\\fusions"

# Output base path
output_prefix = f"data\\{experiment_name}"

train_data_folder = os.path.join(data_prefix, "train")
val_data_folder = os.path.join(data_prefix, "val")
test_data_folder = os.path.join(data_prefix, "test")

train_fusion_data_folder = os.path.join(fusion_data_prefix, "train")
val_fusion_data_folder = os.path.join(fusion_data_prefix, "val")
test_fusion_data_folder = os.path.join(fusion_data_prefix, "test")

output_dir = os.path.join(output_prefix, "generated", "normal")
fusion_output_dir = os.path.join(output_prefix, "generated", "fusions")
loss_output_path = os.path.join(output_prefix, "loss.jpg")
fusion_loss_output_path = os.path.join(output_prefix, "fusion_loss.jpg")
model_output_path = os.path.join(output_prefix, "model.pt")

animation_output_path = os.path.join(output_prefix, "animation.mp4")
fusion_animation_output_path = os.path.join(output_prefix, "fusion_animation.mp4")
animation_sample_image_name = os.path.join(output_prefix, "animation_base.jpg")
fusion_animation_sample_image_name = os.path.join(
    output_prefix, "fusion_animation_base.jpg"
)

test_sample_input_name = os.path.join(output_prefix, "test_sample_input.jpg")
test_sample_output_name = os.path.join(output_prefix, "test_sample_output.jpg")

fusion_test_sample_input_name = os.path.join(
    output_prefix, "fusion_test_sample_input.jpg"
)
fusion_test_sample_output_name = os.path.join(
    output_prefix, "fusion_test_sample_output.jpg"
)
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
if not os.path.exists(fusion_output_dir):
    os.makedirs(fusion_output_dir)

################################################################################
################################## Data Setup ##################################
################################################################################

# Load Standard Data
train = data.load_images_from_folder(train_data_folder, use_noise_images)
val = data.load_images_from_folder(val_data_folder, use_noise_images)
test = data.load_images_from_folder(test_data_folder, use_noise_images)

# Preprocess & Create Data Loaders
transform = data.image2tensor_resize(image_size)

train_data = data.CustomDataset(train, transform)
val_data = data.CustomDataset(val, transform)
test_data = data.CustomDataset(test, transform)

train_fusions = data.FusionDataset(
    train_fusion_data_folder, train, val, test, transform
)
val_fusions = data.FusionDataset(val_fusion_data_folder, train, val, test, transform)
test_fusions = data.FusionDataset(test_fusion_data_folder, train, val, test, transform)

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

train_fusion_dataloader = torch.utils.data.DataLoader(
    train_fusions,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_dataloader_workers,
    pin_memory=gpu,
)
val_fusion_dataloader = torch.utils.data.DataLoader(
    val_fusions,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_dataloader_workers,
    pin_memory=gpu,
)
test_fusion_dataloader = torch.utils.data.DataLoader(
    test_fusions,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_dataloader_workers,
    pin_memory=gpu,
)

# Creating a sample set that we visualize every epoch to show the model's training
sample = data.get_samples_from_data(val_data, 16, fusion=False)
test_sample = data.get_samples_from_data(test_data, 16, fusion=False)
fusion_sample = data.get_samples_from_data(val_fusions, 4, fusion=True)
fusion_test_sample = data.get_samples_from_data(test_fusions, 4, fusion=True)

################################################################################
##################################### Model ####################################
################################################################################

# Create Model
model = models.ConvolutionalAE(
    max_filters=max_filters,
    num_layers=num_layers,
    input_image_dimensions=image_size,
    latent_dim=latent_dim,
    small_conv=small_conv,
)
model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss(reduction="mean")
################################################################################
################################### Training ###################################
################################################################################
# Freeze Conv Layers
if freeze_conv_for_fusions:
    freezable_layers = models.get_freezable_layers(model)

# Train
all_samples = []
all_fusion_samples = []
all_train_loss = []
all_train_fusion_loss = []
all_val_loss = []
all_val_fusion_loss = []

for epoch in range(epochs):
    val_loss = 0
    train_fusion_loss = 0
    val_fusion_loss = 0

    if freeze_conv_for_fusions:
        models.toggle_layer_freezing(freezable_layers, trainable=False)

    # Training Loop - Fusions
    for iteration, batch in enumerate(tqdm(train_fusion_dataloader)):
        # Reset gradients back to zero for this iteration
        optimizer.zero_grad()

        # Move batch to device
        _, (base, fusee, fusion) = batch  # (names), (images)
        base = base.to(device)
        fusee = fusee.to(device)
        fusion = fusion.to(device)

        with torch.no_grad():
            # Get Encoder Output
            base_embedding = model.encoder(base)
            fusee_embedding = model.encoder(fusee)
            # Midpoint Embedding
            midpoint_embedding = (base_embedding * 0.4) + (fusee_embedding * 0.6)

        if fusion_mode == "encoder" or fusion_mode == "both":
            # Run our model & get outputs
            fusion_embedding = model.encoder(fusion)
            # Calculate reconstruction loss:
            # Fusion Embedding vs Midpoint Embedding
            batch_loss = criterion(fusion_embedding, midpoint_embedding)
            # Backprop
            batch_loss.backward()
            # Add the batch's loss to the total loss for the epoch
            train_fusion_loss += batch_loss.item()

        if fusion_mode == "decoder" or fusion_mode == "both":
            # Run our model & get outputs
            fusion_output = model.decoder(midpoint_embedding)
            # Calculate reconstruction loss:
            # Midpoint Embedding Output vs Original Fusion
            batch_loss = criterion(fusion_output, fusion)
            # Backprop
            batch_loss.backward()
            # Add the batch's loss to the total loss for the epoch
            train_fusion_loss += batch_loss.item()

        # Update our optimizer parameters
        # We call it out here instead of inside the if because
        # the gradients are accumulated in case both conditions are true
        optimizer.step()

    if freeze_conv_for_fusions:
        models.toggle_layer_freezing(freezable_layers, trainable=True)

    # Validation Loop - Standard
    with torch.no_grad():
        for iteration, batch in enumerate(tqdm(val_dataloader)):
            # Move batch to device
            _, batch = batch  # Returns key, value for each Pokemon
            batch = batch.to(device)
            # Run our model & get outputs
            reconstructed = model(batch)
            # Calculate reconstruction loss
            batch_loss = criterion(batch, reconstructed)
            # Add the batch's loss to the total loss for the epoch
            val_loss += batch_loss.item()

    # Validation Loop - Fusions
    with torch.no_grad():
        for iteration, batch in enumerate(tqdm(val_fusion_dataloader)):
            # Move batch to device
            _, (base, fusee, fusion) = batch  # (names), (images)
            base = base.to(device)
            fusee = fusee.to(device)
            fusion = fusion.to(device)

            # Get Encoder Output
            base_embedding = model.encoder(base)
            fusee_embedding = model.encoder(fusee)
            # Midpoint Embedding
            midpoint_embedding = (base_embedding * 0.4) + (fusee_embedding * 0.6)

            if fusion_mode == "encoder" or fusion_mode == "both":
                # Run our model & get outputs
                fusion_embedding = model.encoder(fusion)
                # Calculate reconstruction loss:
                # Fusion Embedding vs Midpoint Embedding
                batch_loss = criterion(fusion_embedding, midpoint_embedding)
                # Add the batch's loss to the total loss for the epoch
                val_fusion_loss += batch_loss.item()

            if fusion_mode == "decoder" or fusion_mode == "both":
                # Run our model & get outputs
                fusion_output = model.decoder(midpoint_embedding)
                # Calculate reconstruction loss:
                # Midpoint Embedding Output vs Original Fusion
                batch_loss = criterion(fusion_output, fusion)
                # Add the batch's loss to the total loss for the epoch
                val_fusion_loss += batch_loss.item()

    # Get Sample Outputs for the animation
    with torch.no_grad():
        # Get reconstruction of our normal Pokemon
        epoch_sample = model(sample.to(device))

        # Get base images
        sample_base, sample_fusee, _ = (
            fusion_sample[:, 0],
            fusion_sample[:, 1],
            fusion_sample[:, 2],
        )
        sample_base = sample_base.to(device)
        sample_fusee = sample_fusee.to(device)
        # Get Embeddings of Base Images
        sample_base_embedding = model.encoder(sample_base)
        sample_fusee_embedding = model.encoder(sample_fusee)
        # Get Midpoint -> Decoder -> Fusion
        sample_midpoint_embedding = (sample_base_embedding * 0.4) + (
            sample_fusee_embedding * 0.6
        )
        sample_fusion = model.decoder(sample_midpoint_embedding)
        # Group the images together
        fusion_epoch_sample = torch.stack(
            (sample_base, sample_fusee, sample_fusion), dim=1
        ).flatten(end_dim=1)

    # Add sample reconstruction to our list
    all_samples.append(epoch_sample.detach().cpu())
    all_fusion_samples.append(fusion_epoch_sample.detach().cpu())

    # Compute the average losses for this epoch
    train_fusion_loss = train_fusion_loss / len(train_fusion_dataloader)
    all_train_fusion_loss.append(train_fusion_loss)

    val_loss = val_loss / len(val_dataloader)
    all_val_loss.append(val_loss)

    val_fusion_loss = val_fusion_loss / len(val_fusion_dataloader)
    all_val_fusion_loss.append(val_fusion_loss)

    # Print Metrics
    print(
        f"\nEpoch: {epoch+1}/{epochs}:\
        \nVal Loss = {val_loss}\
        \nTrain Fusion Loss = {train_fusion_loss}\
        \nVal Fusion Loss = {val_fusion_loss}"
    )

################################################################################
################################## Save & Test #################################
################################################################################
# Generate Loss Graph
graphics.draw_loss(all_train_loss, all_val_loss, loss_output_path, mode="ae")
graphics.draw_loss(
    all_train_fusion_loss, all_val_fusion_loss, fusion_loss_output_path, mode="ae"
)

# Save Model
torch.save(model.state_dict(), model_output_path)

# Plot Animation Sample
fig, axis = graphics.make_grid(("Sample", sample), 4, 4)
plt.savefig(animation_sample_image_name)
fusion_sample = [x for y in fusion_sample for x in y]
fig, axis = graphics.make_grid(("Fusion Sample", fusion_sample), 4, 3)
plt.savefig(fusion_animation_sample_image_name)

# Create & Save Animation
anim = graphics.make_animation(graphics.make_grid, all_samples, width=3)
fusion_anim = graphics.make_animation(graphics.make_grid, all_fusion_samples, width=3)
anim.save(animation_output_path)
fusion_anim.save(fusion_animation_output_path)

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
        reconstructed = model(image)

        # Calculate Metrics
        mse = nn.functional.mse_loss(reconstructed, image)
        ssim_score = ssim(
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

# Testing Loop - Fusions
all_mse = []
all_mse_autoencoded = []
all_ssim = []
all_ssim_autoencoded = []
with torch.no_grad():
    for iteration, batch in enumerate(tqdm(test_fusion_dataloader)):
        # Move batch to device
        filenames, images = batch  # (names), (images)
        base_filenames, fusee_filenames, fusion_filenames = filenames
        base, fusee, fusion = images
        base = base.to(device)
        fusee = fusee.to(device)
        fusion = fusion.to(device)

        # Run Model
        # Get Encoder Output
        base_embedding = model.encoder(base)
        fusee_embedding = model.encoder(fusee)
        # Midpoint Embedding
        midpoint_embedding = (base_embedding * 0.4) + (fusee_embedding * 0.6)
        # Get Output Fusion of combining two Pokemon
        fusion_fused_output = model.decoder(midpoint_embedding)
        # Get output of autoencoder on fusion
        fusion_ae_output = model(fusion)

        # Calculate Metrics
        # Print Metrics - Fusion vs Input Fusion
        mse = nn.functional.mse_loss(fusion_fused_output, fusion)
        ssim_score = ssim(
            fusion_fused_output,
            fusion,
            data_range=1.0,
            win_size=11,
            win_sigma=1.5,
            K=(0.01, 0.03),
        )
        # Print Metrics - Fusion vs Autoencoded Fusion
        mse_autoencoded = nn.functional.mse_loss(fusion_fused_output, fusion_ae_output)
        ssim_score_autoencoded = ssim(
            fusion_fused_output,
            fusion_ae_output,
            data_range=1.0,
            win_size=11,
            win_sigma=1.5,
            K=(0.01, 0.03),
        )

        # Add metrics to tracking list
        all_mse.append(mse.detach().cpu().numpy())
        all_ssim.append(ssim_score.detach().cpu().numpy())
        all_mse_autoencoded.append(mse_autoencoded.detach().cpu().numpy())
        all_ssim_autoencoded.append(ssim_score_autoencoded.detach().cpu().numpy())

        # Save
        fusion_fused_output = (
            fusion_fused_output.permute(0, 2, 3, 1).detach().cpu().numpy()
        )
        for image, filename in zip(fusion_fused_output, fusion_filenames):
            plt.imsave(os.path.join(fusion_output_dir, filename), image)

mse = np.asarray(all_mse).mean()
ssim_score = np.asarray(all_ssim).mean()
mse_autoencoded = np.asarray(all_mse_autoencoded).mean()
ssim_score_autoencoded = np.asarray(all_ssim_autoencoded).mean()
print(f"\nFusion vs Input Fusion:\nMSE = {mse}, SSIM = {ssim_score}")
print(
    f"\nFusion vs Autoencoded Fusion\nMSE = {mse_autoencoded}, SSIM = {ssim_score_autoencoded}"
)

# Pick a couple of sample images for an Input v Output comparison
test_sample = data.get_samples_from_data(test_data, 16)
fusion_test_sample = data.get_samples_from_data(test_fusions, 4, fusion=True)

with torch.no_grad():
    reconstructed = model(test_sample.to(device)).detach().cpu()

    # Get reconstruction of our sample
    sample_base, sample_fusee, _ = (
        fusion_test_sample[:, 0],
        fusion_test_sample[:, 1],
        fusion_test_sample[:, 2],
    )
    # Sample Fusion
    sample_base = sample_base.to(device)
    sample_fusee = sample_fusee.to(device)
    # Get Embeddings of Base Images
    sample_base_embedding = model.encoder(sample_base)
    sample_fusee_embedding = model.encoder(sample_fusee)
    # Get Midpoint -> Decoder -> Fusion
    sample_midpoint_embedding = (sample_base_embedding * 0.4) + (
        sample_fusee_embedding * 0.6
    )
    sample_fusion = model.decoder(sample_midpoint_embedding)
    # Group the images together
    sample_reconstruction = torch.stack(
        (sample_base, sample_fusee, sample_fusion), dim=1
    ).flatten(end_dim=1)
    sample_reconstruction = sample_reconstruction.detach().cpu()

# Plot A Set of Test Images
fig, axis = graphics.make_grid(("Test Sample", test_sample), 4, 4)
plt.savefig(test_sample_input_name)
fusion_test_sample = [x for y in fusion_test_sample for x in y]
fig, axis = graphics.make_grid(("Test Fusion Sample", fusion_test_sample), 4, 3)
plt.savefig(fusion_test_sample_input_name)

# Plot A Set of Reconstructed Test Images
fig, axis = graphics.make_grid(("Test Sample", reconstructed), 4, 4)
plt.savefig(test_sample_output_name)
fig, axis = graphics.make_grid(("Test Fusion Sample", sample_reconstruction), 4, 3)
plt.savefig(fusion_test_sample_output_name)