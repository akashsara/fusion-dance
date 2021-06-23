import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
epochs = 5
batch_size = 64
num_dataloader_workers = 0

experiment_name = f"vae_gan_v1"

# VAE Params
vae_num_layers = 4
vae_max_filters = 512
vae_small_conv = True  # To use the 1x1 convolution layer
vae_latent_dim = 256
vae_learning_rate = 1e-4
# GAN (Discriminator) Params
gan_num_layers = 4
gan_max_filters = 512
gan_learning_rate = 1e-4
# Common Params
image_size = 64
gamma = 1

use_noise_images = True
use_sum = False  # Use a sum instead of a mean for our loss function

data_prefix = "data\\final\\standard"
train_data_folder = os.path.join(data_prefix, "train")
val_data_folder = os.path.join(data_prefix, "val")
test_data_folder = os.path.join(data_prefix, "test")

output_prefix = f"data\\{experiment_name}"

output_dir = os.path.join(output_prefix, "generated")
loss_output_path = os.path.join(output_prefix, "loss.jpg")
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
encoder = models.VAEGANEncoder(
    max_filters=vae_max_filters,
    num_layers=vae_num_layers,
    input_image_dimensions=image_size,
    latent_dim=vae_latent_dim,
    small_conv=vae_small_conv,
)
decoder_hidden_dim, decoder_image_size = encoder.get_flattened_size(image_size)
decoder = models.VAEGANDecoder(
    hidden_dim=decoder_hidden_dim,
    image_size=decoder_image_size,
    max_filters=vae_max_filters,
    num_layers=vae_num_layers,
    latent_dim=vae_latent_dim,
    small_conv=vae_small_conv,
)
discriminator = models.VAEGANDiscriminator(
    max_filters=gan_max_filters,
    num_layers=gan_num_layers,
    input_image_dimensions=image_size,
)
encoder.to(device)
decoder.to(device)
discriminator.to(device)

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=vae_learning_rate)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=vae_learning_rate)
discriminator_optimizer = torch.optim.Adam(
    discriminator.parameters(), lr=gan_learning_rate
)

################################################################################
################################### Training ###################################
################################################################################

# Train
all_samples = []
all_train_loss = []
all_val_loss = []

# Get an initial "epoch 0" sample
with torch.no_grad():
    encoder.eval()
    decoder.eval()
    discriminator.eval()
    latent_representation, _, _ = encoder(sample.to(device))
    epoch_sample = decoder(latent_representation)

# Add sample reconstruction to our list
all_samples.append(epoch_sample.detach().cpu())

for epoch in range(epochs):
    train_loss = 0
    train_encoder_loss = 0
    train_decoder_loss = 0
    train_disc_loss = 0
    val_loss = 0
    val_encoder_loss = 0
    val_decoder_loss = 0
    val_disc_loss = 0

    # Training Loop
    encoder.train()
    decoder.train()
    discriminator.train()
    for iteration, batch in enumerate(tqdm(train_dataloader)):
        # Move batch to device
        _, batch = batch  # Returns key, value for each Pokemon
        batch = batch.to(device)

        # Create Labels
        current_batch_size = batch.shape[0]
        y_real = torch.ones(current_batch_size, 1)
        y_fake = torch.zeros(current_batch_size, 1)

        # Encoded Image
        latent_representation, mu, log_var = encoder(batch)

        # Decoded Fake Image (Reconstructed)
        reconstructed_image = decoder(latent_representation)

        # Decoded Fake Image (Noise)
        noise_representation = torch.randn(current_batch_size, vae_latent_dim)
        reconstructed_noise = decoder(noise_representation)

        # Run Discriminator for Real, Fake (Reconstructed), Fake (Noise) Images
        real_output, real_lth_output = discriminator(batch)
        recon_output, recon_lth_output = discriminator(reconstructed_image)
        noise_output, noise_lth_output = discriminator(reconstructed_noise)

        # Calculate Loss
        disc_real_loss = F.binary_cross_entropy(real_output, y_real)
        disc_recon_loss = F.binary_cross_entropy(recon_output, y_fake)
        disc_noise_loss = F.binary_cross_entropy(noise_output, y_fake)
        L_gan = disc_real_loss + disc_recon_loss + disc_noise_loss
        L_prior = loss.kl_divergence(mu, log_var, use_sum)
        L_reconstruction = loss.mse_loss(recon_lth_output, real_lth_output, use_sum)
        discriminator_loss = L_gan
        decoder_loss = gamma * L_reconstruction - L_gan
        encoder_loss = L_prior + L_reconstruction

        # Zero Gradients
        discriminator_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        encoder_optimizer.zero_grad()

        # Backpropagate
        discriminator_loss.backward(retain_graph=True)
        decoder_loss.backward(retain_graph=True)
        encoder_loss.backward()

        # Update Parameters
        discriminator_optimizer.step()
        decoder_optimizer.step()
        encoder_optimizer.step()

        # Add the batch's loss to the total loss for the epoch
        train_encoder_loss += encoder_loss.item()
        train_decoder_loss += decoder_loss.item()
        train_disc_loss += discriminator_loss.item()
    # Get total loss for the epoch
    train_loss = train_encoder_loss + train_decoder_loss + train_disc_loss

    # Validation Loop
    encoder.eval()
    decoder.eval()
    discriminator.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(tqdm(val_dataloader)):
            # Move batch to device
            _, batch = batch  # Returns key, value for each Pokemon
            batch = batch.to(device)

            # Create Labels
            current_batch_size = batch.shape[0]
            y_real = torch.ones(current_batch_size, 1)
            y_fake = torch.zeros(current_batch_size, 1)

            # Encoded Image
            latent_representation, mu, log_var = encoder(batch)

            # Decoded Fake Image (Reconstructed)
            reconstructed_image = decoder(latent_representation)

            # Decoded Fake Image (Noise)
            noise_representation = torch.randn(current_batch_size, vae_latent_dim)
            reconstructed_noise = decoder(noise_representation)

            # Run Discriminator for Real, Fake (Reconstructed), Fake (Noise) Images
            real_output, real_lth_output = discriminator(batch)
            recon_output, recon_lth_output = discriminator(reconstructed_image)
            noise_output, noise_lth_output = discriminator(reconstructed_noise)

            # Calculate Loss
            disc_real_loss = F.binary_cross_entropy(real_output, y_real)
            disc_recon_loss = F.binary_cross_entropy(recon_output, y_fake)
            disc_noise_loss = F.binary_cross_entropy(noise_output, y_fake)
            L_gan = disc_real_loss + disc_recon_loss + disc_noise_loss
            L_prior = loss.kl_divergence(mu, log_var, use_sum)
            L_reconstruction = loss.mse_loss(recon_lth_output, real_lth_output, use_sum)
            discriminator_loss = L_gan
            decoder_loss = gamma * L_reconstruction - L_gan
            encoder_loss = L_prior + L_reconstruction

            # Add the batch's loss to the total loss for the epoch
            val_encoder_loss += encoder_loss.item()
            val_decoder_loss += decoder_loss.item()
            val_disc_loss += discriminator_loss.item()

        # Get total loss for the epoch
        val_loss = val_encoder_loss + val_decoder_loss + val_disc_loss
        # Get reconstruction of our sample
        latent_representation, _, _ = encoder(sample.to(device))
        epoch_sample = decoder(latent_representation)

    # Add sample reconstruction to our list
    all_samples.append(epoch_sample.detach().cpu())

    # Compute the average losses for this epoch
    train_loss = train_loss / len(train_dataloader)
    train_encoder_loss = train_encoder_loss / len(train_dataloader)
    train_decoder_loss = train_decoder_loss / len(train_dataloader)
    train_disc_loss = train_disc_loss / len(train_dataloader)
    all_train_loss.append(
        (train_loss, train_encoder_loss, train_decoder_loss, train_disc_loss)
    )

    val_loss = val_loss / len(val_dataloader)
    val_encoder_loss = val_encoder_loss / len(val_dataloader)
    val_decoder_loss = val_decoder_loss / len(val_dataloader)
    val_disc_loss = val_disc_loss / len(val_dataloader)
    all_val_loss.append((val_loss, val_encoder_loss, val_decoder_loss, val_disc_loss))

    # Print Metrics
    print(
        f"\nEpoch: {epoch+1}/{epochs}:\
        \nTrain Loss = {train_loss}\
        \nTrain Encoder Loss = {train_encoder_loss}\
        \nTrain Decoder Loss = {train_decoder_loss}\
        \nTrain Discriminator Loss = {train_disc_loss}\
        \nVal Loss = {val_loss}\
        \nVal Encoder Loss = {val_encoder_loss}\
        \nVal Decoder Loss = {val_decoder_loss}\
        \nVal Discriminator Loss = {val_disc_loss}"
    )

################################################################################
################################## Save & Test #################################
################################################################################
# Generate Loss Graph
graphics.draw_loss(all_train_loss, all_val_loss, loss_output_path, mode="vaegan")

# Save Model
torch.save(
    {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "discriminator": discriminator.state_dict(),
    },
    model_output_path,
)

# Plot Animation Sample
fig, axis = graphics.make_grid(("Sample", sample), 4, 4)
plt.savefig(animation_sample_image_name)

# Create & Save Animation
anim = graphics.make_animation(graphics.make_grid, all_samples)
anim.save(animation_output_path)

encoder.eval()
decoder.eval()
discriminator.eval()

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
        latent_representation, _, _ = encoder(image)
        reconstructed = decoder(latent_representation)

        # Calculate Metrics
        mse = F.mse_loss(reconstructed, image)
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
    latent_representation, _, _ = encoder(test_sample.to(device))
    reconstructed = decoder(latent_representation).detach().cpu()

# Plot A Set of Reconstructed Test Images
fig, axis = graphics.make_grid(("Test Sample", reconstructed), 4, 4)
plt.savefig(test_sample_output_name)