import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from IPython.display import HTML
from matplotlib import animation, colors
from PIL import Image
import pytorch_msssim
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
fusion_learning_rate = 1e-4
epochs = 25
batch_size = 64
num_dataloader_workers = 2

experiment_name = f"test_model_v1"

num_layers = 4
max_filters = 512
image_size = 64
latent_dim = 256
use_noise_images = True
small_conv = True  # To use the 1x1 convolution layer
use_sum = False  # Use a sum instead of a mean for our loss function
fusion_use_sum = False  # Whether to use sum or mean for the fusion losses
use_ssim_loss = False
mse_weight = 1
ssim_weight = 1
reconstruction_weight = 1
kl_d_weight = 1  # equivalent to beta in a Beta-VAE
fusion_reconstruction_weight = 1
fusion_kl_d_weight = 1  # equivalent to beta in a Beta-VAE
num_mse = 0  # Each increment halves the image sizes and takes the MSE
num_fusion_mse = 0  # Each increment halves the image sizes and takes the MSE

# Fusion Parameters
# Whether to use fusions in the normal training process
train_reconstruction_on_fusions = False
fusion_mode = "both"  # encoder, decoder, both
freeze_conv_for_fusions = True
fusion_training_epoch_start = 0

# Path to Data
data_prefix = "path_to_data_folder"
fusion_data_prefix = "path_to_fusion_data_folder"

# Output base path
output_prefix = f"path_to_output_folder"

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

# Preprocess & Create Data Loaders
transform = data.image2tensor_resize(image_size)

# Load Standard Data
# TODO Not very efficient since it gets loaded again in the first cases
train = data.load_images_from_folder(train_data_folder, use_noise_images)
val = data.load_images_from_folder(val_data_folder, use_noise_images)
test = data.load_images_from_folder(test_data_folder, use_noise_images)

if train_reconstruction_on_fusions:
    train_data = data.CombinedDataset(
        train_data_folder, train_fusion_data_folder, transform, use_noise_images
    )
    val_data = data.CombinedDataset(
        val_data_folder, val_fusion_data_folder, transform, use_noise_images
    )
    test_data = data.CombinedDataset(
        test_data_folder, test_fusion_data_folder, transform, use_noise_images
    )
else:
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
model = models.ConvolutionalVAE(
    max_filters=max_filters,
    num_layers=num_layers,
    input_image_dimensions=image_size,
    latent_dim=latent_dim,
    small_conv=small_conv,
)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=fusion_learning_rate)

ssim_module = None
if use_ssim_loss:
    ssim_module = pytorch_msssim.SSIM(
        data_range=1.0, win_size=11, win_sigma=1.5, K=(0.01, 0.03)
    )

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

# Get an initial "epoch 0" sample
with torch.no_grad():
    # Get reconstruction of our normal Pokemon
    epoch_sample, _, _ = model(sample.to(device))

    # Get base images
    sample_base, sample_fusee, _ = (
        fusion_sample[:, 0],
        fusion_sample[:, 1],
        fusion_sample[:, 2],
    )
    sample_base = sample_base.to(device)
    sample_fusee = sample_fusee.to(device)
    # Get Embeddings of Base Images
    base_mu, base_log_var = model.get_latent_variables(sample_base)
    fusee_mu, fusee_log_var = model.get_latent_variables(sample_fusee)
    # Midpoint Embedding
    mu = (base_mu * 0.4) + (fusee_mu * 0.6)
    log_var = (base_log_var * 0.4) + (fusee_log_var * 0.6)
    midpoint_embedding = model.reparameterize(mu, log_var)
    # Fusion
    sample_fusion = model.decoder(midpoint_embedding)
    # Group the images together
    fusion_epoch_sample = torch.stack(
        (sample_base, sample_fusee, sample_fusion), dim=1
    ).flatten(end_dim=1)

# Add sample reconstruction to our list
all_samples.append(epoch_sample.detach().cpu())
all_fusion_samples.append(fusion_epoch_sample.detach().cpu())

for epoch in range(epochs):
    train_loss = 0
    train_recon_loss = 0
    train_kl_d = 0
    val_loss = 0
    val_recon_loss = 0
    val_kl_d = 0

    train_fusion_loss = 0
    train_fusion_recon_loss = 0
    train_fusion_kl_d = 0
    val_fusion_loss = 0
    val_fusion_recon_loss = 0
    val_fusion_kl_d = 0

    if learning_rate != fusion_learning_rate:
        optimizer = models.set_learning_rate(optimizer, learning_rate)

    # Training Loop - Standard
    for iteration, batch in enumerate(tqdm(train_dataloader)):
        # Reset gradients back to zero for this iteration
        optimizer.zero_grad()
        # Move batch to device
        _, batch = batch  # Returns key, value for each Pokemon
        batch = batch.to(device)
        # Run our model & get outputs
        reconstructed, mu, log_var = model(batch)
        # Calculate reconstruction loss
        batch_loss, loss_dict = loss.VAE_loss(
            reconstructed,
            batch,
            mu,
            log_var,
            use_sum=use_sum,
            ssim_module=ssim_module,
            mse_weight=mse_weight,
            ssim_weight=ssim_weight,
            reconstruction_weight=reconstruction_weight,
            kl_weight=kl_d_weight,
        )
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
        train_kl_d += loss_dict["KL Divergence"]

    if freeze_conv_for_fusions:
        models.toggle_layer_freezing(freezable_layers, trainable=False)

    if learning_rate != fusion_learning_rate:
        optimizer = models.set_learning_rate(optimizer, fusion_learning_rate)

    if epoch >= fusion_training_epoch_start:
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
                base_mu, base_log_var = model.get_latent_variables(base)
                fusee_mu, fusee_log_var = model.get_latent_variables(fusee)
                # Midpoint Embedding
                mu = (base_mu * 0.4) + (fusee_mu * 0.6)
                log_var = (base_log_var * 0.4) + (fusee_log_var * 0.6)
                midpoint_embedding = model.reparameterize(mu, log_var)

            if fusion_mode == "encoder" or fusion_mode == "both":
                # Run our model & get outputs
                fusion_mu, fusion_log_var = model.get_latent_variables(fusion)
                fusion_embedding = model.reparameterize(fusion_mu, fusion_log_var)
                # Calculate reconstruction loss:
                # Fusion Embedding vs Midpoint Embedding
                # Can't use SSIM here
                batch_loss, loss_dict = loss.VAE_loss(
                    fusion_embedding,
                    midpoint_embedding,
                    fusion_mu,
                    fusion_log_var,
                    use_sum=fusion_use_sum,
                    ssim_module=None,
                    mse_weight=mse_weight,
                    ssim_weight=ssim_weight,
                    reconstruction_weight=fusion_reconstruction_weight,
                    kl_weight=fusion_kl_d_weight,
                )
                # Backprop
                batch_loss.backward()
                # Add the batch's loss to the total loss for the epoch
                train_fusion_loss += batch_loss.item()
                train_fusion_recon_loss += loss_dict["MSE"] + loss_dict["SSIM"]
                train_fusion_kl_d += loss_dict["KL Divergence"]

            if fusion_mode == "decoder" or fusion_mode == "both":
                # Run our model & get outputs
                fusion_output = model.decoder(midpoint_embedding)
                # Calculate reconstruction loss:
                # Midpoint Embedding Output vs Original Fusion
                batch_loss, loss_dict = loss.mse_ssim_loss(
                    fusion_output,
                    fusion,
                    use_sum=fusion_use_sum,
                    ssim_module=ssim_module,
                    mse_weight=mse_weight,
                    ssim_weight=ssim_weight,
                )
                # For multiple MSE
                # For every MSE, we halve the image size
                # And take the MSE between the resulting images
                for i in range(num_fusion_mse):
                    new_size = image_size // pow(2, i + 1)
                    with torch.no_grad():
                        resized_batch = nn.functional.interpolate(
                            fusion, size=new_size, mode="bilinear"
                        )
                    resized_output = nn.functional.interpolate(
                        fusion_output, size=new_size, mode="bilinear"
                    )
                    mse = loss.mse_loss(resized_output, resized_batch, use_sum)
                    batch_loss += mse
                    loss_dict["MSE"] += mse.item()
                # Backprop
                batch_loss.backward()
                # Add the batch's loss to the total loss for the epoch
                train_fusion_loss += batch_loss.item()
                train_fusion_recon_loss += loss_dict["MSE"] + loss_dict["SSIM"]

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
            reconstructed, mu, log_var = model(batch)
            # Calculate reconstruction loss
            batch_loss, loss_dict = loss.VAE_loss(
                reconstructed,
                batch,
                mu,
                log_var,
                use_sum=use_sum,
                ssim_module=ssim_module,
                mse_weight=mse_weight,
                ssim_weight=ssim_weight,
                reconstruction_weight=reconstruction_weight,
                kl_weight=kl_d_weight,
            )
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
            val_kl_d += loss_dict["KL Divergence"]

    # Validation Loop - Fusions
    with torch.no_grad():
        for iteration, batch in enumerate(tqdm(val_fusion_dataloader)):
            # Move batch to device
            _, (base, fusee, fusion) = batch  # (names), (images)
            base = base.to(device)
            fusee = fusee.to(device)
            fusion = fusion.to(device)

            # Get Encoder Output
            base_mu, base_log_var = model.get_latent_variables(base)
            fusee_mu, fusee_log_var = model.get_latent_variables(fusee)
            # Midpoint Embedding
            mu = (base_mu * 0.4) + (fusee_mu * 0.6)
            log_var = (base_log_var * 0.4) + (fusee_log_var * 0.6)
            midpoint_embedding = model.reparameterize(mu, log_var)

            if fusion_mode == "encoder" or fusion_mode == "both":
                # Run our model & get outputs
                fusion_mu, fusion_log_var = model.get_latent_variables(fusion)
                fusion_embedding = model.reparameterize(fusion_mu, fusion_log_var)
                # Calculate reconstruction loss:
                # Fusion Embedding vs Midpoint Embedding
                # Can't use SSIM here
                batch_loss, loss_dict = loss.VAE_loss(
                    fusion_embedding,
                    midpoint_embedding,
                    fusion_mu,
                    fusion_log_var,
                    use_sum=fusion_use_sum,
                    ssim_module=None,
                    mse_weight=mse_weight,
                    ssim_weight=ssim_weight,
                    reconstruction_weight=fusion_reconstruction_weight,
                    kl_weight=fusion_kl_d_weight,
                )
                # Add the batch's loss to the total loss for the epoch
                val_fusion_loss += batch_loss.item()
                val_fusion_recon_loss += loss_dict["MSE"] + loss_dict["SSIM"]
                val_fusion_kl_d += loss_dict["KL Divergence"]

            if fusion_mode == "decoder" or fusion_mode == "both":
                # Run our model & get outputs
                fusion_output = model.decoder(midpoint_embedding)
                # Calculate reconstruction loss:
                # Midpoint Embedding Output vs Original Fusion
                batch_loss, loss_dict = loss.mse_ssim_loss(
                    fusion_output,
                    fusion,
                    use_sum=fusion_use_sum,
                    ssim_module=ssim_module,
                    mse_weight=mse_weight,
                    ssim_weight=ssim_weight,
                )
                # For multiple MSE
                # For every MSE, we halve the image size
                # And take the MSE between the resulting images
                for i in range(num_fusion_mse):
                    new_size = image_size // pow(2, i + 1)
                    resized_batch = nn.functional.interpolate(
                        fusion, size=new_size, mode="bilinear"
                    )
                    resized_output = nn.functional.interpolate(
                        fusion_output, size=new_size, mode="bilinear"
                    )
                    mse = loss.mse_loss(resized_output, resized_batch, use_sum)
                    batch_loss += mse
                    loss_dict["MSE"] += mse.item()
                # Add the batch's loss to the total loss for the epoch
                val_fusion_loss += batch_loss.item()
                val_fusion_recon_loss += loss_dict["MSE"] + loss_dict["SSIM"]

    # Get Sample Outputs for the animation
    with torch.no_grad():
        # Get reconstruction of our normal Pokemon
        epoch_sample, _, _ = model(sample.to(device))

        # Get base images
        sample_base, sample_fusee, _ = (
            fusion_sample[:, 0],
            fusion_sample[:, 1],
            fusion_sample[:, 2],
        )
        sample_base = sample_base.to(device)
        sample_fusee = sample_fusee.to(device)
        # Get Embeddings of Base Images
        base_mu, base_log_var = model.get_latent_variables(sample_base)
        fusee_mu, fusee_log_var = model.get_latent_variables(sample_fusee)
        # Midpoint Embedding
        mu = (base_mu * 0.4) + (fusee_mu * 0.6)
        log_var = (base_log_var * 0.4) + (fusee_log_var * 0.6)
        midpoint_embedding = model.reparameterize(mu, log_var)
        # Fusion
        sample_fusion = model.decoder(midpoint_embedding)
        # Group the images together
        fusion_epoch_sample = torch.stack(
            (sample_base, sample_fusee, sample_fusion), dim=1
        ).flatten(end_dim=1)

    # Add sample reconstruction to our list
    all_samples.append(epoch_sample.detach().cpu())
    all_fusion_samples.append(fusion_epoch_sample.detach().cpu())

    # Compute the average losses for this epoch
    train_loss /= len(train_dataloader)
    train_recon_loss /= len(train_dataloader)
    train_kl_d /= len(train_dataloader)
    all_train_loss.append((train_loss, train_recon_loss, train_kl_d))

    train_fusion_loss /= len(train_fusion_dataloader)
    train_fusion_recon_loss /= len(train_fusion_dataloader)
    train_fusion_kl_d /= len(train_fusion_dataloader)
    all_train_fusion_loss.append(
        (train_fusion_loss, train_fusion_recon_loss, train_fusion_kl_d)
    )

    val_loss /= len(val_dataloader)
    val_recon_loss /= len(val_dataloader)
    val_kl_d /= len(val_dataloader)
    all_val_loss.append((val_loss, val_recon_loss, val_kl_d))

    val_fusion_loss /= len(val_fusion_dataloader)
    val_fusion_recon_loss /= len(val_fusion_dataloader)
    val_fusion_kl_d /= len(val_fusion_dataloader)
    all_val_fusion_loss.append(
        (val_fusion_loss, val_fusion_recon_loss, val_fusion_kl_d)
    )

    # Print Metrics
    print(
        f"\nEpoch: {epoch+1}/{epochs}:\
        \nTrain Loss = {train_loss}\
        \nTrain Reconstruction Loss = {train_recon_loss}\
        \nTrain KL Divergence = {train_kl_d}\
        \nTrain Fusion Loss = {train_fusion_loss}\
        \nTrain Fusion Reconstruction Loss = {train_fusion_recon_loss}\
        \nTrain Fusion KL Divergence = {train_fusion_kl_d}\
        \nVal Loss = {val_loss}\
        \nVal Reconstruction Loss = {val_recon_loss}\
        \nVal KL Divergence = {val_kl_d}\
        \nVal Fusion Loss = {val_fusion_loss}\
        \nVal Fusion Reconstruction Loss = {val_fusion_recon_loss}\
        \nVal Fusion KL Divergence = {val_fusion_kl_d}"
    )

################################################################################
################################## Save & Test #################################
################################################################################
# Generate Loss Graph
graphics.draw_loss(all_train_loss, all_val_loss, loss_output_path, mode="vae")
graphics.draw_loss(
    all_train_fusion_loss, all_val_fusion_loss, fusion_loss_output_path, mode="vae"
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
        reconstructed, _, _ = model(image)

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
        base_mu, base_log_var = model.get_latent_variables(base)
        fusee_mu, fusee_log_var = model.get_latent_variables(fusee)
        # Midpoint Embedding
        mu = (base_mu * 0.4) + (fusee_mu * 0.6)
        log_var = (base_log_var * 0.4) + (fusee_log_var * 0.6)
        midpoint_embedding = model.reparameterize(mu, log_var)
        # Get Output Fusion of combining two Pokemon
        fusion_fused_output = model.decoder(midpoint_embedding)
        # Get output of autoencoder on fusion
        fusion_vae_output, _, _ = model(fusion)

        # Calculate Metrics
        # Print Metrics - Fusion vs Input Fusion
        mse = nn.functional.mse_loss(fusion_fused_output, fusion)
        ssim_score = pytorch_msssim.ssim(
            fusion_fused_output,
            fusion,
            data_range=1.0,
            win_size=11,
            win_sigma=1.5,
            K=(0.01, 0.03),
        )
        # Print Metrics - Fusion vs Autoencoded Fusion
        mse_autoencoded = nn.functional.mse_loss(fusion_fused_output, fusion_vae_output)
        ssim_score_autoencoded = pytorch_msssim.ssim(
            fusion_fused_output,
            fusion_vae_output,
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
    reconstructed, _, _ = model(test_sample.to(device))

    # Get reconstruction of our sample
    sample_base, sample_fusee, _ = (
        fusion_test_sample[:, 0],
        fusion_test_sample[:, 1],
        fusion_test_sample[:, 2],
    )
    sample_base = sample_base.to(device)
    sample_fusee = sample_fusee.to(device)
    # Get Embeddings of Base Images
    base_mu, base_log_var = model.get_latent_variables(sample_base)
    fusee_mu, fusee_log_var = model.get_latent_variables(sample_fusee)
    # Midpoint Embedding
    mu = (base_mu * 0.4) + (fusee_mu * 0.6)
    log_var = (base_log_var * 0.4) + (fusee_log_var * 0.6)
    midpoint_embedding = model.reparameterize(mu, log_var)
    # Fusion
    sample_fusion = model.decoder(midpoint_embedding)
    # Group the images together
    sample_reconstruction = torch.stack(
        (sample_base, sample_fusee, sample_fusion), dim=1
    ).flatten(end_dim=1)

    # Move to CPU
    sample_reconstruction = sample_reconstruction.detach().cpu()
    reconstructed = reconstructed.detach().cpu()

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