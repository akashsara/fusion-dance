import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pytorch_msssim
from tqdm import tqdm
from PIL import Image

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

image_size = 64
use_noise_images = True

experiment_name = f"cnn_multirnn_v1"

vq_vae_experiment_name = f"vq_vae_v5.1"
vq_vae_num_layers = 4
vq_vae_max_filters = 512
vq_vae_use_max_filters = True
vq_vae_num_embeddings = 256
vq_vae_embedding_dim = 32
vq_vae_commitment_cost = 0.25
vq_vae_small_conv = True  # To use the 1x1 convolution layer
vq_vae_encoded_image_size = image_size // np.power(2, vq_vae_num_layers)

prior_num_classes = vq_vae_num_embeddings
prior_input_image_size = image_size
prior_input_channels = 6
prior_cnn_output_channels = 512
prior_cnn_blocks = 4
prior_rnn_hidden_size = 512
prior_rnn_bidirectional = False
prior_rnn_type = "lstm"
prior_num_rnns = 4
sequence_length_per_rnn = (vq_vae_encoded_image_size ** 2) // prior_num_rnns
use_image_as_rnn_input = False

normal_weight = 1
background_weight = 1

data_prefix = "data\\pokemon\\final\\standard"
fusion_data_prefix = "data\\pokemon\\final\\fusions"
output_prefix = f"data\\{experiment_name}"
vq_vae_model_prefix = f"outputs\\{vq_vae_experiment_name}"

vq_vae_model_path = os.path.join(vq_vae_model_prefix, "model.pt")

train_data_folder = os.path.join(data_prefix, "train")
val_data_folder = os.path.join(data_prefix, "val")
test_data_folder = os.path.join(data_prefix, "test")

fusion_train_data_folder = os.path.join(fusion_data_prefix, "train")
fusion_val_data_folder = os.path.join(fusion_data_prefix, "val")
fusion_test_data_folder = os.path.join(fusion_data_prefix, "test")

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
vq_vae.to(device)

# Create Model
model = models.CNN_MultiRNN(
    num_classes=prior_num_classes,
    num_rnns=prior_num_rnns,
    input_image_size=prior_input_image_size,
    input_channels=prior_input_channels,
    cnn_output_channels=prior_cnn_output_channels,
    cnn_blocks=prior_cnn_blocks,
    rnn_hidden_size=prior_rnn_hidden_size,
    rnn_bidirectional=prior_rnn_bidirectional,
    rnn_type=prior_rnn_type,
)
model.to(device)
print(model)
# Compute Class Weights
white = Image.new(mode="RGB", size=(image_size, image_size), color="white")
black = Image.new(mode="RGB", size=(image_size, image_size), color="black")
class_weights = torch.full(
    (prior_num_classes,), fill_value=normal_weight, device=device, dtype=torch.float32
)
with torch.no_grad():
    _, _, _, encodings = vq_vae(transform(white).unsqueeze(0).to(device))
    white = encodings.flatten()[0].detach()
    class_weights[white] = background_weight
    _, _, _, encodings = vq_vae(transform(black).unsqueeze(0).to(device))
    black = encodings.flatten()[0].detach()
    class_weights[black] = background_weight

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
################################################################################
################################### Training ###################################
################################################################################

# Train
sequence_length = vq_vae_encoded_image_size ** 2
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
        batch_loss = 0

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
            y = encodings.reshape(
                current_batch_size,
                vq_vae_encoded_image_size,
                vq_vae_encoded_image_size,
            )

        # Get Encoder Outputs
        decoder_input_orig = model(torch.cat([base, fusee], dim=1))

        # Init Hidden State
        decoder_hidden = model.init_hidden_state(current_batch_size, device)

        # Run Through RNN
        for j, rnn in enumerate(model.decoder_rnns):
            if j == 0 or use_image_as_rnn_input:
                decoder_input = decoder_input_orig.detach()
            for i in range(sequence_length_per_rnn):
                # Get Output For Timestep
                decoder_output, decoder_hidden = rnn(decoder_input, decoder_hidden)
                # Prepare Next Input
                decoder_input = decoder_output.detach()
                # Calculate Current Image Indices
                current_num = ((j * sequence_length_per_rnn) + i)
                j =  current_num // vq_vae_encoded_image_size
                i = current_num % vq_vae_encoded_image_size
                # Calculate Loss
                batch_loss += criterion(decoder_output.squeeze(1), y[:, j, i])

        # Normalize Batch Loss by Sequence Length
        batch_loss /= sequence_length

        # Backprop
        batch_loss.backward()

        # Clip Gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        # Update our optimizer parameters
        optimizer.step()

        # Add the batch's loss to the total loss for the epoch
        train_loss += batch_loss.item()

    # Validation Loop
    model.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(tqdm(val_dataloader)):
            batch_loss = 0
            # Move batch to device
            _, (base, fusee, fusion) = batch  # (names), (images)
            base = base.to(device)
            fusee = fusee.to(device)
            fusion = fusion.to(device)
            current_batch_size = base.shape[0]

            # Get Encodings from vq_vae
            _, _, _, encodings = vq_vae(fusion)
            # Reshape encodings
            y = encodings.reshape(
                current_batch_size,
                vq_vae_encoded_image_size,
                vq_vae_encoded_image_size,
            )

            # Get Encoder Outputs
            decoder_input_orig = model(torch.cat([base, fusee], dim=1))

            # Init Hidden State
            decoder_hidden = model.init_hidden_state(current_batch_size, device)

            # Run Through RNN
            for j, rnn in enumerate(model.decoder_rnns):
                if j == 0 or use_image_as_rnn_input:
                    decoder_input = decoder_input_orig.detach()
                for i in range(sequence_length_per_rnn):
                    # Get Output For Timestep
                    decoder_output, decoder_hidden = rnn(decoder_input, decoder_hidden)
                    # Prepare Next Input
                    decoder_input = decoder_output.detach()
                    # Calculate Current Image Indices
                    current_num = ((j * sequence_length_per_rnn) + i)
                    j =  current_num // vq_vae_encoded_image_size
                    i = current_num % vq_vae_encoded_image_size
                    # Calculate Loss
                    batch_loss += criterion(decoder_output.squeeze(1), y[:, j, i])

            # Normalize Batch Loss by Sequence Length
            batch_loss /= sequence_length

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

# Eval Mode
model.eval()

# Set up Evaluation Metrics
mse_loss = torch.nn.MSELoss(reduction="none")
ssim_module = pytorch_msssim.SSIM(
    data_range=1.0, win_size=11, win_sigma=1.5, K=(0.01, 0.03), size_average=False
)
all_ssim = []
all_mse = []

# Evaluate on Test Images
with torch.no_grad():
    for iteration, batch in enumerate(tqdm(test_dataloader)):
        # Move batch to device
        (_, _, fusion_filenames), (base, fusee, fusion) = batch  # (names), (images)
        base = base.to(device)
        fusee = fusee.to(device)
        fusion = fusion.to(device)
        current_batch_size = base.shape[0]

        # Get Encodings from vq_vae
        _, _, _, encodings = vq_vae(fusion)
        # Reshape encodings
        y = encodings.reshape(
            current_batch_size,
            vq_vae_encoded_image_size,
            vq_vae_encoded_image_size,
        )
        y_hat = torch.zeros_like(y)

        # Get Encoder Outputs
        decoder_input_orig = model(torch.cat([base, fusee], dim=1))

        # Init Hidden State
        decoder_hidden = model.init_hidden_state(current_batch_size, device)

        # Run Through RNN
        for j, rnn in enumerate(model.decoder_rnns):
            if j == 0 or use_image_as_rnn_input:
                decoder_input = decoder_input_orig.detach()
            for i in range(sequence_length_per_rnn):
                # Get Output For Timestep
                decoder_output, decoder_hidden = rnn(decoder_input, decoder_hidden)
                # Prepare Next Input
                decoder_input = decoder_output.detach()
                # Calculate Current Image Indices
                current_num = ((j * sequence_length_per_rnn) + i)
                j =  current_num // vq_vae_encoded_image_size
                i = current_num % vq_vae_encoded_image_size
                # Store Prediction
                y_hat[:, j, i] = decoder_output.squeeze(1).argmax(dim=1).detach().cpu()

        # Make y_hat an image
        target_shape = (
            current_batch_size,
            vq_vae_encoded_image_size,
            vq_vae_encoded_image_size,
            vq_vae_embedding_dim,
        )
        y_hat = y_hat.view(-1, 1)
        y_hat = vq_vae.quantize_and_decode(y_hat, target_shape, device)

        # Create mask
        mask = (base == fusee).flatten(start_dim=1).all(dim=1)

        # Calculate MSE
        overall_mse = (
            mse_loss(y_hat, fusion).flatten(start_dim=1).mean(dim=1).detach().cpu()
        )
        base_mse = torch.masked_select(overall_mse, mask)
        fusion_mse = torch.masked_select(overall_mse, torch.logical_not(mask))
        all_mse.append((overall_mse, base_mse, fusion_mse))

        # Calculate SSIM
        overall_ssim = ssim_module(y_hat, fusion).detach().cpu()
        base_ssim = torch.masked_select(overall_ssim, mask)
        fusion_ssim = torch.masked_select(overall_ssim, torch.logical_not(mask))
        all_ssim.append((overall_ssim, base_ssim, fusion_ssim))

        # Save
        y_hat = y_hat.permute(0, 2, 3, 1).detach().cpu().numpy()
        for image, filename in zip(y_hat, fusion_filenames):
            plt.imsave(os.path.join(output_dir, filename), image)

for i, mse_type in enumerate(["Overall", "Base", "Fusion"]):
    test_mse = torch.cat([x[i] for x in all_mse]).mean()
    print(f"Test {mse_type} MSE = {test_mse}")

for i, ssim_type in enumerate(["Overall", "Base", "Fusion"]):
    test_ssim = torch.cat([x[i] for x in all_ssim]).mean()
    print(f"Test {ssim_type} SSIM = {test_ssim}")