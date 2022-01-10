import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import utils.data as data
import utils.graphics as graphics
from models import vqvae
from transformers import GPT2LMHeadModel, GPT2Config

seed = 42
np.random.seed(seed)
_ = torch.manual_seed(seed)

################################################################################
#################################### Config ####################################
################################################################################

experiment_name = f"transformer_prior_v1"

# VQ-VAE Config
vq_vae_image_size = 64
vq_vae_experiment_name = f"vq_vae_v5.8"
vq_vae_num_layers = 1
vq_vae_max_filters = 512
vq_vae_use_max_filters = True
vq_vae_num_embeddings = 256
vq_vae_embedding_dim = 32
vq_vae_commitment_cost = 0.25
vq_vae_small_conv = True  # To use the 1x1 convolution layer

# Transformer Prior Config
image_size = vq_vae_image_size // (2 ** vq_vae_num_layers)
max_seq_length = image_size * image_size
vocab_size = vq_vae_num_embeddings
embedding_size = 768  # Default: 768
num_hidden_layers = 12  # Default: 12
num_attention_heads = 12  # Default: 12

# Generation Config
generation_batches = 5
generation_batch_size = 32
generation_temperature = 1

# Hyperparameters
learning_rate = 1e-4
epochs = 10
batch_size = 32

# Data Config
num_dataloader_workers = 0
use_noise_images = False
data_prefix = "data\\pokemon\\final\\standard"
output_prefix = f"data\\{experiment_name}"
vq_vae_model_prefix = f"outputs\\{vq_vae_experiment_name}"
vq_vae_model_path = os.path.join(vq_vae_model_prefix, "model.pt")

train_data_folder = os.path.join(data_prefix, "train")
val_data_folder = os.path.join(data_prefix, "val")
test_data_folder = os.path.join(data_prefix, "test")

reconstructed_dir = os.path.join(output_prefix, "reconstructed")
generated_dir = os.path.join(output_prefix, "generated")
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
if not os.path.exists(reconstructed_dir):
    os.makedirs(reconstructed_dir)
if not os.path.exists(generated_dir):
    os.makedirs(generated_dir)

################################################################################
################################## Data Setup ##################################
################################################################################

# Preprocess & Create Data Loaders
transform = data.image2tensor_resize(vq_vae_image_size)

# Load Data
train_data = data.CustomDatasetNoMemory(train_data_folder, transform, use_noise_images)
val_data = data.CustomDatasetNoMemory(val_data_folder, transform, use_noise_images)
test_data = data.CustomDatasetNoMemory(test_data_folder, transform, use_noise_images)

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
configuration = GPT2Config(
    vocab_size=vocab_size,
    n_positions=max_seq_length,
    n_embd=embedding_size,
    n_layer=num_hidden_layers,
    n_head=num_attention_heads,
)
model = GPT2LMHeadModel(configuration)
model.to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
    encodings = encodings.reshape(-1, max_seq_length)
    epoch_sample = model(encodings)["logits"].argmax(dim=2).view(-1, 1)
    epoch_sample = vq_vae.quantize_and_decode(epoch_sample, target_shape, device)

# Add sample reconstruction to our list
all_samples.append(epoch_sample.detach().cpu())

for epoch in range(epochs):
    train_loss = 0
    val_loss = 0

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
            encodings = encodings.reshape(-1, max_seq_length)

        # Run our model & get outputs
        loss = model(input_ids=encodings, labels=encodings)["loss"]

        # Backprop
        loss.backward()

        # Update our optimizer parameters
        optimizer.step()

        # Add the batch's loss to the total loss for the epoch
        train_loss += loss.item()

    # Validation Loop
    model.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(tqdm(val_dataloader)):
            # Move batch to device
            _, batch = batch  # Returns key, value for each Pokemon
            batch = batch.to(device)

            # Get Encodings from vq_vae
            _, _, _, encodings = vq_vae(batch)
            encodings = encodings.reshape(-1, max_seq_length)

            # Run our model & get outputs
            loss = model(input_ids=encodings, labels=encodings)["loss"]

            # Add the batch's loss to the total loss for the epoch
            val_loss += loss.item()

        # Get reconstruction of our sample
        target_shape = (sample.shape[0], image_size, image_size, vq_vae_embedding_dim)
        _, _, _, encodings = vq_vae(sample.to(device))
        encodings = encodings.reshape(-1, max_seq_length)
        epoch_sample = model(encodings)["logits"].argmax(dim=2).view(-1, 1)
        epoch_sample = vq_vae.quantize_and_decode(epoch_sample, target_shape, device)

    # Add sample reconstruction to our list
    all_samples.append(epoch_sample.detach().cpu())

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
graphics.draw_loss(all_train_loss, all_val_loss, loss_output_path, mode="standard")

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
with torch.no_grad():
    for iteration, batch in enumerate(tqdm(test_dataloader)):
        # Move batch to device
        filenames, batch = batch  # Returns key, value for each Pokemon
        batch = batch.to(device)

        # Get Encodings from vq_vae
        _, _, _, encodings = vq_vae(batch)
        encodings = encodings.reshape(-1, max_seq_length)

        # Run our model & get outputs
        outputs = model(input_ids=encodings, labels=encodings)
        loss = outputs["loss"]
        reconstructed = outputs["logits"]

        # Add the batch's loss to the total loss for the epoch
        test_loss += loss.item()

        # Save
        target_shape = (
            reconstructed.shape[0],
            image_size,
            image_size,
            vq_vae_embedding_dim,
        )

        reconstructed = reconstructed.argmax(dim=2).view(-1, 1)
        reconstructed = vq_vae.quantize_and_decode(reconstructed, target_shape, device)
        reconstructed = reconstructed.permute(0, 2, 3, 1).detach().cpu().numpy()
        for image, filename in zip(reconstructed, filenames):
            plt.imsave(os.path.join(reconstructed_dir, filename), image)

# Print Metrics
test_loss = test_loss / len(test_dataloader)
print(f"\nTest Loss = {test_loss}")

# Pick a couple of sample images for an Input v Output comparison
test_sample = data.get_samples_from_data(test_data, 16)

# Plot A Set of Test Images
fig, axis = graphics.make_grid(("Test Sample", test_sample), 4, 4)
plt.savefig(test_sample_input_name)

target_shape = (test_sample.shape[0], image_size, image_size, vq_vae_embedding_dim)
with torch.no_grad():
    _, _, _, encodings = vq_vae(test_sample.to(device))
    encodings = encodings.reshape(-1, max_seq_length)
    test_sample = model(encodings)["logits"].argmax(dim=2).view(-1, 1)
    test_sample = vq_vae.quantize_and_decode(test_sample, target_shape, device)
    reconstructed = test_sample.detach().cpu()

# Plot A Set of Reconstructed Test Images
fig, axis = graphics.make_grid(("Test Sample", reconstructed), 4, 4)
plt.savefig(test_sample_output_name)

with torch.no_grad():
    # Get most common pixel values to feed into generation script
    _, _, _, encodings = vq_vae(sample.to(device))
    encodings, counts = encodings.unique(return_counts=True)
    bg1, bg2 = encodings[counts.topk(k=2, largest=True).indices].cpu().numpy()

    # Generate New Images
    for i in range(generation_batches):
        bg1_inputs = torch.zeros((generation_batch_size, 1)).int() + bg1
        bg2_inputs = torch.zeros((generation_batch_size, 1)).int() + bg2
        input_ids = torch.cat((bg1_inputs, bg2_inputs))
        generated = model.generate(
            input_ids=input_ids,
            max_length=max_seq_length,
            temperature=generation_temperature,
            do_sample=True,
        )
        target_shape = (
            generated.shape[0],
            image_size,
            image_size,
            vq_vae_embedding_dim,
        )
        generated = vq_vae.quantize_and_decode(
            generated.view(-1, 1), target_shape, device
        )
        generated = generated.permute(0, 2, 3, 1).detach().cpu().numpy()
        for j, image in enumerate(generated):
            filename = f"{(generation_batch_size * i) + j}.png"
            plt.imsave(os.path.join(generated_dir, filename), image)
