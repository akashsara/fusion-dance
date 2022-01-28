# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from tqdm import tqdm

import utils.data as data
import utils.graphics as graphics
import utils.loss as loss
from models import gan, utils

seed = 42
np.random.seed(seed)
_ = torch.manual_seed(seed)

################################################################################
#################################### Config ####################################
################################################################################

experiment_name = f"gan_v1"

# Hyperparameters
learning_rate = 0.0002
num_epochs = 25
batch_size = 64
num_dataloader_workers = 0
beta1 = 0.5  # Beta1 hyperparam for Adam optimizers

# GAN Config
num_output_channels = 3
latent_dim = 100
generator_num_filters = 64
discriminator_num_filters = 64

# Data Config
image_size = 64
use_noise_images = True

data_prefix = "data\\Pokemon\\final\\standard"
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


# Preprocess & Create Data Loaders
transform = data.image2tensor_resize(image_size)

dataset = data.EverythingDataset(
    train_data_folder, val_data_folder, test_data_folder, transform, use_noise_images
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_dataloader_workers,
    pin_memory=gpu,
)

# Create batch of latent vectors that we will use to visualize
# the progression of the generator
sample = torch.randn((64, latent_dim, 1, 1), device=device)

################################################################################
##################################### Model ####################################
################################################################################

# Create the model
netG = gan.Generator(
    latent_dim=latent_dim,
    num_filters=generator_num_filters,
    num_output_channels=num_output_channels,
).to(device)
netD = gan.Discriminator(
    num_filters=discriminator_num_filters, num_output_channels=num_output_channels
).to(device)

# Apply the weights_init function to randomly initialize all weights
# to mean=0, stdev=0.02.
netG.apply(utils.weights_init)
netD.apply(utils.weights_init)

# Print the model
print(netG)
print(netD)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))

################################################################################
################################### Training ###################################
################################################################################

# Establish convention for real and fake labels during training
real_label = 1.0
fake_label = 0.0

# Training Loop

# Lists to keep track of progress
training_samples = []
all_generator_loss = []
all_discriminator_loss = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    generator_loss = 0
    discriminator_loss = 0
    # For each batch in the dataloader
    for iteration, batch in enumerate(tqdm(dataloader)):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # Zero Grad
        netD.zero_grad()
        # Format batch
        _, batch = batch
        real_cpu = batch.to(device)
        b_size = real_cpu.size(0)

        ## Train with all-real batch
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Save Losses for plotting later
        generator_loss += errG.item()
        discriminator_loss += errD.item()

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or (
            (epoch == num_epochs - 1) and (iteration == len(dataloader) - 1)
        ):
            with torch.no_grad():
                fake = netG(sample).detach().cpu()
            training_samples.append(vutils.make_grid(fake, padding=2, normalize=True))
        iters += 1

    generator_loss = generator_loss / len(dataloader)
    discriminator_loss = discriminator_loss / len(dataloader)
    all_generator_loss.append(generator_loss)
    all_discriminator_loss.append(discriminator_loss)
    # Output training stats
    print(
        f"[{epoch+1}/{num_epochs}]\tLoss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}\tD(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}"
    )

################################################################################
################################## Save & Test #################################
################################################################################

# Save output graphs
graphics.draw_loss(
    all_generator_loss, all_discriminator_loss, loss_output_path, mode="gan"
)

# Create & Save Animation
anim = graphics.make_gan_animation(training_samples)
anim.save(animation_output_path)

# Save Model
torch.save(
    {
        "epoch": epoch,
        "generator_model_state_dict": netG.state_dict(),
        "generator_optimizer_state_dict": optimizerG.state_dict(),
        "generator_loss": all_generator_loss,
        "discriminator_model_state_dict": netD.state_dict(),
        "discriminator_optimizer_state_dict": optimizerD.state_dict(),
        "discriminator_loss": all_discriminator_loss,
    },
    model_output_path,
)

# Pick a couple of sample images for an Input v Output comparison
sample = data.get_samples_from_data(dataset, 16)

# Plot A Set of Test Images
fig, axis = graphics.make_grid(("Test Sample", sample), 4, 4)
plt.savefig(test_sample_input_name)

# Generate fake images
with torch.no_grad():
    # Generate batch of latent vectors
    noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
    # Generate fake image batch with G
    generated = netG(noise).detach().cpu()

# Plot fake images
fig, axis = graphics.make_grid(("Test Sample", generated), 4, 4)
plt.savefig(test_sample_output_name)
