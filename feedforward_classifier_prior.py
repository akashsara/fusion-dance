import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import utils.data as data
import utils.graphics as graphics
import utils.loss as loss
from models import classifier, vqvae, vae
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

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

experiment_name = f"cnn_classifier_v1"
model_type = "vqvae"  # vqvae/vae

if model_type == "vqvae":
    # VQ-VAE Config
    mode = "discrete"
    embedding_experiment_name = f"vq_vae_v5.10"
    vq_vae_num_layers = 0
    vq_vae_max_filters = 512
    vq_vae_use_max_filters = True
    vq_vae_num_embeddings = 256
    vq_vae_embedding_dim = 32
    vq_vae_commitment_cost = 0.25
    vq_vae_small_conv = True  # To use the 1x1 convolution layer
else:
    embedding_experiment_name = f"convolutional_vae_v16.5"
    vae_max_filters = 512
    vae_num_layers = 4
    vae_latent_dim = 256
    vae_small_conv = True

# Classifier Config
num_layers = 2
if model_type == "vqvae":
    input_dimension = (image_size // (2 ** vq_vae_num_layers)) ** 2
else:
    input_dimension = vae_latent_dim
start_filters = input_dimension * 2

# Data Config
label_name = "type1"  # For Pokemon: type1 or type2
data_prefix = "data\\pokemon\\final\\standard"
labels_prefix = "data\\pokemon\\classification"
output_prefix = f"data\\{experiment_name}"

embedding_model_prefix = f"outputs\{embedding_experiment_name}"
embedding_model_path = os.path.join(embedding_model_prefix, "model.pt")

train_labels_file = os.path.join(labels_prefix, "train.joblib")
val_labels_file = os.path.join(labels_prefix, "val.joblib")
test_labels_file = os.path.join(labels_prefix, "test.joblib")

train_data_folder = os.path.join(data_prefix, "train")
val_data_folder = os.path.join(data_prefix, "val")
test_data_folder = os.path.join(data_prefix, "test")

model_output_path = os.path.join(output_prefix, "model.pt")
################################################################################
##################################### Setup ####################################
################################################################################

# Setup Device
gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")
print(gpu, device)

# Create Output Paths
if not os.path.exists(output_prefix):
    os.makedirs(output_prefix)

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

    train_data = data.CustomDatasetWithLabels(
        train, train_labels_file, label_name, transform
    )
    val_data = data.CustomDatasetWithLabels(val, val_labels_file, label_name, transform)
    test_data = data.CustomDatasetWithLabels(
        test, test_labels_file, label_name, transform
    )
else:
    train_data = data.CustomDatasetNoMemoryWithLabels(
        train_data_folder, train_labels_file, label_name, use_noise_images, transform
    )
    val_data = data.CustomDatasetNoMemoryWithLabels(
        val_data_folder, val_labels_file, label_name, use_noise_images, transform
    )
    test_data = data.CustomDatasetNoMemoryWithLabels(
        test_data_folder, test_labels_file, label_name, use_noise_images, transform
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

if model_type == "vqvae":
    # Create & Load VQVAE Model
    embedding_model = vqvae.VQVAE(
        num_layers=vq_vae_num_layers,
        input_image_dimensions=image_size,
        small_conv=vq_vae_small_conv,
        embedding_dim=vq_vae_embedding_dim,
        num_embeddings=vq_vae_num_embeddings,
        commitment_cost=vq_vae_commitment_cost,
        use_max_filters=vq_vae_use_max_filters,
        max_filters=vq_vae_max_filters,
    )
else:
    embedding_model = vae.ConvolutionalVAE(
        max_filters=vae_max_filters,
        num_layers=vae_num_layers,
        input_image_dimensions=image_size,
        latent_dim=vae_latent_dim,
        small_conv=vae_small_conv,
    )
embedding_model.load_state_dict(torch.load(embedding_model_path, map_location=device))
embedding_model.eval()
embedding_model.to(device)

# Setup Label Encoder
all_classes = train_data.get_classes()
num_classes = len(all_classes)
label_encoder = LabelEncoder().fit(all_classes)

# Create Model
model = classifier.ANNMultiClassClassifier(
    num_layers=num_layers,
    input_dimension=input_dimension,
    start_filters=start_filters,
    num_output_classes=num_classes,
)
model.to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
        _, batch, labels = batch  # (names), (images), (labels)
        batch = batch.to(device)
        labels = torch.as_tensor(label_encoder.transform(labels)).long()
        labels = labels.to(device)
        current_batch_size = batch.shape[0]

        with torch.no_grad():
            if model_type == "vqvae":
                # Get Encodings from vq_vae
                _, _, _, encodings = embedding_model(batch)
                x = encodings.reshape(current_batch_size, -1)
                x = x / vq_vae_num_embeddings
            else:
                mu, log_var = embedding_model.get_latent_variables(batch)
                x = embedding_model.reparameterize(mu, log_var)

        # Run our model & get outputs
        y_hat = model.forward(x)
        batch_loss = criterion(y_hat, labels)

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
            _, batch, labels = batch  # (names), (images), (labels)
            batch = batch.to(device)
            labels = torch.as_tensor(label_encoder.transform(labels)).long()
            labels = labels.to(device)
            current_batch_size = batch.shape[0]

            with torch.no_grad():
                if model_type == "vqvae":
                    # Get Encodings from vq_vae
                    _, _, _, encodings = embedding_model(batch)
                    x = encodings.reshape(current_batch_size, -1)
                    x = x / vq_vae_num_embeddings
                else:
                    mu, log_var = embedding_model.get_latent_variables(batch)
                    x = embedding_model.reparameterize(mu, log_var)

            # Run our model & get outputs
            y_hat = model.forward(x)
            batch_loss = criterion(y_hat, labels)

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
graphics.draw_loss(all_train_loss, all_val_loss, output_prefix, mode="autoencoder")

# Save Model
torch.save(
    {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": all_train_loss,
        "val_loss": all_val_loss,
        "label_encoder": label_encoder,
    },
    model_output_path,
)

# Evaluation Time
model.eval()

# Compute loss on test data
test_loss = 0
all_y = []
all_y_hat = []
with torch.no_grad():
    for iteration, batch in enumerate(tqdm(test_dataloader)):
        # Move batch to device
        _, batch, labels = batch  # (names), (images), (labels)
        batch = batch.to(device)
        labels = torch.as_tensor(label_encoder.transform(labels)).long()
        labels = labels.to(device)
        current_batch_size = batch.shape[0]

        with torch.no_grad():
            if model_type == "vqvae":
                # Get Encodings from vq_vae
                _, _, _, encodings = embedding_model(batch)
                x = encodings.reshape(current_batch_size, -1)
                x = x / vq_vae_num_embeddings
            else:
                mu, log_var = embedding_model.get_latent_variables(batch)
                x = embedding_model.reparameterize(mu, log_var)

        # Run our model & get outputs
        y_hat = model.forward(x)
        batch_loss = criterion(y_hat, labels)

        # Store labels and predictions for metric computation
        all_y.extend(labels.detach().cpu())
        all_y_hat.extend(y_hat.argmax(dim=1).detach().cpu())

        # Add the batch's loss to the total loss for the epoch
        test_loss += batch_loss.item()

test_loss = test_loss / len(test_dataloader)
print(f"Test Loss: {test_loss}")

all_y = label_encoder.inverse_transform(all_y)
all_y_hat = label_encoder.inverse_transform(all_y_hat)
print(classification_report(all_y, all_y_hat, digits=4))
