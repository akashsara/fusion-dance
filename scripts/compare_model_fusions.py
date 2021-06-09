import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.random import f
from PIL import Image

sys.path.append("./")
import models
from utils import data, graphics

def pick_images(dir, num_images=8, max_tries=10000):
    """
    Picks num_images sprites from the dataset such that no image appears twice.
    """
    unique_ids = []
    selected = []
    all_images = os.listdir(dir)
    tries = -1
    while len(selected) < num_images:
        tries += 1
        if tries > max_tries:
            print("Error picking Images!")
            break
        choice = np.random.choice(all_images)
        id_ = choice.split("_")[0]
        if id_ in unique_ids:
            continue
        # To guarantee we have a fusion
        if int(id_.split("-")[0]) > 251:
            continue
        selected.append(choice)
        unique_ids.append(id_.split("-")[0])
    selected = [list(x) for x in np.array(selected).reshape(-1, 2)]
    unique_ids = [list(x) for x in np.array(unique_ids).reshape(-1, 2)]
    return selected, unique_ids


def load_and_process_image(file_path, transform, image_size):
    image = Image.open(file_path)
    image = image.resize((image_size, image_size), resample=Image.BICUBIC)
    image = image.convert("RGBA")
    background = Image.new("RGBA", (image_size, image_size), (255, 255, 255))
    image = Image.alpha_composite(background, image).convert("RGB")
    return transform(image)


def get_images(dir, image_size, transform, images_to_load):
    """
    Loads the images from the given directory.
    """
    bases = []
    fusees = []
    for (base, fusee) in images_to_load:
        # Make path
        base_path = os.path.join(dir, base)
        fusee_path = os.path.join(dir, fusee)
        # Load Image
        base = load_and_process_image(base_path, transform, image_size)
        fusee = load_and_process_image(fusee_path, transform, image_size)
        bases.append(base)
        fusees.append(fusee)
    return torch.stack(bases), torch.stack(fusees)


def get_fusion_images(dir, image_size, transform, images_to_load):
    """
    Loads the images from the given directory.
    """
    images = []
    for (base, fusee) in images_to_load:
        base = str(int(base))
        fusee = str(int(fusee))
        target = f"{base}.{fusee}.png"
        file_path = os.path.join(dir, base, target)
        image = load_and_process_image(file_path, transform, image_size)
        images.append(image)
    return torch.stack(images)


def load_model(
    model_type,
    max_filters,
    num_layers,
    input_image_dimensions,
    latent_dim,
    small_conv,
    model_path,
    device,
):
    if model_type == "autoencoder":
        model = models.ConvolutionalAE(
            max_filters=max_filters,
            num_layers=num_layers,
            input_image_dimensions=image_size,
            latent_dim=latent_dim,
            small_conv=small_conv,
        )
    else:
        model = models.ConvolutionalVAE(
            max_filters=max_filters,
            num_layers=num_layers,
            input_image_dimensions=image_size,
            latent_dim=latent_dim,
            small_conv=small_conv,
        )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")

base_dir = "data\\original\\original_data"
base_fusion_dir = "data\\backup\\japeal_renamed"
output_dir = "data"
model_prefix = f"outputs/"
num_images = 4  # Number of fusions
batch_size = 64
num_layers = 4
max_filters = 512
image_size = 64
small_conv = True  # To use the 1x1 convolution layer

# model_name: latent_dim
model_config = {
    "convolutional_vae_v6": 256,
    "convolutional_vae_v12": 256,
    "convolutional_vae_v12.1": 256,
    "convolutional_vae_v12.2": 256,
    "convolutional_vae_v12.3": 256,
    "convolutional_vae_v12.4": 256,
    "convolutional_vae_v12.5": 256,
    "convolutional_vae_v12.6": 256,
    "convolutional_vae_v12.7": 256,
    "convolutional_vae_v15": 256,
    "convolutional_vae_v15.1": 256,
    "convolutional_vae_v15.2": 256,
    "convolutional_vae_v15.3": 256,
    "convolutional_vae_v15.4": 256,
    "convolutional_vae_v15.5": 256,
    "convolutional_vae_v15.6": 256,
}

transform = data.image2tensor_resize(image_size)

images_to_load, ids = pick_images(base_dir, num_images * 2)
bases, fusees = get_images(base_dir, image_size, transform, images_to_load)
fusions = get_fusion_images(base_fusion_dir, image_size, transform, ids)

fusion_sample = torch.stack((bases, fusees, fusions), dim=1).flatten(end_dim=1)

# Show base images
caption = "base"
fig, axis = graphics.make_grid((caption, fusion_sample), 4, 3)
plt.savefig(os.path.join(output_dir, f"{caption}.png"))
print(caption)

bases = bases.to(device)
fusees = fusees.to(device)
for model_name, latent_dim in model_config.items():
    # Load Model
    model_path = os.path.join(model_prefix, f"{model_name}/model.pt")

    model_type = "vae" if "vae" in model_name else "autoencoder"
    model = load_model(
        model_type,
        max_filters,
        num_layers,
        image_size,
        latent_dim,
        small_conv,
        model_path,
        device,
    )

    with torch.no_grad():
        if model_type == "vae":
            # Get Model Outputs
            bases_out, _, _ = model(bases)
            fusees_out, _, _ = model(fusees)
            # Get Fusion Outputs
            base_mu, base_log_var = model.get_latent_variables(bases)
            fusee_mu, fusee_log_var = model.get_latent_variables(fusees)
            mu = (base_mu * 0.4) + (fusee_mu * 0.6)
            log_var = (base_log_var * 0.4) + (fusee_log_var * 0.6)
            midpoint_embedding = model.reparameterize(mu, log_var)
            fusions_out = model.decoder(midpoint_embedding)
        else:
            # Get Model Outputs
            bases_out = model(bases)
            fusees_out = model(fusees)
            # Get Fusion Outputs
            # Get Embeddings of Base Images
            base_embedding = model.encoder(bases) * 0.4
            fusee_embedding = model.encoder(fusees) * 0.6
            # Get Midpoint -> Decoder -> Fusion
            midpoint_embedding = base_embedding + fusee_embedding
            fusions_out = model.decoder(midpoint_embedding)

        # Group the images together
        fusion_sample = torch.stack(
            (bases_out.cpu(), fusees_out.cpu(), fusions_out.cpu()), dim=1
        ).flatten(end_dim=1)
    # Plot
    fig, axis = graphics.make_grid((model_name, fusion_sample), 4, 3)
    # Save
    plt.savefig(os.path.join(output_dir, f"{model_name}.png"))
    print(model_name)
