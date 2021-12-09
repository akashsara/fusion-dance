# Compares generated sprites of the specified models
# Does NOT load models or generate sprites.
# Only exception is the base model.

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import torch

sys.path.append("./")
from models import vqvae
from utils import data, graphics


def pick_images(dir, num_images=16, max_tries=10000, fusions=False):
    """
    Picks num_images sprites from the dataset such that no image appears twice.
    """
    unique_images = []
    selected_images = []
    all_images = os.listdir(dir)
    tries = -1
    while len(selected_images) < num_images:
        tries += 1
        if tries > max_tries:
            print("Error picking Images!")
            break
        selected = np.random.choice(all_images)
        if fusions:
            id_ = selected.split(".")[0]
        else:
            id_ = selected.split("_")[0]
        # Ignore Pokemon Already Selected
        if id_ in unique_images:
            continue
        selected_images.append(selected)
        unique_images.append(id_)
    return selected_images


def get_images(dir, images_to_load, fusion_dir=None):
    """
    Loads the images from the given directory.
    """
    images = []
    for image in images_to_load:
        file_path = os.path.join(dir, image)
        image = np.array(Image.open(file_path))
        images.append(image)
    return images


def make_image_grid(images, title):
    """
    Creates a square grid of all the images.
    """
    dimensions = np.sqrt(len(images)).astype("uint8")
    height, width = dimensions, dimensions
    i, j = 0, 0
    fig, axis = plt.subplots(height, width, figsize=(8, 6), dpi=150)
    for num, image in enumerate(images):
        if num == height * width:
            break
        axis[i, j].imshow(image)
        if j == width - 1:
            j = 0
            i += 1
        else:
            j += 1
    fig.suptitle(title, va="baseline")
    plt.tight_layout()
    return fig, axis


base_dir = "data\\pokemon\\final\\fusions\\test"
output_dir = "data\\"
model_prefix = f"outputs\\"
num_images = 16
model_list = [
    "cnn_multirnn_v5.4",
    "cnn_multirnn_v5.5",
    "cnn_multirnn_v5.6",
    "fusion_cnn_prior_v6",
    "fusion_cnn_linear_prior_v1",
    "fusion_cnn_autoencoding_prior_v2",
    "fusion_cnn_prior_eieo_v1",
]
is_fusions = "fusions" in base_dir
output_prefix = "fusions_" if is_fusions else ""

# VQ-VAE Config
get_base_model_predictions = True
image_size = 64
model_prefix = f"outputs\\"
vq_vae_model_name = f"vq_vae_v5.16"
vq_vae_num_layers = 2
vq_vae_max_filters = 512
vq_vae_use_max_filters = True
vq_vae_num_embeddings = 512
vq_vae_embedding_dim = 64
vq_vae_commitment_cost = 0.25
vq_vae_small_conv = True  # To use the 1x1 convolution layer

images_to_load = pick_images(base_dir, num_images, fusions=is_fusions)

caption = output_prefix + "base"
images = get_images(base_dir, images_to_load)
fig, axis = make_image_grid(images, caption)
plt.savefig(os.path.join(output_dir, f"{caption}.png"))
print(caption)

if get_base_model_predictions:
    # Setup Device
    gpu = torch.cuda.is_available()
    device = torch.device("cuda" if gpu else "cpu")

    # Setup Transform
    transform = data.image2tensor_resize(image_size)

    # Load Model
    model = vqvae.VQVAE(
        num_layers=vq_vae_num_layers,
        input_image_dimensions=image_size,
        small_conv=vq_vae_small_conv,
        embedding_dim=vq_vae_embedding_dim,
        num_embeddings=vq_vae_num_embeddings,
        commitment_cost=vq_vae_commitment_cost,
        use_max_filters=vq_vae_use_max_filters,
        max_filters=vq_vae_max_filters,
    )
    model_path = os.path.join(model_prefix, vq_vae_model_name, "model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Process Data
    images = torch.stack([transform(x) for x in images])

    # Get Model Output
    output = model(images)[1].detach().permute(0, 2, 3, 1).numpy()
    
    # Save
    caption = output_prefix + vq_vae_model_name
    fig, axis = make_image_grid(output, caption)
    plt.savefig(os.path.join(output_dir, f"{caption}.png"))
    print(caption)

for model in model_list:
    model_dir = os.path.join(model_prefix, model, "generated")
    images = get_images(model_dir, images_to_load)
    fig, axis = make_image_grid(images, model)
    plt.savefig(os.path.join(output_dir, f"{output_prefix}{model}.png"))
    print(model)