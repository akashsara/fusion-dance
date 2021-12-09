# Samples N images from the given data
# Then for each model specified, it loads the model.
# Generates fusions using that model.
# And Saves.
# Supports only Single Model Architectures.
# So no RNNs/MultiRNNs/Fusion Priors etc.
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.random import f
from PIL import Image

sys.path.append("./")
from models import vqvae, autoencoder, vae
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
        # Allow multiple forms of the same Pokemon
        if id_ in unique_ids:
            continue
        # To guarantee we have a fusion
        id_ = id_.split("-")[0]
        if int(id_) > 251:
            continue
        # To ensure color/sprite matches fusion
        filenames = [
            f"{id_}_base_bw_whiteBG_0rotation.png",
            f"{id_}_base_female_bw_whiteBG_0rotation.png",
        ]
        if choice not in filenames:
            for filename in filenames:
                if filename in all_images:
                    choice = filename
                    break
        if choice not in filenames:
            print("MEEP MORP")
        selected.append(choice)
        unique_ids.append(id_)
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
    image_size,
    model_parameters,
    small_conv,
    model_path,
    device,
):
    if model_type == "vae":
        model = vae.ConvolutionalVAE(
            max_filters=max_filters,
            num_layers=num_layers,
            input_image_dimensions=image_size,
            latent_dim=model_parameters,
            small_conv=small_conv,
        )
    elif model_type == "dual_input_vae":
        model = vae.FusionVAE(
            max_filters=max_filters,
            num_layers=num_layers,
            input_image_dimensions=image_size,
            latent_dim=model_parameters,
            small_conv=small_conv,
        )
    elif model_type == "vq_vae":
        num_embeddings = model_parameters["K"]
        embedding_dim = model_parameters["D"]
        commitment_cost = model_parameters["commitment_cost"]
        model = vqvae.VQVAE(
            num_layers=num_layers,
            input_image_dimensions=image_size,
            small_conv=small_conv,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost,
        )
    elif model_type == "dual_input_autoencoder":
        model = autoencoder.FusionAE(
            max_filters=max_filters,
            num_layers=num_layers,
            input_image_dimensions=image_size,
            latent_dim=model_parameters,
            small_conv=small_conv,
        )
    else:
        model = autoencoder.ConvolutionalAE(
            max_filters=max_filters,
            num_layers=num_layers,
            input_image_dimensions=image_size,
            latent_dim=model_parameters,
            small_conv=small_conv,
        )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


gpu = torch.cuda.is_available()
device = torch.device("cuda" if gpu else "cpu")

base_dir = "data\\final\\standard\\test"
base_fusion_dir = "data\\backup\\japeal_renamed"
output_dir = "data"
model_prefix = f"outputs\\tbd\\"
num_images = 4  # Number of fusions
batch_size = 64
num_layers = 4
max_filters = 512
image_size = 64
small_conv = True  # To use the 1x1 convolution layer
vq_vae_fusion_version = 3

# model_name: latent_dim
model_config = {
    "vq_vae_v1": {"K": 512, "D": 64, "commitment_cost": 0.25},
    "vq_vae_v1.1": {"K": 512, "D": 64, "commitment_cost": 1.0},
    "vq_vae_v1.2": {"K": 1024, "D": 64, "commitment_cost": 0.25},
    "vq_vae_v1.3": {"K": 256, "D": 64, "commitment_cost": 0.25},
    "vq_vae_v1.4": {"K": 512, "D": 128, "commitment_cost": 0.25},
    "vq_vae_v1.5": {"K": 512, "D": 32, "commitment_cost": 0.25},
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
for model_name, model_parameters in model_config.items():
    # Load Model
    model_path = os.path.join(model_prefix, f"{model_name}/model.pt")
    temp_num_layers = num_layers

    model_type = "autoencoder"
    if "dual_input" in model_name:
        if "vae" in model_name:
            model_type = "dual_input_vae"
        else:
            model_type = "dual_input_autoencoder"
    elif "vq_vae" in model_name:
        model_type = "vq_vae"
        temp_num_layers = 2
    elif "vae" in model_name:
        model_type = "vae"

    model = load_model(
        model_type,
        max_filters,
        temp_num_layers,
        image_size,
        model_parameters,
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
        elif model_type == "dual_input_vae":
            bases_out, _, _ = model(bases, bases)
            fusees_out, _, _ = model(fusees, fusees)
            fusions_out, _, _ = model(bases, fusees)
        elif model_type == "dual_input_autoencoder":
            bases_out = model(bases, bases)
            fusees_out = model(fusees, fusees)
            fusions_out = model(bases, fusees)
        elif model_type == "vq_vae":
            # Get Model Outputs
            _, bases_out, _, _ = model(bases)
            _, fusees_out, _, _ = model(fusees)
            # Get Fusion Outputs
            base_embedding = model.encoder(bases)
            fusee_embedding = model.encoder(fusees)

            if vq_vae_fusion_version == 1:
                # Version 1
                midpoint_embedding = (base_embedding * 0.4) + (fusee_embedding * 0.6)
                _, quantized, _, _ = model.vq_vae(midpoint_embedding)
                fusions_out = model.decoder(quantized)
            elif vq_vae_fusion_version == 2:
                # Version 2
                _, base_quantized, _, _ = model.vq_vae(base_embedding)
                _, fusee_quantized, _, _ = model.vq_vae(fusee_embedding)
                midpoint_quantized = (base_quantized * 0.4) + (fusee_quantized * 0.6)
                fusions_out = model.decoder(midpoint_quantized)
            elif vq_vae_fusion_version == 3:
                # Version 3
                _, _, _, base_encoding_indices = model.vq_vae(base_embedding)
                _, _, _, fusee_encoding_indices = model.vq_vae(fusee_embedding)
                fused = torch.zeros_like(base_encoding_indices)
                for i, _ in enumerate(fused):
                    if torch.rand(1) < 0.5:
                        fused[i] = base_encoding_indices[i]
                    else:
                        fused[i] = fusee_encoding_indices[i]
                height = np.sqrt(fused.shape[0] / bases.shape[0]).astype(np.int32)
                width = height
                target_shape = (bases.shape[0], height, width, model_parameters["D"])
                fusions_out = model.quantize_and_decode(fused, target_shape, device)

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
