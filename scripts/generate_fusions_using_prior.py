# Samples N images from the given data
# Load the original VQ-VAE and the prior model.
# Only supports CNNPrior
# Generates fusions using the models and saves.
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.append("./")
from models import vqvae, cnn_prior
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


def make_prediction(
    vq_vae, prior, to_predict, prior_output_dim, vq_vae_embedding_dim, device, mode
):
    encoding = prior(to_predict)
    if mode == "discrete":
        encoding = encoding.argmax(dim=1).flatten(start_dim=1).view(-1, 1)
        target_shape = (
            to_predict.shape[0],
            prior_output_dim,
            prior_output_dim,
            vq_vae_embedding_dim,
        )
        return vq_vae.quantize_and_decode(encoding, target_shape, device)
    elif mode == "continuous" or "continuous-final_image":
        return model.decoder(encoding)


if __name__ == "__main__":
    ## Config
    identifier = "val"  # Prepended to filename
    data_dir = f"data\\pokemon\\final\\standard\\{identifier}"
    fusion_dir = "data\\pokemon\\backup\\japeal_renamed"
    model_prefix = f"outputs\\"
    output_dir = f"data"

    image_size = 64
    use_noise_images = True
    num_images = 4  # Number of fusions

    vq_vae_model_name = f"vq_vae_v5.10"
    vq_vae_num_layers = 0
    vq_vae_max_filters = 512
    vq_vae_use_max_filters = True
    vq_vae_num_embeddings = 256
    vq_vae_embedding_dim = 32
    vq_vae_commitment_cost = 0.25
    vq_vae_small_conv = True  # To use the 1x1 convolution layer

    mode = "discrete"
    prior_model_name = f"fusion_cnn_prior_v2"
    prior_input_channels = 6  # Two Images
    prior_output_channels = (
        vq_vae_num_embeddings if mode == "discrete" else vq_vae_embedding_dim
    )
    prior_input_dim = image_size
    prior_output_dim = prior_input_dim // np.power(2, vq_vae_num_layers)

    ## Setup Devices
    gpu = torch.cuda.is_available()
    device = torch.device("cuda" if gpu else "cpu")

    ## Setup Transform
    transform = data.image2tensor_resize(image_size)

    ## Load VQVAE Model
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

    ## Load Prior
    prior = cnn_prior.CNNPrior(
        input_channels=prior_input_channels,
        output_channels=prior_output_channels,
        input_dim=prior_input_dim,
        output_dim=prior_output_dim,
    )
    model_path = os.path.join(model_prefix, prior_model_name, "model.pt")
    prior.load_state_dict(torch.load(model_path, map_location=device))
    prior.eval()

    ## Setup Data
    images_to_load, ids = pick_images(data_dir, num_images * 2)
    bases, fusees = get_images(data_dir, image_size, transform, images_to_load)
    fusions = get_fusion_images(fusion_dir, image_size, transform, ids)
    fusion_sample = torch.stack((bases, fusees, fusions), dim=1).flatten(end_dim=1)

    ## Plot Original Images
    caption = "Original Images"
    fig, axis = graphics.make_grid((caption, fusion_sample), 4, 3)
    plt.savefig(os.path.join(output_dir, f"{identifier}_{caption}.png"))
    print(caption)

    ## Move Data to Device
    bases = bases.to(device)
    fusees = fusees.to(device)
    fusions = fusions.to(device)

    ## Plot Reconstructions of VQ-VAE
    caption = "Reconstructed Images"
    # Get Model Outputs
    with torch.no_grad():
        recon_bases = model(bases)[1].detach().cpu()
        recon_fusees = model(fusees)[1].detach().cpu()
        recon_fusions = model(fusions)[1].detach().cpu()
    # Group the images together
    recon_sample = torch.stack(
        (recon_bases, recon_fusees, recon_fusions), dim=1
    ).flatten(end_dim=1)
    # Plot Reconsturcted Versions
    fig, axis = graphics.make_grid((caption, recon_sample), 4, 3)
    # Save
    plt.savefig(os.path.join(output_dir, f"{identifier}_{caption}.png"))
    print(caption)

    ## Plot Fusions using Prior
    with torch.no_grad():
        # Bases
        predict = torch.cat([bases, bases], dim=1)
        bases_final = make_prediction(
            model, prior, predict, prior_output_dim, vq_vae_embedding_dim, device, mode
        )
        # Fusees
        predict = torch.cat([fusees, fusees], dim=1)
        fusees_final = make_prediction(
            model, prior, predict, prior_output_dim, vq_vae_embedding_dim, device, mode
        )
        # Fusions - Combination
        predict = torch.cat([bases, fusees], dim=1)
        fusions_final = make_prediction(
            model, prior, predict, prior_output_dim, vq_vae_embedding_dim, device, mode
        )
        # Fusions - Reconstruction
        predict = torch.cat([fusions, fusions], dim=1)
        fusions_recon = make_prediction(
            model, prior, predict, prior_output_dim, vq_vae_embedding_dim, device, mode
        )
    # Plot Reconstructed Fusion Version
    caption = "Prior Reconstructions"
    final_sample = torch.stack(
        (bases_final, fusees_final, fusions_recon), dim=1
    ).flatten(end_dim=1)
    fig, axis = graphics.make_grid((caption, final_sample), 4, 3)
    plt.savefig(os.path.join(output_dir, f"{identifier}_{caption}.png"))
    print(caption)
    # Plot Reconstructed Fusion Version
    caption = "Prior Outputs"
    final_sample = torch.stack(
        (bases_final, fusees_final, fusions_final), dim=1
    ).flatten(end_dim=1)
    fig, axis = graphics.make_grid((caption, final_sample), 4, 3)
    plt.savefig(os.path.join(output_dir, f"{identifier}_{caption}.png"))
    print(caption)