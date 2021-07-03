import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys


def pick_images(dir, num_images=16, max_tries=10000):
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
        id_ = selected.split("_")[0]
        # Ignore Fusion Images
        if '_' not in selected:
            continue
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


base_dir = "data\\final\\standard\\test"
output_dir = "data\\"
model_prefix = f"outputs\\tbd\\"
num_images = 16
model_list = [
    "vq_vae_v1",
    "vq_vae_v1.1",
    "vq_vae_v1.2",
    "vq_vae_v1.3",
    "vq_vae_v1.4",
    "vq_vae_v1.5",
]

images_to_load = pick_images(os.path.join(model_prefix, model_list[0], "generated"), num_images)

caption = "base"
images = get_images(base_dir, images_to_load)
fig, axis = make_image_grid(images, caption)
plt.savefig(os.path.join(output_dir, f"{caption}.png"))
print(caption)

for model in model_list:
    model_dir = os.path.join(model_prefix, model, "generated")
    images = get_images(model_dir, images_to_load)
    fig, axis = make_image_grid(images, model)
    plt.savefig(os.path.join(output_dir, f"{model}.png"))
    print(model)