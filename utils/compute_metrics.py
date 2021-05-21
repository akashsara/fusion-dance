import os
import sys

import torch
from PIL import Image
from pytorch_msssim import ssim
from torchvision import transforms
from tqdm import tqdm


def load_images_from_dir(dir, transform):
    images = []
    for image in tqdm(os.listdir(dir)):
        image = Image.open(os.path.join(dir, image))
        image = transform(image).unsqueeze(0)
        images.append(image)
    return torch.cat(images)


input_dir = sys.argv[1]
output_dir = sys.argv[2]
image_size = int(sys.argv[3])

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(
            image_size, interpolation=transforms.InterpolationMode.BICUBIC
        ),
    ]
)

inputs = load_images_from_dir(input_dir, transform)
print(inputs.shape)
outputs = load_images_from_dir(output_dir, transform)
print(outputs.shape)

mse = torch.nn.functional.mse_loss(outputs, inputs)
ssim_score = ssim(
    outputs, inputs, data_range=1.0, win_size=11, win_sigma=1.5, K=(0.01, 0.03)
)

# Print Metrics
print(f"MSE = {mse}, SSIM = {ssim_score}")
