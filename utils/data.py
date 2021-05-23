import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

from PIL import Image
import os


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset=None, transform=None):
        self.dataset = list(dataset.values())
        self.keys = list(dataset.keys())
        self.transform = transform

    def __getitem__(self, index):
        data = self.dataset[index]
        key = self.keys[index]
        if self.transform is not None:
            data = self.transform(data)
        return key, data

    def __len__(self):
        return len(self.dataset)


class FusionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        fusion_dataset_path,
        base_train_images,
        base_val_images,
        base_test_images,
        transform,
    ):
        self.dataset_path = fusion_dataset_path
        self.all_images = os.listdir(fusion_dataset_path)
        self.transform = transform

        self.base_dataset = [base_train_images, base_val_images, base_test_images]

    def to_3_digit(self, num):
        return "0" * (3 - len(num)) + num

    def get_base_image(self, num, background):
        # if Base BW is not there, search for female BW
        filenames = [
            f"{num}_base_bw_{background}_0rotation.png",
            f"{num}_base_female_bw_{background}_0rotation.png",
        ]
        for dataset in self.base_dataset:
            for filename in filenames:
                if filename in dataset:
                    return dataset[filename], filename
        return None, None

    def __getitem__(self, index):
        fusion_filename = self.all_images[index]
        # Get two base names
        first, second, background, _ = fusion_filename.split(".")
        first = self.to_3_digit(first)
        second = self.to_3_digit(second)
        # Get Base
        image, base_filename = self.get_base_image(first, background)
        base = self.transform(image)
        # Get Fusee
        image, fusee_filename = self.get_base_image(second, background)
        fusee = self.transform(image)
        # Get Fusion
        fusion_loc = os.path.join(self.dataset_path, fusion_filename)
        image = Image.open(fusion_loc).convert("RGB")
        fusion = self.transform(image)
        return (base_filename, fusee_filename, fusion_filename), (base, fusee, fusion)

    def __len__(self):
        return len(self.all_images)


def load_images_from_folder(folder, use_noise_images):
    dataset = {}
    for file in os.listdir(folder):
        if "noise" in file and not use_noise_images:
            continue
        image = Image.open(os.path.join(folder, file))
        dataset[file] = np.array(image)
    print(f"Loaded {len(dataset)} images.")
    return dataset


def get_samples_from_data(data, sample_size, fusion=False):
    sample = []
    for i in np.random.choice(len(data), size=sample_size, replace=False):
        if fusion:
            sample.append([np.asarray(x) for x in data[i][1]])
        else:
            sample.append(np.asarray(data[i][1]))
    return torch.as_tensor(sample)


def image2tensor_resize(image_size):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                image_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
        ]
    )
