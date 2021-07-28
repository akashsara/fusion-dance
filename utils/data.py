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


class CustomDatasetV2(torch.utils.data.Dataset):
    def __init__(self, dataset_directory, transform):
        self.dataset_path = dataset_directory
        self.all_images = os.listdir(dataset_directory)
        self.transform = transform

    def __getitem__(self, index):
        filename = self.all_images[index]
        image_path = os.path.join(self.dataset_path, filename)
        image = Image.open(image_path).convert("RGB")
        fusion = self.transform(image)
        return filename, fusion

    def __len__(self):
        return len(self.all_images)


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


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, fusion_dataset_path, transform, use_noise_images):
        self.dataset_path = dataset_path
        self.fusion_dataset_path = fusion_dataset_path
        if use_noise_images:
            self.base_images = os.listdir(dataset_path)
            self.fusion_images = os.listdir(fusion_dataset_path)
        else:
            all_images = os.listdir(dataset_path)
            self.base_images = [x for x in all_images if "noise" not in x]
            all_images = os.listdir(fusion_dataset_path)
            self.fusion_images = [x for x in all_images if "noise" not in x]
        self.all_images = self.base_images + self.fusion_images
        self.transform = transform

    def __getitem__(self, index):
        filename = self.all_images[index]
        if filename in self.base_images:
            filepath = os.path.join(self.dataset_path, filename)
        elif filename in self.fusion_images:
            filepath = os.path.join(self.fusion_dataset_path, filename)
        else:
            raise ValueError(f"Could not find file: {filename}")
        image = Image.open(filepath).convert("RGB")
        image = self.transform(image)
        return filename, image

    def __len__(self):
        return len(self.all_images)


class FusionDatasetV2(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path,
        fusion_dataset_path,
        dataset_parent_dir,
        transform,
        use_noise_images,
        only_fusions=False,
    ):
        self.dataset_path = dataset_path
        self.fusion_dataset_path = fusion_dataset_path
        self.dataset_parent_dir = dataset_parent_dir
        """
        # We cache all file locations here regardless of train/test/val
        # However, train fusions only use images from the training set.
        # This is needed for val & test fusions 
        # Which may need base images from the training data
        """
        all_data = {}
        for folder in os.listdir(dataset_parent_dir):
            all_data[folder] = []
            for file in os.listdir(os.path.join(dataset_parent_dir, folder)):
                all_data[folder].append(file)
            all_data[folder] = set(all_data[folder])
        self.all_data = all_data
        if use_noise_images:
            self.base_images = os.listdir(dataset_path)
            self.fusion_images = os.listdir(fusion_dataset_path)
        else:
            all_images = os.listdir(dataset_path)
            self.base_images = [x for x in all_images if "noise" not in x]
            all_images = os.listdir(fusion_dataset_path)
            self.fusion_images = [x for x in all_images if "noise" not in x]
        if only_fusions:
            self.all_images = self.fusion_images
        else:
            self.all_images = self.base_images + self.fusion_images
        self.transform = transform

    def to_3_digit(self, num):
        return "0" * (3 - len(num)) + num

    def get_base_image(self, num, background):
        # if Base BW is not there, search for female BW
        filenames = [
            f"{num}_base_bw_{background}_0rotation.png",
            f"{num}_base_female_bw_{background}_0rotation.png",
        ]
        for dataset in self.all_data:
            for filename in filenames:
                if filename in self.all_data[dataset]:
                    filepath = os.path.join(self.dataset_parent_dir, dataset, filename)
                    image = Image.open(filepath).convert("RGB")
                    return image, filename
        return None, None

    def __getitem__(self, index):
        filename = self.all_images[index]
        if filename in self.base_images:
            # If normal image
            filepath = os.path.join(self.dataset_path, filename)
            image = Image.open(filepath).convert("RGB")
            image = self.transform(image)
            base_name = fusee_name = fusion_name = filename
            base = fusee = fusion = image
        elif filename in self.fusion_images:
            # If fusion
            # Get two base names
            base, fusee, background, _ = filename.split(".")
            base = self.to_3_digit(base)
            fusee = self.to_3_digit(fusee)
            # Get Base
            base, base_name = self.get_base_image(base, background)
            base = self.transform(base)
            # Get Fusee
            fusee, fusee_name = self.get_base_image(fusee, background)
            fusee = self.transform(fusee)
            # Get Fusion
            fusion_name = filename
            fusion = os.path.join(self.fusion_dataset_path, fusion_name)
            fusion = Image.open(fusion).convert("RGB")
            fusion = self.transform(fusion)
        else:
            raise ValueError(f"Could not find file: {filename}")
        return (base_name, fusee_name, fusion_name), (base, fusee, fusion)

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


def get_samples_from_FusionDatasetV2(data, sample_size, mode):
    samples = []
    previous_i = []
    iterations = 0
    while len(samples) < sample_size and iterations <= 10000:
        i = np.random.choice(len(data))
        while i in previous_i:
            i = np.random.choice(len(data))
        _, (base, fusee, fusion) = data[i]
        tensors_are_same = torch.equal(base, fusee)
        if mode == "standard" and tensors_are_same:
            samples.append(np.asarray(base))
        elif mode == "fusion" and not tensors_are_same:
            sample = [np.asarray(x) for x in [base, fusee, fusion]]
            samples.append(sample)
        previous_i.append(i)
        iterations += 1
    if len(samples) != sample_size:
        raise ValueError(
            f"Error obtaining samples. Iterations={iterations}, Sample Size={sample_size}, Samples Obtained={len(samples)}"
        )
    return torch.as_tensor(samples)


def image2tensor_resize(image_size):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                image_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
        ]
    )
