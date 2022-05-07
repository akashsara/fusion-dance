import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import pandas as pd
import joblib
import random

from PIL import Image
import os


class ConditioningLabelsHandler:
    """
    A class to handle conditioning information for certain models.
    If a single column is given, each call returns a one hot vector.
    For multiple columns, we treat it as a multiclass system.
    So each call would return an n-hot vector.
    If there were 2 columns, we would have the same one hot vector as above.
    BUT, we would have two instances of "1" in this vector.
    Note that the second may be optional as there may be cases where
    certain data can have 1 or 2 labels.
    In either case, the first column is used for setting up our label encoder.
    This is because the first column is mandatory for all data.
    """

    def __init__(self, label_file, label_columns):
        self.labels = joblib.load(label_file)
        self.label_columns = label_columns
        column_unique_values = []
        column_unique_values_dict = {}
        df = pd.DataFrame(self.labels).T
        for label_column in label_columns:
            temp = df[~df[label_column].isnull()][label_column].unique()
            # Not using a set here so members of the same class (type/egg/etc.)
            # are next to each other in the vector 
            # This is mostly just so that it is easier to understand the vector
            for item in temp:
                if item not in column_unique_values:
                    column_unique_values.append(item)
            column_unique_values_dict[label_column] = temp
        self.column_unique_values_dict = column_unique_values_dict
        self.encoding_dict = {y: x for x, y in enumerate(column_unique_values)}
        self.reverse_encoding_dict = {x: y for x, y in enumerate(column_unique_values)}
        self.conditioning_size = len(self.encoding_dict)

    def __call__(self, keys):
        vectors = []
        for key in keys:
            vector = [0] * self.conditioning_size
            for column in self.label_columns:
                label_column = self.labels.get(key)
                # Weird python thing to check for nan
                if str(label_column[column]) != "nan":
                    vector[self.encoding_dict[label_column[column]]] = 1
            vectors.append(vector)
        return vectors

    def reverse_transform(self, label):
        label = np.array(label)
        # Single number
        if label.ndim == 0:
            return self.reverse_encoding_dict[int(label)]
        # Single array of numbers
        elif label.ndim == 1:
            return [self.reverse_encoding_dict[int(x)] for x in label]
        # (batch_size, N)
        elif label.ndim == 2:
            return [[self.reverse_encoding_dict[int(x)] for x in row] for row in label]

    def vector_to_text(self, vector):
        vector = np.array(vector)
        if vector.ndim == 1:
            vector = np.flatnonzero(vector)
            return self.reverse_transform(vector)
        elif vector.ndim == 2:
            return [self.reverse_transform(np.flatnonzero(row)) for row in vector]

    def sample_conditions(self, num_samples, columns):
        vectors = []
        for _ in range(num_samples):
            vector = [0] * self.conditioning_size
            for column, column_type in columns.items():
                # Ignore column
                if column_type == 0:
                    pass
                # Always use column
                elif column_type == 1:
                    choice = random.choice(self.column_unique_values_dict[column])
                    vector[self.encoding_dict[choice]] = 1
                # Leave it to luck (for secondary things like type2)
                elif column_type == 2 and random.choice([True, True, True, False, False]):
                    choice = random.choice(self.column_unique_values_dict[column])
                    vector[self.encoding_dict[choice]] = 1
            vectors.append(vector)
        return vectors

    def get_size(self):
        """Returns the size of the conditioning vector"""
        return self.conditioning_size

    def save(self, output_file):
        joblib.dump(self.encoding_dict, output_file)


class CustomDataset(torch.utils.data.Dataset):
    """
    Requires the dataset as a dict of form filename:image.
    Returns filename, image
    """

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


class CustomDatasetWithLabels(torch.utils.data.Dataset):
    """
    Requires the dataset as a dict of form filename:image.
    Returns filename, image
    """

    def __init__(self, dataset, label_file, label_column, transform=None):
        self.dataset = list(dataset.values())
        self.keys = list(dataset.keys())
        labels = joblib.load(label_file)
        labels = pd.DataFrame(labels).T[label_column].fillna("None")
        self.classes = labels.unique()
        self.labels = labels.to_dict()
        self.transform = transform

    def __getitem__(self, index):
        data = self.dataset[index]
        key = self.keys[index]
        label = self.labels[key]
        if self.transform is not None:
            data = self.transform(data)
        return key, data, label

    def __len__(self):
        return len(self.dataset)

    def get_classes(self):
        return self.classes


class CustomDatasetNoMemory(torch.utils.data.Dataset):
    """
    Requires the path to a dataset.
    Essentially the same as above but it doesn't load all the data to memory.
    Returns filename, image.
    """

    def __init__(self, dataset_directory, transform, use_noise_images):
        self.dataset_path = dataset_directory
        all_images = os.listdir(dataset_directory)
        if not use_noise_images:
            all_images = [x for x in all_images if "noise" not in x]
        self.all_images = all_images
        self.transform = transform

    def __getitem__(self, index):
        filename = self.all_images[index]
        image_path = os.path.join(self.dataset_path, filename)
        image = Image.open(image_path).convert("RGB")
        processed = self.transform(image)
        return filename, processed

    def __len__(self):
        return len(self.all_images)


class CustomDatasetNoMemoryWithLabels(torch.utils.data.Dataset):
    """
    Requires the path to a dataset.
    Essentially the same as above but it doesn't load all the data to memory.
    Returns filename, image.
    """

    def __init__(
        self,
        dataset_directory,
        label_file,
        label_column,
        use_noise_images,
        transform=None,
    ):
        self.dataset_path = dataset_directory
        all_images = os.listdir(dataset_directory)
        if not use_noise_images:
            all_images = [x for x in all_images if "noise" not in x]
        self.all_images = all_images
        labels = joblib.load(label_file)
        labels = pd.DataFrame(labels).T[label_column].fillna("None")
        self.classes = labels.unique()
        self.labels = labels.to_dict()
        self.transform = transform

    def __getitem__(self, index):
        filename = self.all_images[index]
        image_path = os.path.join(self.dataset_path, filename)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[filename]
        return filename, image, label

    def __len__(self):
        return len(self.all_images)

    def get_classes(self):
        return self.classes


class CustomImage2ImageDatasetWithLabels(torch.utils.data.Dataset):
    """
    Unlike the previous CustomDatasets,
    this requires both a feature dir and a labels dir.
    Essentially we use this for tasks where we want to transform
    the input feature image into the output label image.
    I.E. our FusionEnhancer.
    Returns filename, feature_image, label_image
    """

    def __init__(self, features_directory, labels_directory, transform):
        self.features_directory = features_directory
        self.labels_directory = labels_directory
        self.all_images = os.listdir(features_directory)
        self.transform = transform

    def __getitem__(self, index):
        filename = self.all_images[index]
        image_path = os.path.join(self.features_directory, filename)
        feature = self.transform(Image.open(image_path).convert("RGB"))
        image_path = os.path.join(self.labels_directory, filename)
        label = self.transform(Image.open(image_path).convert("RGB"))
        return filename, feature, label

    def __len__(self):
        return len(self.all_images)


class CustomDatasetNoMemoryAddBackground(torch.utils.data.Dataset):
    """
    This is for generating embeddings. It just loads all the data.
    In some cases we add a background.
    """

    def __init__(self, dataset_directory, dataset, transform, background_color):
        self.dataset_path = dataset_directory
        all_images = os.listdir(dataset_directory)
        approved_images = []
        done = set()
        if "pokemon" in dataset:
            # We want only one sprite per Pokemon form
            # For consistency we try to use the most recent game sprites
            for image in all_images:
                image_id = image.split("_")[0]
                if image_id in done:
                    continue
                image_formats = [
                    f"{image_id}_base_bw.png",
                    f"{image_id}_base_hgss.png",
                    f"{image_id}_base_plat.png",
                    f"{image_id}_base_dp.png",
                ]
                for image in image_formats:
                    if image in all_images:
                        approved_images.append(image)
                        done.add(image_id)
                        break
                else:
                    print(f"Exception. No matching image found for: {image_id}")
        elif dataset == "tinyhero":
            # No special instructions needed
            approved_images = all_images
        elif dataset == "sprites":
            # Find all unique sprite IDs
            for image in all_images:
                image_id = image.split("_")[0]
                if image_id in done:
                    continue
                done.add(image_id)
            # Sample N=10% IDs
            random.seed(42)
            sampled = random.sample(list(done), k=len(done) // 10)
            # Save only those selected IDs
            approved_images = [x for x in all_images if x.split("_")[0] in sampled]
        self.all_images = approved_images
        self.transform = transform
        self.background = background_color  # Tuple (R,G,B)
        self.dataset = dataset

    def __getitem__(self, index):
        filename = self.all_images[index]
        image_path = os.path.join(self.dataset_path, filename)
        if self.dataset != "sprites":
            # Sprites dataset has an existing black background
            image = Image.open(image_path).convert("RGBA")
            background = Image.new("RGBA", image.size, self.background)
            image = Image.alpha_composite(background, image).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
        processed = self.transform(image)
        return filename, processed

    def __len__(self):
        return len(self.all_images)


class InpaintingDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transform):
        self.dataset_path = dataset_path
        index_path = os.path.join(dataset_path, "index.joblib")
        self.dataset_index = joblib.load(index_path)
        self.transform = transform

    def __getitem__(self, index):
        filename = self.dataset_index[index]
        src_filename = os.path.join(self.dataset_path, "inputs", filename)
        target_filename = os.path.join(self.dataset_path, "labels", filename)
        src = Image.open(src_filename).convert("RGB")
        target = Image.open(target_filename).convert("RGB")
        src = self.transform(src)
        target = self.transform(target)
        return filename, src, target

    def __len__(self):
        return len(self.dataset_index)


class EverythingDataset(torch.utils.data.Dataset):
    """
    For use by GANs.
    Essentially loads images from all datasets since GANs
    don't have a val or test dataset.
    """

    def __init__(
        self, train_datapath, val_datapath, test_datapath, transform, use_noise_images
    ):
        all_images = []
        for datapath in [train_datapath, val_datapath, test_datapath]:
            for file in os.listdir(datapath):
                if use_noise_images or "noise" not in file:
                    all_images.append(os.path.join(datapath, file))
        self.all_images = all_images
        self.transform = transform

    def __getitem__(self, index):
        filename = self.all_images[index]
        image = Image.open(filename).convert("RGB")
        image = self.transform(image)
        return filename, image

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
    return torch.as_tensor(np.array(sample))


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
