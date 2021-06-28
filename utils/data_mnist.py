import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

from PIL import Image
import os


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset=None, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        data = self.dataset[index]
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.dataset)


def image2tensor():
    return transforms.Compose([transforms.ToTensor()])

def get_samples_from_data(data, sample_size, fusion=False):
    sample = []
    for i in np.random.choice(len(data), size=sample_size, replace=False):
        sample.append(np.asarray(data[i]))
    return torch.as_tensor(sample)