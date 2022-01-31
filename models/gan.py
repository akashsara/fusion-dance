from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim, num_filters, num_output_channels):
        super(DCGANGenerator, self).__init__()
        self.main = nn.Sequential(
            # input is latent_dim, going into a convolution
            nn.ConvTranspose2d(
                in_channels=latent_dim,
                out_channels=num_filters * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(num_filters * 8),
            nn.ReLU(True),
            # state size. (num_filters*8) x 4 x 4
            nn.ConvTranspose2d(
                in_channels=num_filters * 8,
                out_channels=num_filters * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU(True),
            # state size. (num_filters*4) x 8 x 8
            nn.ConvTranspose2d(
                in_channels=num_filters * 4,
                out_channels=num_filters * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(True),
            # state size. (num_filters*2) x 16 x 16
            nn.ConvTranspose2d(
                in_channels=num_filters * 2,
                out_channels=num_filters,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),
            # state size. (num_filters) x 32 x 32
            nn.ConvTranspose2d(
                in_channels=num_filters,
                out_channels=num_output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh()
            # state size. (num_output_channels) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class DCGANDiscriminator(nn.Module):
    def __init__(self, num_filters, num_output_channels):
        super(DCGANDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (num_output_channels) x 64 x 64
            nn.Conv2d(
                in_channels=num_output_channels,
                out_channels=num_filters,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_filters) x 32 x 32
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_filters*2) x 16 x 16
            nn.Conv2d(
                in_channels=num_filters * 2,
                out_channels=num_filters * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_filters*4) x 8 x 8
            nn.Conv2d(
                in_channels=num_filters * 4,
                out_channels=num_filters * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_filters*8) x 4 x 4
            nn.Conv2d(
                in_channels=num_filters * 8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)
