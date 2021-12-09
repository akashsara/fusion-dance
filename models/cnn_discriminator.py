import torch
import torch.nn as nn
import numpy as np


class CNNDiscriminator(nn.Module):
    def __init__(self, input_channels, input_dim, num_filters, num_layers):
        super(CNNDiscriminator, self).__init__()
        channel_sizes = self.calculate_channel_sizes(
            input_channels, num_filters, num_layers
        )
        layers = nn.ModuleList()

        kernel_size = 1
        stride = 1
        for i, (in_channels, out_channels) in enumerate(channel_sizes):
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            )
            layers.append(nn.ReLU(True))
            if i == 0:
                kernel_size = 2
                stride = 2
        layers.append(nn.Flatten())
        flattened_size = num_filters * ((input_dim // np.power(2, num_layers - 1)) ** 2)
        layers.append(nn.Linear(flattened_size, 1))
        self.model = nn.Sequential(*layers)

    def calculate_channel_sizes(self, image_channels, max_filters, num_layers):
        channel_sizes = [(image_channels, max_filters // np.power(2, num_layers - 1))]
        for i in range(1, num_layers):
            prev = channel_sizes[-1][-1]
            new = prev * 2
            channel_sizes.append((prev, new))
        return channel_sizes

    def forward(self, x):
        return self.model(x)
