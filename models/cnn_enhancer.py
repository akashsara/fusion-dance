import torch
import torch.nn as nn
import numpy as np


class ImageEnhancerCNN(nn.Module):
    def __init__(self, input_channels, num_filters, num_layers, use_4by4=False):
        super(ImageEnhancerCNN, self).__init__()
        self.use_4by4 = use_4by4
        if num_layers < 2:
            raise ValueError("Model should have at least two layers.")
        if self.use_4by4:
            kernel_size = 4
            stride = 4
        else:
            num_layers -= 1
            kernel_size = 1
            stride = 1
        channel_sizes = self.calculate_channel_sizes(
            input_channels, num_filters, num_layers
        )
        layers = nn.ModuleList()
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
                stride = 1
                kernel_size = 1
        if self.use_4by4:
            layers.append(nn.Flatten())
            in_size = num_filters * 16 * 16
            out_size = input_channels * 64 * 64
            layers.append(nn.Linear(in_size, out_size))
        else:
            # Output Layer
            layers.append(
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=input_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            )
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def calculate_channel_sizes(self, image_channels, max_filters, num_layers):
        channel_sizes = [(image_channels, max_filters // np.power(2, num_layers - 1))]
        for i in range(1, num_layers):
            prev = channel_sizes[-1][-1]
            new = prev * 2
            channel_sizes.append((prev, new))
        return channel_sizes

    def forward(self, x):
        if self.use_4by4:
            x = self.model(x)
            return x.view(-1, 3, 64, 64)
        else:
            return self.model(x)
