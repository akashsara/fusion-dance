import torch
import torch.nn as nn
import numpy as np


class CNNPrior(nn.Module):
    def __init__(self, input_channels, output_channels, input_dim, output_dim):
        super(CNNPrior, self).__init__()
        num_layers = self.get_number_of_layers(input_dim, output_dim)
        channel_sizes = self.calculate_channel_sizes(
            input_channels, output_channels, num_layers
        )
        layers = nn.ModuleList()

        for i, (in_channels, out_channels) in enumerate(channel_sizes):
            if i == 0 or i == len(channel_sizes) - 1:
                kernel_size = 1
                stride = 1
            else:
                kernel_size = 2
                stride = 2
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            )
            if i != len(channel_sizes) - 1:
                layers.append(nn.ReLU(True))
        self.model = nn.Sequential(*layers)

    def get_number_of_layers(self, input_dim, output_dim):
        num_layers = 2
        while input_dim != output_dim:
            output_dim *= 2
            num_layers += 1
        return num_layers

    def calculate_channel_sizes(self, image_channels, max_filters, num_layers):
        channel_sizes = [(image_channels, max_filters // np.power(2, num_layers - 1))]
        for i in range(1, num_layers):
            prev = channel_sizes[-1][-1]
            new = prev * 2
            channel_sizes.append((prev, new))
        return channel_sizes

    def forward(self, x):
        return self.model(x)


class CNNPriorV2(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        input_dim,
        output_dim,
        hidden_size,
        max_filters,
        kernel_size,
        stride,
    ):
        super(CNNPriorV2, self).__init__()
        flattened_size = self.calculate_flattened_size(
            input_dim, max_filters, kernel_size, stride
        )
        output_size = output_dim * output_dim * output_channels
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=max_filters // 2,
                kernel_size=1,
                stride=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=max_filters // 2,
                out_channels=max_filters,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(flattened_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, output_size),
        )
        self.output_channels = output_channels
        self.output_dim = output_dim

    def calculate_flattened_size(self, input_dim, max_filters, kernel_size, stride):
        new_size = ((input_dim - kernel_size) / stride) + 1
        return int(max_filters * new_size * new_size)

    def forward(self, x):
        return self.model(x).view(
            -1, self.output_channels, self.output_dim, self.output_dim
        )


class CNNPriorV3(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        input_dim,
        output_dim,
        hidden_size,
        max_filters,
        kernel_size,
        stride,
    ):
        super(CNNPriorV3, self).__init__()
        flattened_size = self.calculate_size(
            input_dim, max_filters, kernel_size, stride
        )
        decoder_input_size = ((output_dim // 2) ** 2) * max_filters // 2
        out_shape = [max_filters // 2, output_dim // 2, output_dim // 2]
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=max_filters // 2,
                kernel_size=1,
                stride=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=max_filters // 2,
                out_channels=max_filters,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(flattened_size, hidden_size // 2),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, decoder_input_size),
            nn.ReLU(True),
            nn.Unflatten(dim=1, unflattened_size=out_shape),
            nn.ConvTranspose2d(
                in_channels=max_filters // 2,
                out_channels=max_filters,
                kernel_size=2,
                stride=2,
            ),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=max_filters,
                out_channels=output_channels,
                kernel_size=1,
                stride=1,
            ),
            nn.ReLU(True),
        )

    def calculate_size(self, input_dim, max_filters, kernel_size, stride):
        new_size = ((input_dim - kernel_size) / stride) + 1
        return int(max_filters * new_size * new_size)

    def forward(self, x1, x2):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        return self.decoder(torch.cat([x1, x2], dim=1))


class InfillingCNNPrior(nn.Module):
    def __init__(self, num_layers, max_filters, input_channels=1):
        super(InfillingCNNPrior, self).__init__()

        layers = nn.ModuleList()
        assert max_filters % num_layers == 0
        # Conv Layers
        in_channels = input_channels
        out_channels = max_filters // 2 ** num_layers
        for i in range(num_layers):
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2,
                    stride=2,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(True))
            if i != num_layers - 1:
                in_channels = out_channels
                out_channels = out_channels * 2
        # Transposed Conv Layers
        for i in range(num_layers):
            in_channels = out_channels
            if i == num_layers - 1:
                out_channels = input_channels
            else:
                out_channels = out_channels // 2
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2,
                    stride=2,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            if i == num_layers - 1:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU(True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
