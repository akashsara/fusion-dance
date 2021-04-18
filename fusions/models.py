import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np


class ConvolutionalAE(nn.Module):
    def __init__(
        self,
        image_channels=3,
        max_filters=512,
        num_layers=4,
        kernel_size=2,
        stride=2,
        padding=0,
        latent_dim=128,
        input_image_dimensions=96,
    ):
        super(ConvolutionalAE, self).__init__()
        channel_sizes = self.calculate_channel_sizes(
            image_channels, max_filters, num_layers
        )
        # Calculate shape of the flattened image
        hidden_dim, image_size = self.get_flattened_size(
            kernel_size, stride, max_filters, input_image_dimensions, num_layers
        )

        # Encoder
        encoder_layers = nn.ModuleList()
        for i, channel_size in enumerate(channel_sizes):
            in_channels = channel_size[0]
            out_channels = channel_size[1]
            # Convolutional Layer
            encoder_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                )
            )
            # Batch Norm
            encoder_layers.append(nn.BatchNorm2d(out_channels))
            # ReLU
            encoder_layers.append(nn.ReLU())
        # Flatten Encoder Output
        encoder_layers.append(nn.Flatten())
        # Hidden Dim -> Latent Dim
        encoder_layers.append(nn.Linear(hidden_dim, latent_dim))
        encoder_layers.append(nn.Sigmoid())
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = nn.ModuleList()
        # Latent Dim -> Hidden Dim
        decoder_layers.append(nn.Linear(latent_dim, hidden_dim))
        decoder_layers.append(nn.Sigmoid())
        # Unflatten to a shape of (Channels, Height, Width)
        decoder_layers.append(nn.Unflatten(1, (max_filters, image_size, image_size)))
        for i, channel_size in enumerate(channel_sizes[::-1]):
            in_channels = channel_size[1]
            out_channels = channel_size[0]
            # Add Transposed Convolutional Layer
            decoder_layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                )
            )
            # Batch Norm
            decoder_layers.append(nn.BatchNorm2d(out_channels))
            # ReLU if not final layer
            if i != num_layers - 1:
                decoder_layers.append(nn.ReLU())
            # Sigmoid if final layer
            else:
                decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def get_flattened_size(
        self, kernel_size, stride, filters, input_image_dimensions, num_layers
    ):
        x = input_image_dimensions
        for i in range(num_layers):
            x = ((x - kernel_size) // stride) + 1
        return filters * x * x, x

    def calculate_channel_sizes(self, image_channels, max_filters, num_layers):
        channel_sizes = [(image_channels, max_filters // np.power(2, num_layers - 1))]
        for i in range(1, num_layers):
            prev = channel_sizes[-1][-1]
            new = prev * 2
            channel_sizes.append((prev, new))
        return channel_sizes

    def forward(self, x):
        # Encode
        hidden_state = self.encoder(x)
        # Decode
        reconstructed = self.decoder(hidden_state)
        return reconstructed


# Ref: https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb
class ConvolutionalVAE(nn.Module):
    def __init__(
        self,
        image_channels=3,
        max_filters=512,
        num_layers=4,
        kernel_size=2,
        stride=2,
        padding=0,
        latent_dim=128,
        input_image_dimensions=96,
    ):
        super(ConvolutionalVAE, self).__init__()
        channel_sizes = self.calculate_channel_sizes(
            image_channels, max_filters, num_layers
        )
        # Encoder
        encoder_layers = nn.ModuleList()
        for i, channel_size in enumerate(channel_sizes):
            in_channels = channel_size[0]
            out_channels = channel_size[1]
            # Convolutional Layer
            encoder_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                )
            )
            # Batch Norm
            encoder_layers.append(nn.BatchNorm2d(out_channels))
            # ReLU
            encoder_layers.append(nn.ReLU())
        # Flatten Encoder Output
        encoder_layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*encoder_layers)

        # Calculate shape of the flattened image
        hidden_dim, image_size = self.get_flattened_size(
            kernel_size, stride, max_filters, input_image_dimensions
        )

        # Latent Space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        decoder_layers = nn.ModuleList()
        # Feedforward/Dense Layer to expand our latent dimensions
        decoder_layers.append(nn.Linear(latent_dim, hidden_dim))
        # Unflatten to a shape of (Channels, Height, Width)
        decoder_layers.append(nn.Unflatten(1, (max_filters, image_size, image_size)))
        for i, channel_size in enumerate(channel_sizes[::-1]):
            in_channels = channel_size[1]
            out_channels = channel_size[0]
            # Add Transposed Convolutional Layer
            decoder_layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                )
            )
            # Batch Norm
            decoder_layers.append(nn.BatchNorm2d(out_channels))
            # ReLU if not final layer
            if i != num_layers - 1:
                decoder_layers.append(nn.ReLU())
            # Sigmoid if final layer
            else:
                decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def get_flattened_size(self, kernel_size, stride, filters, input_image_dimensions):
        x = input_image_dimensions
        for layer in self.encoder:
            if "Conv2d" in str(layer):
                x = ((x - kernel_size) // stride) + 1
        return filters * x * x, x

    def calculate_channel_sizes(self, image_channels, max_filters, num_layers):
        channel_sizes = [(image_channels, max_filters // np.power(2, num_layers - 1))]
        for i in range(1, num_layers):
            prev = channel_sizes[-1][-1]
            new = prev * 2
            channel_sizes.append((prev, new))
        return channel_sizes

    def forward(self, x):
        # Encode
        hidden_state = self.encoder(x)
        # Reparameterize
        mu = self.fc_mu(hidden_state)
        log_var = self.fc_log_var(hidden_state)
        z = self.reparameterize(mu, log_var)
        # Decode
        reconstructed = self.decoder(z)
        return reconstructed, mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var.mul(0.5))  # log sqrt(x) = log x^0.5 = 0.5 log x
        epsilon = torch.randn_like(mu)
        z = mu + (epsilon * std)
        return z