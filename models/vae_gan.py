import torch
import torch.nn as nn
import numpy as np


class VAEGANEncoder(nn.Module):
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
        small_conv=False,
    ):
        super(VAEGANEncoder, self).__init__()
        if small_conv:
            num_layers += 1
        channel_sizes = self.calculate_channel_sizes(
            image_channels, max_filters, num_layers
        )

        # Encoder
        encoder_layers = nn.ModuleList()
        # Encoder Convolutions
        for i, (in_channels, out_channels) in enumerate(channel_sizes):
            if small_conv and i == 0:
                # 1x1 Convolution
                encoder_layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            else:
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
        hidden_dim, image_size = self.get_flattened_size(input_image_dimensions)

        # Latent Space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def calculate_layer_size(self, input_size, kernel_size, stride, padding=0):
        numerator = input_size - kernel_size + (2 * padding)
        denominator = stride
        return (numerator // denominator) + 1

    def get_flattened_size(self, image_size):
        for layer in self.encoder:
            if "Conv2d" in str(layer):
                kernel_size = layer.kernel_size[0]
                stride = layer.stride[0]
                padding = layer.padding[0]
                filters = layer.out_channels
                image_size = self.calculate_layer_size(
                    image_size, kernel_size, stride, padding
                )
        return filters * image_size * image_size, image_size

    def calculate_channel_sizes(self, image_channels, max_filters, num_layers):
        channel_sizes = [(image_channels, max_filters // np.power(2, num_layers - 1))]
        for i in range(1, num_layers):
            prev = channel_sizes[-1][-1]
            new = prev * 2
            channel_sizes.append((prev, new))
        return channel_sizes

    def forward(self, x):
        mu, log_var = self.get_latent_variables(x)
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var.mul(0.5))  # log sqrt(x) = log x^0.5 = 0.5 log x
        epsilon = torch.randn_like(mu)
        z = mu + (epsilon * std)
        return z

    def get_latent_variables(self, x):
        # Encode
        hidden_state = self.encoder(x)
        # Get latent variables
        mu = self.fc_mu(hidden_state)
        log_var = self.fc_log_var(hidden_state)
        return mu, log_var


class VAEGANDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        image_size,
        image_channels=3,
        max_filters=512,
        num_layers=4,
        kernel_size=2,
        stride=2,
        padding=0,
        latent_dim=128,
        small_conv=False,
    ):
        super(VAEGANDecoder, self).__init__()
        if small_conv:
            num_layers += 1
        channel_sizes = self.calculate_channel_sizes(
            image_channels, max_filters, num_layers
        )
        # Decoder
        decoder_layers = nn.ModuleList()
        # Feedforward/Dense Layer to expand our latent dimensions
        decoder_layers.append(nn.Linear(latent_dim, hidden_dim))
        # Unflatten to a shape of (Channels, Height, Width)
        decoder_layers.append(nn.Unflatten(1, (max_filters, image_size, image_size)))
        # Decoder Convolutions
        for i, (out_channels, in_channels) in enumerate(channel_sizes[::-1]):
            if small_conv and i == num_layers - 1:
                # 1x1 Transposed Convolution
                decoder_layers.append(
                    nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            else:
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

    def forward(self, x):
        return self.decoder(x)

    def calculate_channel_sizes(self, image_channels, max_filters, num_layers):
        channel_sizes = [(image_channels, max_filters // np.power(2, num_layers - 1))]
        for i in range(1, num_layers):
            prev = channel_sizes[-1][-1]
            new = prev * 2
            channel_sizes.append((prev, new))
        return channel_sizes


class VAEGANDiscriminator(nn.Module):
    def __init__(
        self,
        image_channels=3,
        num_layers=4,
        max_filters=512,
        kernel_size=2,
        stride=2,
        padding=0,
        input_image_dimensions=96,
    ):
        super(VAEGANDiscriminator, self).__init__()
        channel_sizes = self.calculate_channel_sizes(
            image_channels, max_filters, num_layers
        )
        layers = nn.ModuleList()
        # Convolutional Blocks
        for i, (in_channels, out_channels) in enumerate(channel_sizes):
            # Convolutional Layer
            layers.append(
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
            layers.append(nn.BatchNorm2d(out_channels))
            # Activation Function
            layers.append(nn.ReLU())
        # Flatten Encoder Output
        self.conv = nn.Sequential(*layers)

        # Get Hidden Dimension Size
        hidden_dim, _ = self.get_flattened_size(input_image_dimensions)

        # Fully Connected Block
        fc_layers = nn.ModuleList()
        fc_layers.append(nn.Flatten())
        fc_layers.append(nn.Linear(hidden_dim, 1))
        fc_layers.append(nn.Sigmoid())
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        lth_output = self.conv(x)
        output = self.fc(lth_output)
        return output, lth_output

    def calculate_layer_size(self, input_size, kernel_size, stride, padding=0):
        numerator = input_size - kernel_size + (2 * padding)
        denominator = stride
        return (numerator // denominator) + 1

    def get_flattened_size(self, image_size):
        for layer in self.conv:
            if "Conv2d" in str(layer):
                kernel_size = layer.kernel_size[0]
                stride = layer.stride[0]
                padding = layer.padding[0]
                filters = layer.out_channels
                image_size = self.calculate_layer_size(
                    image_size, kernel_size, stride, padding
                )
        return filters * image_size * image_size, image_size

    def calculate_channel_sizes(self, image_channels, max_filters, num_layers):
        channel_sizes = [(image_channels, max_filters // np.power(2, num_layers - 1))]
        for i in range(1, num_layers):
            prev = channel_sizes[-1][-1]
            new = prev * 2
            channel_sizes.append((prev, new))
        return channel_sizes
