import torch
import torch.nn as nn
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
        small_conv=False,
    ):
        super(ConvolutionalAE, self).__init__()
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

        # Calculate shape of the flattened image
        hidden_dim, image_size = self.get_flattened_size(
            input_image_dimensions, encoder_layers
        )

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

    def calculate_layer_size(self, input_size, kernel_size, stride, padding=0):
        numerator = input_size - kernel_size + (2 * padding)
        denominator = stride
        return (numerator // denominator) + 1

    def get_flattened_size(self, image_size, encoder_layers):
        for layer in encoder_layers:
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

    def forward(self, x, return_logits=False):
        # Encode
        hidden_state = self.encoder(x)
        # Decode
        if return_logits:
            for layer in self.decoder[:-1]:
                hidden_state = layer(hidden_state)
            logits = hidden_state
            reconstructed = self.decoder[-1](hidden_state)
            return reconstructed, logits
        else:
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
        small_conv=False,
    ):
        super(ConvolutionalVAE, self).__init__()
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

    def forward(self, x, return_logits=False):
        mu, log_var = self.get_latent_variables(x)
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        # Decode
        if return_logits:
            for layer in self.decoder[:-1]:
                z = layer(z)
            logits = z
            reconstructed = self.decoder[-1](z)
            return reconstructed, mu, log_var, logits
        else:
            reconstructed = self.decoder(z)
            return reconstructed, mu, log_var

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


class FusionVAE(nn.Module):
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
        super(FusionVAE, self).__init__()
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

    def forward(self, x1, x2, return_logits=False):
        mu1, log_var1 = self.get_latent_variables(x1)
        mu2, log_var2 = self.get_latent_variables(x2)
        mu = (0.4 * mu1) + (0.6 * mu2)
        log_var = (0.4 * log_var1) + (0.6 * log_var2)
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        # Decode
        if return_logits:
            for layer in self.decoder[:-1]:
                z = layer(z)
            logits = z
            reconstructed = self.decoder[-1](z)
            return reconstructed, mu, log_var, logits
        else:
            reconstructed = self.decoder(z)
            return reconstructed, mu, log_var

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


class FusionAE(nn.Module):
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
        super(FusionAE, self).__init__()
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
        # Calculate shape of the flattened image
        hidden_dim, image_size = self.get_flattened_size(
            input_image_dimensions, encoder_layers
        )

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

    def calculate_layer_size(self, input_size, kernel_size, stride, padding=0):
        numerator = input_size - kernel_size + (2 * padding)
        denominator = stride
        return (numerator // denominator) + 1

    def get_flattened_size(self, image_size, encoder_layers):
        for layer in encoder_layers:
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

    def forward(self, x1, x2, return_logits=False):
        # Encode
        hidden_state1 = self.encoder(x1)
        hidden_state2 = self.encoder(x2)
        hidden_state = (0.4 * hidden_state1) + (0.4 * hidden_state2)
        # Decode
        if return_logits:
            for layer in self.decoder[:-1]:
                hidden_state = layer(hidden_state)
            logits = hidden_state
            reconstructed = self.decoder[-1](hidden_state)
            return reconstructed, logits
        else:
            return self.decoder(hidden_state)


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


# https://nbviewer.jupyter.org/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
class VectorQuantizerEMA(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost, decay=0.0, epsilon=1e-5
    ):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        # Euclidean Distance = Sqrt(Sum(Square(Differences)))
        # We ignore sqrt because we're taking the nearest neighbor
        # Which doesn't change when we take sqrt, so we have that compute
        # Since we're working with multi-dimensional matrices
        # We can get rid of the sum as this is a vectorized operation
        # So we compute Square(Differences)
        # I.E. (A - B)^2 = A^2 + B^2 - 2AB
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        # Get Nearest Neighbors
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # Create a one hot encoded matrix where 1 indicates the nearest neighbor
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            # N_t = N_(t-1) * gamma + n_t * (1 - gamma)
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            # m_t = m_(t-1) * gamma + z_t * (1 - gamma)
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            # e = N_t / m_t
            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        # Loss
        e_latent_loss = nn.functional.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return (
            loss,
            quantized.permute(0, 3, 1, 2).contiguous(),
            perplexity,
            encoding_indices,
        )

    def quantize_encoding_indices(self, encoding_indices, target_shape, device):
        # For use in inference/fusion generation
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=device
        )
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self._embedding.weight).view(target_shape)
        return quantized.permute(0, 3, 1, 2).contiguous()


class VQVAE(nn.Module):
    def __init__(
        self,
        image_channels=3,
        num_layers=4,
        kernel_size=2,
        stride=2,
        padding=0,
        input_image_dimensions=96,
        small_conv=False,
        embedding_dim=64,
        max_filters=512,
        use_max_filters=False,
        num_embeddings=512,
        commitment_cost=0.25,
        decay=0.99,
    ):
        super(VQVAE, self).__init__()
        if small_conv:
            num_layers += 1
        if not use_max_filters:
            max_filters = embedding_dim
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

        # Final Conv Layer to ensure we have embedding_dim channels
        if use_max_filters:
            encoder_layers.append(
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=embedding_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
        # Make Encoder
        self.encoder = nn.Sequential(*encoder_layers)

        # Vector Quantizer
        self.vq_vae = VectorQuantizerEMA(
            num_embeddings, embedding_dim, commitment_cost, decay
        )

        # Decoder
        decoder_layers = nn.ModuleList()
        # Initial Conv Layer to change the channels back to the required number
        if use_max_filters:
            decoder_layers.append(
                nn.Conv2d(
                    in_channels=embedding_dim,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
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

    def calculate_channel_sizes(self, image_channels, max_filters, num_layers):
        channel_sizes = [(image_channels, max_filters // np.power(2, num_layers - 1))]
        for i in range(1, num_layers):
            prev = channel_sizes[-1][-1]
            new = prev * 2
            channel_sizes.append((prev, new))
        return channel_sizes

    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)
        # VQ-VAE
        loss, quantized, perplexity, encodings = self.vq_vae(encoded)
        # Decoder
        reconstructed = self.decoder(quantized)
        return loss, reconstructed, perplexity, encodings

    def quantize_and_decode(self, x, target_shape, device):
        quantized = self.vq_vae.quantize_encoding_indices(x, target_shape, device)
        return self.decoder(quantized)


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


class CNN_RNN(nn.Module):
    def __init__(
        self,
        num_classes,
        input_image_size=64,
        input_channels=3,
        cnn_output_channels=512,
        cnn_blocks=4,
        rnn_hidden_size=512,
        rnn_bidirectional=False,
        rnn_type="LSTM",
    ):
        super(CNN_RNN, self).__init__()
        # Save Certain Variables
        self.rnn_hidden_size = rnn_hidden_size
        self.num_classes = num_classes
        self.rnn_type = rnn_type.lower()
        self.rnn_bidirectional = rnn_bidirectional
        ## Encoder
        # CNN
        layers = nn.ModuleList()
        channel_sizes = self.calculate_channel_sizes(
            input_channels, cnn_output_channels, cnn_blocks
        )
        kernel_size = 1
        stride = 1
        for i, (in_channels, out_channels) in enumerate(channel_sizes):
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(True))
            if i == 0:
                kernel_size = 2
                stride = 2
        layers.append(nn.Flatten())
        self.encoder_cnn = nn.Sequential(*layers)
        # FC
        image_size = input_image_size // np.power(2, cnn_blocks - 1)
        cnn_out_features = image_size * image_size * out_channels
        self.encoder_fc = nn.Linear(cnn_out_features, num_classes)
        ## Decoder
        if self.rnn_type == "lstm":
            # LSTM
            self.decoder_rnn = nn.LSTM(
                input_size=num_classes,
                hidden_size=rnn_hidden_size,
                bidirectional=rnn_bidirectional,
                batch_first=True,
            )
        else:
            # GRU
            self.decoder_rnn = nn.GRU(
                input_size=num_classes,
                hidden_size=rnn_hidden_size,
                bidirectional=rnn_bidirectional,
                batch_first=True,
            )
        # FC
        rnn_out_features = (
            rnn_hidden_size * 2 if self.rnn_bidirectional else rnn_hidden_size
        )
        self.decoder_fc = nn.Linear(rnn_out_features, num_classes)

    def forward(self):
        pass

    def encode(self, x):
        image_latent = self.encoder_cnn(x)
        encoded = self.encoder_fc(image_latent)
        return encoded.view(-1, 1, self.num_classes)

    def decode(self, decoder_input, decoder_hidden):
        decoder_output, decoder_hidden = self.decoder_rnn(decoder_input, decoder_hidden)
        decoder_output = self.decoder_fc(decoder_output)
        return decoder_output, decoder_hidden

    def calculate_channel_sizes(self, image_channels, max_filters, num_layers):
        channel_sizes = [(image_channels, max_filters // np.power(2, num_layers - 1))]
        for i in range(1, num_layers):
            prev = channel_sizes[-1][-1]
            new = prev * 2
            channel_sizes.append((prev, new))
        return channel_sizes

    def init_hidden_state(self, batch_size, device):
        first_dim = 1
        if self.rnn_bidirectional:
            first_dim = 2
        if self.rnn_type == "lstm":
            return (
                torch.zeros(first_dim, batch_size, self.rnn_hidden_size, device=device),
                torch.zeros(first_dim, batch_size, self.rnn_hidden_size, device=device),
            )
        else:
            return torch.zeros(
                first_dim, batch_size, self.rnn_hidden_size, device=device
            )


class RNNBlock(nn.Module):
    def __init__(self, num_classes, hidden_size, bidirectional, rnn_type):
        super(RNNBlock, self).__init__()
        rnn_out_features = hidden_size * 2 if bidirectional else hidden_size
        # LSTM
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=num_classes,
                hidden_size=hidden_size,
                bidirectional=bidirectional,
                batch_first=True,
            )
        # GRU
        else:
            self.rnn = nn.GRU(
                input_size=num_classes,
                hidden_size=hidden_size,
                bidirectional=bidirectional,
                batch_first=True,
            )
        # FC
        self.fc = nn.Linear(rnn_out_features, num_classes)

    def forward(self, decoder_input, decoder_hidden):
        decoder_output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)
        return self.fc(decoder_output), decoder_hidden


class CNN_MultiRNN(nn.Module):
    def __init__(
        self,
        num_classes,
        num_rnns,
        input_image_size=64,
        input_channels=3,
        cnn_output_channels=512,
        cnn_blocks=4,
        rnn_hidden_size=512,
        rnn_bidirectional=False,
        rnn_type="LSTM",
    ):
        super(CNN_MultiRNN, self).__init__()
        rnn_type = rnn_type.lower()
        ## Encoder
        # CNN
        layers = nn.ModuleList()
        channel_sizes = self.calculate_channel_sizes(
            input_channels, cnn_output_channels, cnn_blocks
        )
        kernel_size = 1
        stride = 1
        for i, (in_channels, out_channels) in enumerate(channel_sizes):
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(True))
            if i == 0:
                kernel_size = 2
                stride = 2
        layers.append(nn.Flatten())
        self.encoder_cnn = nn.Sequential(*layers)
        # FC
        image_size = input_image_size // np.power(2, cnn_blocks - 1)
        cnn_out_features = image_size * image_size * out_channels
        self.encoder_fc = nn.Linear(cnn_out_features, num_classes)
        ## Decoder
        self.decoder_rnns = nn.ModuleList()
        for i in range(num_rnns):
            self.decoder_rnns.append(
                RNNBlock(num_classes, rnn_hidden_size, rnn_bidirectional, rnn_type)
            )
        ## Save Certain Variables
        self.rnn_hidden_size = rnn_hidden_size
        self.num_classes = num_classes
        self.rnn_type = rnn_type.lower()
        self.num_rnns = num_rnns
        self.rnn_bidirectional = rnn_bidirectional

    def forward(self, x):
        # Only gives the Encoder Output
        image_latent = self.encoder_cnn(x)
        encoded = self.encoder_fc(image_latent)
        return encoded.view(-1, 1, self.num_classes)

    def decode(self, decoder_inputs, decoder_hiddens):
        decoder_outputs = torch.zeros_like(decoder_inputs)
        for i, decoder_rnn in enumerate(self.decoder_rnns):
            decoder_outputs[i], decoder_hiddens[i] = decoder_rnn(
                decoder_inputs[i], decoder_hiddens[i]
            )
        return decoder_outputs, decoder_hiddens

    def calculate_channel_sizes(self, image_channels, max_filters, num_layers):
        channel_sizes = [(image_channels, max_filters // np.power(2, num_layers - 1))]
        for i in range(1, num_layers):
            prev = channel_sizes[-1][-1]
            new = prev * 2
            channel_sizes.append((prev, new))
        return channel_sizes

    def init_hidden_state(self, batch_size, device):
        first_dim = 1
        if self.rnn_bidirectional:
            first_dim = 2
        if self.rnn_type == "lstm":
            return (
                torch.zeros(first_dim, batch_size, self.rnn_hidden_size, device=device),
                torch.zeros(first_dim, batch_size, self.rnn_hidden_size, device=device),
            )
        return torch.zeros(first_dim, batch_size, self.rnn_hidden_size, device=device)


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


def get_freezable_layers(model):
    # Freeze Conv Layers
    freezable_layers = []
    for layer in model.encoder:
        if "Linear" not in str(layer):
            freezable_layers.append(layer)
    for layer in model.decoder:
        if "Linear" not in str(layer):
            freezable_layers.append(layer)
    return freezable_layers


def toggle_layer_freezing(layers, trainable):
    for layer in layers:
        layer.requires_grad_(trainable)


def set_learning_rate(optimizer, learning_rate):
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return optimizer
