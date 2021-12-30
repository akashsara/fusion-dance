import torch
import torch.nn as nn
import numpy as np

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
            return reconstructed, mu, log_var, None

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


class VAEPrior(nn.Module):
    def __init__(
        self,
        max_filters=512,
        num_layers=4,
        kernel_size=2,
        stride=2,
        padding=0,
        latent_dim=128,
        input_dimensions=64,
        small_conv=False,
        num_classes=256,
    ):
        super(VAEPrior, self).__init__()
        if small_conv:
            num_layers += 1
        channel_sizes = self.calculate_channel_sizes(
            num_classes, max_filters, num_layers
        )
        self.out_channels = num_classes
        self.out_size = input_dimensions

        # Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=num_classes)

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
        hidden_dim, image_size = self.get_flattened_size(input_dimensions)

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
            decoder_layers.append(nn.ReLU())
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

    def forward(self, x):
        mu, log_var = self.get_latent_variables(x)
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        # Decode
        reconstructed = self.decoder(z)
        reconstructed = reconstructed.view(
            -1, self.out_channels, self.out_size, self.out_size
        )
        return reconstructed, mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var.mul(0.5))  # log sqrt(x) = log x^0.5 = 0.5 log x
        epsilon = torch.randn_like(mu)
        z = mu + (epsilon * std)
        return z

    def get_latent_variables(self, x):
        # Encode
        x = self.embedding(x.flatten(start_dim=1))
        x = x.reshape(-1, self.out_channels, self.out_size, self.out_size)
        hidden_state = self.encoder(x)
        # Get latent variables
        mu = self.fc_mu(hidden_state)
        log_var = self.fc_log_var(hidden_state)
        return mu, log_var
