import torch
import torch.nn as nn
import numpy as np

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
        # Which doesn't change when we take sqrt
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


class FixedVQVAE(nn.Module):
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
        super(FixedVQVAE, self).__init__()
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
                nn.ConvTranspose2d(
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
