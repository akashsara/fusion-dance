import torch
import torch.nn as nn
import numpy as np


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
