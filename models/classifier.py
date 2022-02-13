import torch
import torch.nn as nn
import numpy as np


class CNNMultiClassClassifier(nn.Module):
    def __init__(
        self,
        num_layers,
        max_filters,
        num_output_classes,
        input_dimension=64,
        input_channels=1,
    ):
        super(CNNMultiClassClassifier, self).__init__()
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
            # Setup next layer channel info
            if i != num_layers - 1:
                in_channels = out_channels
                out_channels = out_channels * 2
            # Update image dimension in the model
            input_dimension = input_dimension / 2
        layers.append(nn.Flatten())
        layers.append(
            nn.Linear(
                in_features=out_channels * (int(input_dimension) ** 2),
                out_features=num_output_classes,
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ANNMultiClassClassifier(nn.Module):
    def __init__(self, num_layers, input_dimension, start_filters, num_output_classes):
        super(ANNMultiClassClassifier, self).__init__()
        layers = nn.ModuleList()
        # Conv Layers
        for i in range(num_layers):
            if i == num_layers - 1:
                layers.append(
                    nn.Linear(
                        in_features=input_dimension,
                        out_features=num_output_classes,
                    )
                )
            else:
                layers.append(
                    nn.Linear(
                        in_features=input_dimension,
                        out_features=start_filters,
                    )
                )
                layers.append(nn.ReLU(True))
            input_dimension = start_filters
            start_filters = start_filters * 2

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ANNMultiClassClassifierV2(nn.Module):
    def __init__(self, num_layers, input_dimension, hidden_size, num_output_classes):
        super(ANNMultiClassClassifier, self).__init__()
        layers = nn.ModuleList()
        layers.append(
            nn.Linear(
                in_features=input_dimension,
                out_features=hidden_size,
            )
        )
        layers.append(nn.Dropout(p=0.5, inplace=True))
        layers.append(nn.ReLU(True))
        layers.append(
            nn.Linear(
                in_features=hidden_size,
                out_features=num_output_classes,
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)