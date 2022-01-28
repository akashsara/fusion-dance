import torch
import torch.nn as nn
import numpy as np


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

def weights_init(m, mean=0.0, std=0.02):
    """
    Custom weight initialization function.
    Sample weights from a normal distribution.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, mean, std)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, std)
        nn.init.constant_(m.bias.data, 0)