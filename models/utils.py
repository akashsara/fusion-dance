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
