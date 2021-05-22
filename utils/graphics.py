import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import colors
from PIL import Image

from tqdm import tqdm

import os


def make_grid(images, height, width, fig=None, axis=None):
    if fig is None or axis is None:
        fig, axis = plt.subplots(height, width, figsize=(8, 6), dpi=80)
    i, j = 0, 0
    text, images = images
    for num, image in enumerate(images):
        if num == height * width:
            break
        axis[i, j].imshow(np.clip(np.asarray(image.permute(1, 2, 0)), 0, 1))
        if j == width - 1:
            j = 0
            i += 1
        else:
            j += 1
    if type(text) == int:
        text = f"Epoch: {text}"
    fig.suptitle(text, va="baseline")
    plt.tight_layout()
    return fig, axis


def make_animation(make_grid, all_samples, height=4, width=4):
    fig, axis = plt.subplots(height, width, figsize=(8, 6), dpi=80)
    anim = animation.FuncAnimation(
        fig=fig,
        func=make_grid,
        frames=list(enumerate(all_samples)),
        fargs=(height, width, fig, axis),
        interval=100,
        repeat=False,
    )
    return anim


def draw_loss(all_train_loss, all_val_loss, loss_output_path, mode):
    plt.figure(figsize=(8, 6), dpi=100)
    ax = plt.subplot()
    if mode == "vae":
        plt.plot([x[1] for x in all_train_loss], label="Train Reconstruction Loss")
        plt.plot([x[2] for x in all_train_loss], label="Train KL-Divergence")
        plt.plot([x[1] for x in all_val_loss], label="Validation Reconstruction Loss")
        plt.plot([x[2] for x in all_val_loss], label="Validation KL-Divergence")
    else:
        plt.plot([x for x in all_train_loss], label="Train Loss")
        plt.plot([x for x in all_val_loss], label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_output_path)