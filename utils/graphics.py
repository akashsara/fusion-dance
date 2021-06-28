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
    if mode == "vae":
        for i, label in enumerate(
            ["Total Loss", "Reconstruction Loss", "KL-Divergence"]
        ):
            train_loss = [x[i] for x in all_train_loss]
            val_loss = [x[i] for x in all_val_loss]
            train_label = label = f"Train {label}"
            val_label = f"Validation {label}"
            label = label.lower().replace(" ", "_")
            output_path = os.path.join(loss_output_path, f"{label}.jpg")
            plot_and_save_loss(
                train_loss, train_label, val_loss, val_label, output_path
            )
    elif mode == "vaegan":
        for i, label in enumerate(
            ["Total Loss", "Encoder Loss", "Decoder Loss", "Discriminator Loss"]
        ):
            train_loss = [x[i] for x in all_train_loss]
            val_loss = [x[i] for x in all_val_loss]
            train_label = f"Train {label}"
            val_label = f"Validation {label}"
            label = label.lower().replace(" ", "_")
            output_path = os.path.join(loss_output_path, f"{label}.jpg")
            plot_and_save_loss(
                train_loss, train_label, val_loss, val_label, output_path
            )
    else:
        train_label = "Train Loss"
        val_label = "Validation Loss"
        plot_and_save_loss(
            all_train_loss, train_label, all_val_loss, val_label, loss_output_path
        )


def plot_and_save_loss(train_loss, train_label, val_loss, val_label, output_path):
    plt.figure(figsize=(8, 6), dpi=100)
    ax = plt.subplot()
    plt.plot(train_loss, label=train_label)
    plt.plot(val_loss, label=val_label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)