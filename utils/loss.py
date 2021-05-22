import torch
import torch.nn as nn


def VAELoss(x, reconstructed_x, mu, log_var, use_sum=True):
    if use_sum:
        reconstruction_loss = nn.functional.mse_loss(
            reconstructed_x, x, reduction="sum"
        )
        KL_d = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return reconstruction_loss + KL_d, reconstruction_loss, KL_d
    else:
        reconstruction_loss = nn.functional.mse_loss(
            reconstructed_x, x, reduction="mean"
        )
        KL_d = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return reconstruction_loss + KL_d, reconstruction_loss, KL_d


def VAEWeightedLoss(
    x, reconstructed_x, mu, log_var, reconstruction_weight=1, kl_weight=1, use_sum=True
):
    if use_sum:
        reconstruction_loss = nn.functional.mse_loss(
            reconstructed_x, x, reduction="sum"
        )
        KL_d = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    else:
        reconstruction_loss = nn.functional.mse_loss(
            reconstructed_x, x, reduction="mean"
        )
        KL_d = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    weighted_loss = (reconstruction_weight * reconstruction_loss) + (kl_weight * KL_d)
    true_loss = reconstruction_loss + KL_d
    return true_loss, weighted_loss, reconstruction_loss, KL_d