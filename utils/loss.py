import torch
import torch.nn as nn
import pytorch_msssim


def ssim_loss(x, reconstructed_x):
    return pytorch_msssim.ssim(
        reconstructed_x,
        x,
        data_range=1.0,
        win_size=11,
        win_sigma=1.5,
        K=(0.01, 0.03),
    )


def mse_loss(x, reconstructed_x, use_sum):
    if use_sum:
        return nn.functional.mse_loss(reconstructed_x, x, reduction="sum")
    else:
        return nn.functional.mse_loss(reconstructed_x, x, reduction="mean")


def kl_divergence(mu, log_var, use_sum):
    if use_sum:
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    else:
        return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())


def VAE_loss(x, reconstructed_x, mu, log_var, use_sum=True):
    mse = mse_loss(x, reconstructed_x, use_sum)
    KL_d = kl_divergence(mu, log_var, use_sum)
    return mse + KL_d, mse, KL_d


def VAE_loss_with_ssim(x, reconstructed_x, mu, log_var, use_sum=True):
    mse = mse_loss(x, reconstructed_x, use_sum)
    ssim = ssim_loss(x, reconstructed_x)
    KL_d = kl_divergence(mu, log_var, use_sum)
    return mse + ssim + KL_d, mse, ssim, KL_d


def VAE_weighted_loss(
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