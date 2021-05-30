import torch
import torch.nn as nn


def mse_loss(reconstructed_x, x, use_sum):
    if use_sum:
        return nn.functional.mse_loss(reconstructed_x, x, reduction="sum")
    else:
        return nn.functional.mse_loss(reconstructed_x, x, reduction="mean")


def kl_divergence(mu, log_var, use_sum):
    if use_sum:
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    else:
        return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())


def mse_ssim_loss(
    reconstructed_x, x, use_sum, ssim_module=None, mse_weight=1, ssim_weight=1
):
    mse = mse_weight * mse_loss(reconstructed_x, x, use_sum)
    if ssim_module:
        # ssim gives a score from 0-1 where 1 is the highest.
        # So we do 1 - ssim in order to minimize it.
        ssim = ssim_weight * (1 - ssim_module(reconstructed_x, x))
    else:
        ssim = 0
    return mse + ssim, mse, ssim


def VAE_loss(
    reconstructed_x,
    x,
    mu,
    log_var,
    use_sum=True,
    ssim_module=None,
    mse_weight=1,
    ssim_weight=1,
    reconstruction_weight=1,
    kl_weight=1,
):
    mse_ssim, mse, ssim = mse_ssim_loss(
        reconstructed_x,
        x,
        use_sum,
        ssim_module=ssim_module,
        mse_weight=mse_weight,
        ssim_weight=ssim_weight,
    )
    KL_d = kl_divergence(mu, log_var, use_sum)
    weighted_loss = (reconstruction_weight * mse_ssim) + (kl_weight * KL_d)
    return weighted_loss, (mse, ssim, KL_d)