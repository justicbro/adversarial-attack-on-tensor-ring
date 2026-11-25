"""Quality metrics of a reconstruction X_hat with respect to the clean tensor X."""
import math

import torch


def rse(X, X_hat):
    """Relative standard error ||X - X_hat||_F / ||X||_F."""
    return (torch.norm(X - X_hat) / torch.norm(X)).item()


def psnr(X, X_hat, data_range=1.0):
    """Peak signal-to-noise ratio in dB for data in [0, data_range]."""
    mse = torch.mean((X - X_hat) ** 2).item()
    return float("inf") if mse == 0 else 10.0 * math.log10(data_range ** 2 / mse)
