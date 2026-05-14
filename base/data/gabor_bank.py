# gabor_bank.py
import math
import torch

def make_gabor_kernel(ks: int, sigma: float, theta: float, lam: float, gamma: float, psi: float, device="cpu"):
    """
    Returns (ks, ks) Gabor kernel, zero-mean, L2-normalized.
    theta in radians.
    """
    half = ks // 2
    y, x = torch.meshgrid(
        torch.arange(-half, half + 1, device=device, dtype=torch.float32),
        torch.arange(-half, half + 1, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # rotate coords
    ct, st = math.cos(theta), math.sin(theta)
    x_theta = x * ct + y * st
    y_theta = -x * st + y * ct

    gb = torch.exp(-(x_theta**2 + (gamma**2) * y_theta**2) / (2 * sigma**2)) \
         * torch.cos((2 * math.pi * x_theta / lam) + psi)

    gb = gb - gb.mean()
    gb = gb / (gb.norm() + 1e-12)
    return gb

def build_gabor_bank(
    kernel_size=9,
    thetas_deg=(0, 45, 90, 135),
    sigma=2.0,
    lam=4.0,
    gamma=0.5,
    psi=0.0,
    device="cpu",
):
    """
    Returns weight tensor for conv2d: (K, 1, ks, ks)
    """
    thetas = [math.radians(t) for t in thetas_deg]
    kernels = []
    for th in thetas:
        k = make_gabor_kernel(kernel_size, sigma, th, lam, gamma, psi, device=device)
        kernels.append(k)
    W = torch.stack(kernels, dim=0).unsqueeze(1)  # (K,1,ks,ks)
    return W
