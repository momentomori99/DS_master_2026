import torch

def sample_per_neuron_param(N, base, rel_std=0.1, min_val=None, max_val=None, device="cpu"):
        """
        Sample per-neuron parameter values around `base`.
        rel_std = 0.1 means std = 10% of base.
        """
        x = torch.normal(mean=float(base), std=float(base)*rel_std, size=(N,), device=device)
        if min_val is not None or max_val is not None:
            lo = -float("inf") if min_val is None else float(min_val)
            hi =  float("inf") if max_val is None else float(max_val)
            x = x.clamp(lo, hi)
        return x