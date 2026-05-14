import torch
import numpy as np


def calculate_fisher_ratio(pairs):
    """
    Compute the multiclass Fisher ratio from reservoir spike feature vectors.

    J = tr(S_B) / tr(S_W)

    S_B: between-class scatter matrix (how far class centroids are from the global mean)
    S_W: within-class scatter matrix (how spread samples are around their class centroid)

    A higher J indicates better class separability in the reservoir representation.

    Args:
        pairs: list of (feature_tensor, label) as returned by framework.run_stimulation()

    Returns:
        J (float): Fisher ratio. Returns 0.0 if S_W trace is zero.
    """
    features = torch.stack([f for f, _ in pairs])   # (N, D)
    labels   = torch.tensor([l for _, l in pairs])  # (N,)

    classes = labels.unique()
    global_mean = features.mean(dim=0)              # (D,)

    S_W = torch.zeros(features.shape[1])            # diagonal only — trace-efficient
    S_B = torch.zeros(features.shape[1])

    for c in classes:
        mask   = labels == c
        X_c    = features[mask]                     # (N_c, D)
        mu_c   = X_c.mean(dim=0)                    # (D,)
        n_c    = X_c.shape[0]

        # Within-class: variance of samples around their class centroid
        S_W += ((X_c - mu_c) ** 2).sum(dim=0)

        # Between-class: weighted squared distance of class centroid from global mean
        diff = mu_c - global_mean
        S_B += n_c * (diff ** 2)

    trace_SW = S_W.sum().item()
    trace_SB = S_B.sum().item()

    if trace_SW == 0:
        return 0.0

    return trace_SB / trace_SW

def calculate_CV(spikes, dt):
    # spikes shape: (T, 1, N) -> squeeze to (T, N)
        spikes = spikes.squeeze(1).cpu()
        T, N = spikes.shape
        
        cv_list = []
        
        for i in range(N):
            # Find indices where spike == 1
            spike_times = torch.where(spikes[:, i] == 1)[0].float()
            
            if len(spike_times) < 3:
                # Need at least 3 spikes to get 2 intervals for a standard deviation
                continue
                
            # Calculate ISIs in milliseconds
            intervals = (spike_times[1:] - spike_times[:-1]) * dt
            
            # Brunel CV calculation
            cv_i = torch.std(intervals) / torch.mean(intervals)
            cv_list.append(cv_i.item())
        
        return np.mean(cv_list) if cv_list else 0.0

def calculate_rho_mean(spikes: torch.Tensor, dt, bin_ms: float = 5.0) -> float:
        # spikes -> (T, N)
        spikes = spikes.squeeze(1) if spikes.dim() == 3 else spikes
        spikes = spikes.detach().float().cpu()

        T, N = spikes.shape
        if N < 2 or T < 2:
            return 0.0

        # bin in time
        bin_steps = max(1, int(round(bin_ms / dt)))
        n_bins = T // bin_steps
        if n_bins < 2:
            return 0.0

        x = spikes[: n_bins * bin_steps].reshape(n_bins, bin_steps, N).sum(dim=1)  # (n_bins, N)

        # population activity
        r_t = x.mean(dim=1)  # (n_bins,)

        var_r = torch.var(r_t, unbiased=False)
        var_i = torch.var(x, dim=0, unbiased=False)
        mean_var_i = torch.mean(var_i)

        if mean_var_i.item() <= 1e-12:
            return 0.0

        return float((var_r / mean_var_i).item())


def calculate_rate(spikes_counts, time):
    return (spikes_counts / (time / 1000.0)).mean() #Hz to spikes/sec