from bindsnet.encoding import PoissonEncoder
import numpy as np
import torch


def encode_feature_map(feature_map, time, dt, intensity):
    fm = feature_map.detach().float()

    #normalize to [0, 1]
    m = fm.max()
    if m > 0:
        fm = fm / m
    rates = fm * intensity
    rates = rates.flatten()  # (D, )

    encoder = PoissonEncoder(time=time, dt=dt)
    spikes = encoder(rates) # (T, D)
    return spikes.unsqueeze(1) # (T, 1, D)

    

def spikes_to_binned_counts(E_spikes, bin_ms, dt, time):
    """
    Converts the spikes observed in the network to binned counts.
    """
    s = E_spikes.squeeze(1) if E_spikes.dim() == 3 else E_spikes
    s = s.detach().cpu().numpy().astype(int)

    bin_steps = int(round(bin_ms / dt))
    N_bins = time // bin_steps

    T, N = s.shape
    trim_T = N_bins * bin_steps
    s = s[:trim_T]  # Now shape (N_bins * bin_steps, N)
    s_binned = s.reshape(N_bins, bin_steps, N)
    binned_counts = s_binned.sum(axis=1)

    return torch.tensor(binned_counts, dtype=torch.float32)