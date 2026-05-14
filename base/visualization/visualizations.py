import torch
import matplotlib.pyplot as plt
import time
import numpy as np

def plot_raster(E_spikes, I_spikes, title_excitatory, title_inhibitory, g, eta, CV, rho_mean, rate):
    

        sE = E_spikes.squeeze(1)
        sI = I_spikes.squeeze(1)

        tE, nE = torch.where(sE > 0)
        tI, nI = torch.where(sI > 0)

        fig, ax = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
        ax[0].scatter(tE.cpu(), nE.cpu(), color="slategrey", marker=".", s=10, edgecolors="none", alpha=0.8)
        ax[0].set_title(title_excitatory)
        ax[0].set_ylabel("Neuron idx")
        ax[1].scatter(tI.cpu(), nI.cpu(), color="slategrey", marker=".", s=10, edgecolors="none", alpha=0.8)
        ax[1].set_title(title_inhibitory)
        ax[1].set_xlabel("Time (timestep) ms")
        ax[1].set_ylabel("Neuron idx")
        ax[0].grid(True, linestyle="--", alpha=0.6)
        ax[1].grid(True, linestyle="--", alpha=0.6)

        # Add label with variable values
        label_text = (f"g = {g:.2f}\n"
                      f"eta = {eta:.2f}\n"
                      f"CV = {CV:.2f}\n"
                      f"rho = {rho_mean:.2f}\n"
                      f"rate = {rate:.2f} Hz")
        # place label in the upper-right corner of the first subplot
        ax[0].text(
            0.99, 0.95, label_text, 
            transform=ax[0].transAxes, 
            fontsize=10, 
            verticalalignment='top',
            horizontalalignment='right', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6)
        )

        plt.tight_layout()
        #plt.show(block=True)
        plt.savefig(f"/Users/daquiry/Home/Masters_repo/Structured/BindsNet/results/raster_plot_{time.time()}.png")
        plt.close()

def plot_rate_distribution(time, E_spike_counts, I_spike_counts, title_excitatory, title_inhibitory):
        neuron_rates_E = E_spike_counts / (time / 1000.0) #Hz to spikes/sec
        avr_rate_E = neuron_rates_E.mean()
        neuron_rates_I = I_spike_counts / (time / 1000.0) #Hz to spikes/sec
        avr_rate_I = neuron_rates_I.mean()

        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax[0].hist(neuron_rates_E.cpu().numpy(), bins=100, color='skyblue', alpha=0.7)
        ax[0].set_title(title_excitatory)
        ax[0].axvline(avr_rate_E, color='salmon', linestyle='--', linewidth=2, label=f"Average rate: {avr_rate_E:.2f} Hz")
        ax[0].legend()
        ax[1].hist(neuron_rates_I.cpu().numpy(), bins=100, color='salmon', alpha=0.7)
        ax[1].set_title(title_inhibitory)
        ax[1].axvline(avr_rate_I, color='skyblue', linestyle='--', linewidth=2, label=f"Average rate: {avr_rate_I:.2f} Hz")
        ax[1].legend()
        plt.xlabel('Firing rate (spikes/s)')
        plt.ylabel('Number of neurons')
        plt.tight_layout()
        plt.show(block=True)
        plt.close()

def plot_spike_distribution(E_spike_counts, I_spike_counts, title):
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharex=True)


        E_flat = E_spike_counts.flatten()
        I_flat = I_spike_counts.flatten()

        for i, val in enumerate(E_flat):
            num_spikes = int(val)
            ax.vlines(i, 0, num_spikes, color='skyblue', alpha= 0.8)
        for i, val in enumerate(I_flat):
            i = len(E_flat) + i
            num_spikes = int(val)
            ax.vlines(i, 0, num_spikes, color='salmon', alpha= 0.8)

        ax.set_title(title)
        ax.set_xlabel("Neuron index")
        ax.set_ylabel("Number of spikes")
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        
        plt.show(block=True)
        plt.close()

def self_tuning_plot(CV_list, rho_mean_list, rate_list, g_list, eta_list):
    CV_arr   = np.array(CV_list)
    rho_arr  = np.array(rho_mean_list)
    rate_arr = np.array(rate_list)
    g_arr    = np.array(g_list)
    eta_arr  = np.array(eta_list)

    print(f"Average CV:       {CV_arr.mean():.4f}")
    print(f"Average rho_mean: {rho_arr.mean():.4f}")

    steps = np.arange(len(CV_arr))

    # Self-tuning thresholds (must match _self_tune)
    CV_low,  CV_high  = 0.6,  1.6
    rho_high          = 0.1
    rate_low, rate_high = 2.0, 80.0

    # Muted academic colour palette
    C = {
        'cv':   '#4878D0',
        'rho':  '#956CB4',
        'rate': '#EE854A',
        'g':    '#D65F5F',
        'eta':  '#6ACC65',
    }

    def _smooth(arr, w=15):
        """Rolling mean; returns (x_offset, smoothed_y)."""
        if len(arr) < w:
            return np.arange(len(arr)), arr.copy()
        kernel = np.ones(w) / w
        return np.arange(w - 1, len(arr)), np.convolve(arr, kernel, mode='valid')

    def _panel(ax, x, y, color, ylabel, title, smooth_w=15,
               hlines=None, hspan=None):
        """Draw one trace panel with raw data, smoothed overlay, and optional reference lines."""
        # Raw trace (thin, transparent)
        ax.plot(x, y, color=color, alpha=0.2, linewidth=0.8)
        # Smoothed trace
        sx, sy = _smooth(y, smooth_w)
        ax.plot(sx, sy, color=color, linewidth=2.0, label='Rolling mean')
        # Target band
        if hspan is not None:
            lo, hi = hspan
            ax.axhspan(lo, hi, color=color, alpha=0.10, zorder=0)
            ax.axhline(lo, color=color, linestyle='--', linewidth=0.9, alpha=0.55)
            ax.axhline(hi, color=color, linestyle='--', linewidth=0.9, alpha=0.55,
                       label=f'Target [{lo}, {hi}]')
        # Individual threshold lines
        if hlines is not None:
            for val, lbl in hlines:
                ax.axhline(val, color=color, linestyle='--', linewidth=0.9,
                           alpha=0.55, label=lbl)
        # Cosmetics
        ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel('Sample', fontsize=10)
        ax.tick_params(labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
        if hspan is not None or hlines is not None:
            ax.legend(fontsize=8, framealpha=0.6, loc='upper right')

    fig, axs = plt.subplots(3, 2, figsize=(13, 11))

    _panel(axs[0, 0], steps, CV_arr,   C['cv'],   r'CV',
           r'Coefficient of Variation (CV)',
           hspan=(CV_low, CV_high))

    _panel(axs[0, 1], steps, rho_arr,  C['rho'],  r'$\rho_\mathrm{mean}$',
           r'Population Synchrony ($\rho_\mathrm{mean}$)',
           hlines=[(rho_high, fr'$\rho_\mathrm{{high}}={rho_high}$')])

    _panel(axs[1, 0], steps, rate_arr, C['rate'], r'Rate (Hz)',
           r'Mean Firing Rate',
           hlines=[(rate_low,  fr'$r_\mathrm{{low}}={rate_low}$ Hz'),
                   (rate_high, fr'$r_\mathrm{{high}}={rate_high}$ Hz')])

    _panel(axs[1, 1], steps, g_arr,    C['g'],    r'$g$',
           r'Inhibitory Strength ($g$)')

    _panel(axs[2, 0], steps, eta_arr,  C['eta'],  r'$\eta$',
           r'External Drive ($\eta$)')

    # Phase portrait: g vs η trajectory
    ax_phase = axs[2, 1]
    n = len(g_arr)
    colors_grad = plt.cm.plasma(np.linspace(0.1, 0.9, n))
    for i in range(n - 1):
        ax_phase.plot(eta_arr[i:i+2], g_arr[i:i+2], color=colors_grad[i], linewidth=1.2)
    sc = ax_phase.scatter(eta_arr, g_arr, c=np.linspace(0, 1, n),
                          cmap='plasma', s=12, zorder=3)
    ax_phase.scatter(eta_arr[0],  g_arr[0],  marker='o', s=60,
                     color='black', zorder=5, label='Start')
    ax_phase.scatter(eta_arr[-1], g_arr[-1], marker='*', s=80,
                     color='black', zorder=5, label='End')
    fig.colorbar(sc, ax=ax_phase, label='Normalised time', pad=0.02)
    ax_phase.set_xlabel(r'$\eta$', fontsize=10)
    ax_phase.set_ylabel(r'$g$',   fontsize=10)
    ax_phase.set_title(r'Parameter Trajectory ($\eta$, $g$)',
                       fontsize=11, fontweight='bold', pad=6)
    ax_phase.tick_params(labelsize=9)
    ax_phase.spines['top'].set_visible(False)
    ax_phase.spines['right'].set_visible(False)
    ax_phase.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
    ax_phase.legend(fontsize=8, framealpha=0.6)

    fig.suptitle('Self-Tuning Dynamics', fontsize=13, fontweight='bold', y=1.005)
    plt.tight_layout()
    plt.savefig(f"/Users/daquiry/Home/Masters_repo/Structured/BindsNet/results/self_tuning_dynamics_{time.time()}.png", dpi=150, bbox_inches='tight')
    plt.show(block=True)