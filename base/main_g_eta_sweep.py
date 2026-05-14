from pathlib import Path
import gc

import numpy as np
import matplotlib.pyplot as plt

from framework import Framework
from readout import Readout
from data.input_data_CNN import Data as Data_CNN
from tools.metrics import calculate_fisher_ratio

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "1g_and_eta_sweeps_simple_brunel_intensity_150.txt"


# ── Sweep values ──────────────────────────────────────────────────────────────
G_VALUES   = [2, 3, 4, 5, 6, 7]
ETA_VALUES = [0.6, 0.9, 1.2, 1.5]

name_for_the_run = "Simple Brunel"


# pramaters
n_neurons = 1000
n_epochs = 100
examples_train = 500
examples_test = 125
pca = False

n_components = 60

time = 250
dt   = 1.0
bin_ms = 50        # width of each spike-count bin [ms]

intensity = 64
seed = 54

mnist_input = True
heterogeneity = False
self_tuning = False
spatial = False
convolution = False
log_normal = False
stdp = False
stdp_samples = 100


sigma_input = 1
sigma_network = 1
epsilon = 0.5

data_CNN = Data_CNN(dt=dt, intensity=intensity, kernel_size=9, thetas_deg=(0, 45, 90, 135), convolution=convolution)
train_dataset, test_dataset = data_CNN.load_MNIST()


# results[metric][eta][g]
accuracy_results = {eta: {} for eta in ETA_VALUES}
cv_results       = {eta: {} for eta in ETA_VALUES}
rho_results      = {eta: {} for eta in ETA_VALUES}
rate_results     = {eta: {} for eta in ETA_VALUES}

for g in G_VALUES:
    for eta in ETA_VALUES:
        print(f"Running g={g}, eta={eta} ...")
        framework = Framework(
            n_neurons=n_neurons,
            time=time,
            dt=dt,
            bin_ms=bin_ms,
            seed=seed,
            log_normal=log_normal,
            heterogeneity=heterogeneity,
            mnist_input=mnist_input,
            self_tuning=self_tuning,
            spatial=spatial,
            convolution=convolution,
            g=g,
            eta=eta,
            sigma_input=sigma_input,
            sigma_network=sigma_network,
            epsilon=epsilon,
            intensity=intensity,
            stdp=stdp,
            nu_stdp=(1e-6, 1e-4),
            norm_stdp=None,
        )
        framework.build_network()

        pairs_train, CV_list, rho_mean_list, rate_list, g_list, eta_list = framework.run_stimulation(train_dataset, examples_train)
        pairs_test, *_ = framework.run_stimulation(test_dataset, examples_test)

        feature_dim = pairs_train[0][0].numel()
        readout = Readout(input_size=feature_dim, num_classes=10, seed=seed)
        readout.train_readout(pairs_train, n_epochs=n_epochs)
        acc = readout.test_readout(pairs_test)
        print(f"Accuracy: {acc:.2f}%")

        accuracy_results[eta][g] = acc
        cv_results[eta][g]       = float(np.mean(CV_list))
        rho_results[eta][g]      = float(np.mean(rho_mean_list))
        rate_results[eta][g]     = float(np.mean(rate_list))

        with open(RESULTS_FILE, "a") as f:
            f.write(
                f"{name_for_the_run}  g={g}  eta={eta}"
                f"  accuracy={acc:.2f}%"
                f"  CV={cv_results[eta][g]:.3f}"
                f"  rho={rho_results[eta][g]:.3f}"
                f"  rate={rate_results[eta][g]:.3f}\n"
            )

        del framework, pairs_train, pairs_test, readout
        gc.collect()

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(ETA_VALUES)))

for color, eta in zip(colors, ETA_VALUES):
    accs = [accuracy_results[eta][g] for g in G_VALUES]
    ax.plot(G_VALUES, accs, marker="o", color=color, label=f"eta={eta}")

ax.set_xlabel("g")
ax.set_ylabel("Accuracy (%)")
ax.set_title(f"{name_for_the_run} — g / eta sweep")
ax.legend(title="eta")
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "g_eta_sweep.png", dpi=150)
plt.show()
print(f"Plot saved to {RESULTS_DIR / 'g_eta_sweep.png'}")
