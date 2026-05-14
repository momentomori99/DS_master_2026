import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection

# ============================================================
#  Parameters
# ============================================================
seed             = 41
n_neurons        = 500
n_epochs         = 100
examples_train   = 500
examples_test    = 500
time             = 250    # simulation time per sample [ms]
dt               = 1.0    # timestep [ms]
intensity        = 64.0   # input encoding intensity
gpu              = False

# Five bin counts to sweep. All must divide `time` evenly.
# time=250 → valid divisors include: 1, 2, 5, 10, 25, 50, 125, 250
BIN_COUNTS = [25, 50, 75, 125, 250]
#BIN_COUNTS = [75, 125, 250]
#BIN_COUNTS = [1, 5, 10, 15, 20]
#BIN_COUNTS = [250]
# Number of independent readout runs per bin count (for mean ± std estimate).
# Each run uses a different seed: seed, seed+1, ..., seed+N_RUNS-1.
N_RUNS = 10
# ============================================================

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False
torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device =", device)

# ── Build reservoir ──────────────────────────────────────────
network = Network(dt=dt)
inpt = Input(784, shape=(1, 28, 28))
network.add_layer(inpt, name="I")
output = LIFNodes(n_neurons, thresh=-52 + np.random.randn(n_neurons).astype(float))
network.add_layer(output, name="O")
C1 = Connection(source=inpt, target=output, w=0.5 * torch.randn(inpt.n, output.n))
C2 = Connection(source=output, target=output, w=0.5 * torch.randn(output.n, output.n))
network.add_connection(C1, source="I", target="O")
network.add_connection(C2, source="O", target="O")

spikes = {}
for l in network.layers:
    spikes[l] = Monitor(network.layers[l], ["s"], time=time, device=device)
    network.add_monitor(spikes[l], name="%s_spikes" % l)

if gpu:
    network.to("cuda")

# ── Load MNIST (separate train / test splits) ────────────────
tf = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
)

train_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    train=True,
    transform=tf,
)

test_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    train=False,
    transform=tf,
)

def collect_reservoir_responses(dataset, n_examples, shuffle, run_seed):
    """Run the reservoir on n_examples samples and return raw spike tensors.

    A fresh DataLoader is created each call so that shuffle=True draws a
    different subset of examples on every run.
    """
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=shuffle,
        num_workers=0, pin_memory=gpu,
        generator=torch.Generator().manual_seed(run_seed),
    )
    raw_pairs = []
    pbar = tqdm(enumerate(dataloader), total=n_examples)
    for i, dataPoint in pbar:
        if i >= n_examples:
            break
        datum = dataPoint["encoded_image"].view(int(time / dt), 1, 1, 28, 28).to(device)
        label = dataPoint["label"]
        network.run(inputs={"I": datum}, time=time)
        raw_pairs.append((spikes["O"].get("s").cpu(), label))
        network.reset_state_variables()
        pbar.set_description_str("Reservoir (%d / %d)" % (i + 1, n_examples))
    return raw_pairs


# ── Readout model ────────────────────────────────────────────
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear_1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear_1(x.float().view(1, -1))  # raw logits: (1, num_classes)


def bin_pairs(raw_pairs, n_bins):
    """Convert stored raw spike tensors to binned feature vectors.

    Bins directly from n_bins rather than going through bin_ms→bin_steps to
    avoid rounding errors when time/n_bins is not an integer.
    """
    pairs = []
    for raw, label in raw_pairs:
        s = raw.squeeze(1) if raw.dim() == 3 else raw   # (T, N)
        T, N = s.shape
        trim_T = (T // n_bins) * n_bins                  # drop remainder timesteps
        s_trimmed = s[:trim_T].float()
        binned = s_trimmed.reshape(n_bins, trim_T // n_bins, N).sum(dim=1)  # (n_bins, N)
        pairs.append((binned.flatten(), label))
    return pairs


def train_readout(train_pairs, input_size):
    model = NN(input_size, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20
    )

    model.train()
    for epoch in range(n_epochs):
        avg_loss = 0.0
        shuffled = train_pairs.copy()
        random.shuffle(shuffled)
        for s, l in shuffled:
            optimizer.zero_grad()
            logits = model(s.to(device))
            target = torch.tensor([int(l)], dtype=torch.long).to(device)
            loss = criterion(logits, target)
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_loss /= len(train_pairs)
        scheduler.step(avg_loss)
    return model


def evaluate_readout(model, test_pairs):
    model.eval()
    correct = 0
    with torch.no_grad():
        for s, label in test_pairs:
            logits = model(s.to(device))
            predicted = logits.argmax(dim=1).item()
            correct += int(predicted == int(label))
    return 100.0 * correct / len(test_pairs)


# ── Sweep: outer = runs, inner = bin counts ──────────────────
# Each run uses a different seed, giving:
#   • a different shuffled subset of training samples
#   • different Poisson spike encodings (via torch Generator)
#   • different readout weight initialisation and epoch shuffle order
print(f"\nSweeping bin counts: {BIN_COUNTS}  ({N_RUNS} runs each)")
run_results = {n_bins: [] for n_bins in BIN_COUNTS}

for run in range(N_RUNS):
    run_seed = seed + run
    print(f"\n════ Run {run + 1}/{N_RUNS}  (seed={run_seed}) ════")

    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)

    print("  Collecting train reservoir responses...")
    raw_train_pairs = collect_reservoir_responses(
        train_dataset, examples_train, shuffle=True, run_seed=run_seed
    )
    print("  Collecting test reservoir responses...")
    raw_test_pairs = collect_reservoir_responses(
        test_dataset, examples_test, shuffle=False, run_seed=run_seed
    )

    for n_bins in BIN_COUNTS:
        bin_ms_val = time / n_bins
        train_pairs = bin_pairs(raw_train_pairs, n_bins)
        test_pairs  = bin_pairs(raw_test_pairs,  n_bins)
        input_size  = train_pairs[0][0].numel()

        random.seed(run_seed)
        torch.manual_seed(run_seed)
        model = train_readout(train_pairs, input_size)
        acc   = evaluate_readout(model, test_pairs)
        run_results[n_bins].append(acc)
        print(f"  n_bins={n_bins:3d} ({bin_ms_val:.1f} ms)  acc={acc:.2f}%")

means = []
stds  = []
for n_bins in BIN_COUNTS:
    mean_acc = float(np.mean(run_results[n_bins]))
    std_acc  = float(np.std(run_results[n_bins]))
    means.append(mean_acc)
    stds.append(std_acc)
    print(f"n_bins={n_bins}  → mean={mean_acc:.2f}%  std={std_acc:.2f}%")

# ── Plot ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(BIN_COUNTS, means, yerr=stds,
            marker="o", linewidth=2, markersize=8,
            color="steelblue", capsize=5, capthick=1.5, elinewidth=1.5)
for x, y, e in zip(BIN_COUNTS, means, stds):
    ax.annotate(f"{y:.1f}±{e:.1f}%", (x, y),
                textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=8)
ax.set_xlabel("Number of bins", fontsize=12)
ax.set_ylabel("Test accuracy (%)", fontsize=12)
ax.set_title(f"Readout accuracy vs. bin count  ({N_RUNS} runs, mean ± std)\n"
             f"(reservoir: {n_neurons} neurons, {examples_train} train / {examples_test} test, "
             f"{n_epochs} epochs)", fontsize=10)
ax.set_xticks(BIN_COUNTS)
ax.set_xticklabels([f"{n}\n({time/n:.1f} ms)" for n in BIN_COUNTS])
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()

out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "bin_sweep")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "bin_sweep_accuracy.png")
fig.savefig(out_path, dpi=150)
print(f"\nPlot saved to {out_path}")
plt.show(block=True)
