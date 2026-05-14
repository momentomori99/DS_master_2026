import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

# Allow imports from the parent BindsNet directory (tools/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools.feature_encoding import spikes_to_binned_counts

from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.network import Network

# Build a simple two-layer, input-output network.
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.utils import get_square_weights

# ============================================================
#  Parameters
# ============================================================
seed              = 0
n_neurons         = 500
n_epochs          = 100
examples          = 500
time              = 250       # simulation time per sample [ms]
dt                = 1.0       # timestep [ms]
bin_ms            = 50        # width of each spike-count bin [ms]
intensity         = 64.0      # input encoding intensity
progress_interval = 10
update_interval   = 250
plot              = False
gpu               = False
# ============================================================

# Compute binning dimensions
bin_steps = int(round(bin_ms / dt))
n_bins = time // bin_steps          # number of bins per sample
readout_input_size = n_bins * n_neurons

np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False
torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)
print(f"Binning: bin_ms={bin_ms}, bin_steps={bin_steps}, n_bins={n_bins}, readout input size={readout_input_size}")

# Create simple Torch NN
network = Network(dt=dt)
inpt = Input(784, shape=(1, 28, 28))
network.add_layer(inpt, name="I")
output = LIFNodes(n_neurons, thresh=-52 + np.random.randn(n_neurons).astype(float))
network.add_layer(output, name="O")
C1 = Connection(source=inpt, target=output, w=0.5 * torch.randn(inpt.n, output.n))
C2 = Connection(source=output, target=output, w=0.5 * torch.randn(output.n, output.n))

network.add_connection(C1, source="I", target="O")
network.add_connection(C2, source="O", target="O")

# Monitors for visualizing activity
spikes = {}
for l in network.layers:
    spikes[l] = Monitor(network.layers[l], ["s"], time=time, device=device)
    network.add_monitor(spikes[l], name="%s_spikes" % l)

voltages = {"O": Monitor(network.layers["O"], ["v"], time=time, device=device)}
network.add_monitor(voltages["O"], name="O_voltages")

# Directs network to GPU
if gpu:
    network.to("cuda")

# Load MNIST training and test splits separately to avoid leakage.
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

inpt_axes = None
inpt_ims = None
spike_axes = None
spike_ims = None
weights_im = None
weights_im2 = None
voltage_ims = None
voltage_axes = None

# Create dataloaders for training and test splits
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=gpu
)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=gpu
)

# Run training data on reservoir computer and store (binned spike counts, label) per example.
# Note: Because this is a reservoir network, no adjustments of neuron parameters occurs in this phase.
n_iters = examples
training_pairs = []
pbar = tqdm(enumerate(train_dataloader))
for i, dataPoint in pbar:
    if i > n_iters:
        break

    # Extract & resize the MNIST samples image data for training
    #       int(time / dt)  -> length of spike train
    #       28 x 28         -> size of sample
    datum = dataPoint["encoded_image"].view(int(time / dt), 1, 1, 28, 28).to(device)
    label = dataPoint["label"]
    pbar.set_description_str("Train progress: (%d / %d)" % (i, n_iters))

    # Run network on sample image
    network.run(inputs={"I": datum}, time=time)

    # Bin spike counts: (T, 1, N) -> (n_bins, N) -> (n_bins * N,)
    raw_spikes = spikes["O"].get("s")           # (T, 1, N_neurons)
    binned = spikes_to_binned_counts(raw_spikes, bin_ms=bin_ms, dt=dt, time=time)  # (n_bins, N_neurons)
    features = binned.flatten().float()         # (n_bins * N_neurons,)
    training_pairs.append([features, label])

    # Plot spiking activity using monitors
    if plot:
        # Plot the current image and reconstructed/encoded image
        inpt_axes, inpt_ims = plot_input(
            dataPoint["image"].view(28, 28),
            datum.view(int(time / dt), 784).sum(0).view(28, 28),
            label=label,
            axes=inpt_axes,
            ims=inpt_ims,
        )
        # Plot spikes
        spike_ims, spike_axes = plot_spikes(
            {layer: spikes[layer].get("s").view(time, -1) for layer in spikes},
            axes=spike_axes,
            ims=spike_ims,
        )
        # Plot voltages
        voltage_ims, voltage_axes = plot_voltages(
            {layer: voltages[layer].get("v").view(time, -1) for layer in voltages},
            ims=voltage_ims,
            axes=voltage_axes,
        )
        # Plot weights between input and output
        weights_im = plot_weights(
            get_square_weights(C1.w, 23, 28), im=weights_im, wmin=-2, wmax=2
        )
        # Plot weights between output and output
        weights_im2 = plot_weights(C2.w, im=weights_im2, wmin=-2, wmax=2)

        plt.pause(1e-8)
    network.reset_state_variables()


# Define logistic regression model using PyTorch.
# These neurons will take the reservoirs output as its input, and be trained to classify the images.
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.linear_1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear_1(x.float().view(1, -1))
        return out  # raw logits: (1, num_classes)


# Create and train logistic regression model on reservoir outputs.
model = NN(readout_input_size, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=20
)

# Training the Model
print("\n Training the read out")
model.train()
pbar = tqdm(enumerate(range(n_epochs)))
for epoch, _ in pbar:
    avg_loss = 0

    # Extract spike outputs from reservoir for a training sample
    #       i   -> Loop index
    #       s   -> Binned spike count features
    #       l   -> Image label
    shuffled_pairs = training_pairs.copy()
    random.shuffle(shuffled_pairs)

    for i, (s, l) in enumerate(shuffled_pairs):
        # Reset gradients to 0
        optimizer.zero_grad()

        # Run binned features through logistic regression model
        logits = model(s)  # (1, num_classes)

        # Calculate CrossEntropy loss
        target = torch.tensor([int(l)], dtype=torch.long).to(device)
        loss = criterion(logits, target)
        avg_loss += loss.item()

        # Optimize parameters
        loss.backward()
        optimizer.step()

    avg_loss /= len(training_pairs)
    scheduler.step(avg_loss)
    pbar.set_description_str(
        "Epoch: %d/%d, Loss: %.4f"
        % (epoch + 1, n_epochs, avg_loss)
    )

# Run simulation on reservoir with held-out MNIST test split.
n_iters = examples
test_pairs = []
pbar = tqdm(enumerate(test_dataloader))
for i, dataPoint in pbar:
    if i > n_iters:
        break
    datum = dataPoint["encoded_image"].view(int(time / dt), 1, 1, 28, 28).to(device)
    label = dataPoint["label"]
    pbar.set_description_str("Testing progress: (%d / %d)" % (i, n_iters))

    network.run(inputs={"I": datum}, time=time)

    # Bin spike counts: (T, 1, N) -> (n_bins, N) -> (n_bins * N,)
    raw_spikes = spikes["O"].get("s")
    binned = spikes_to_binned_counts(raw_spikes, bin_ms=bin_ms, dt=dt, time=time)
    features = binned.flatten().float()
    test_pairs.append([features, label])

    if plot:
        inpt_axes, inpt_ims = plot_input(
            dataPoint["image"].view(28, 28),
            datum.view(time, 784).sum(0).view(28, 28),
            label=label,
            axes=inpt_axes,
            ims=inpt_ims,
        )
        spike_ims, spike_axes = plot_spikes(
            {layer: spikes[layer].get("s").view(time, -1) for layer in spikes},
            axes=spike_axes,
            ims=spike_ims,
        )
        voltage_ims, voltage_axes = plot_voltages(
            {layer: voltages[layer].get("v").view(time, -1) for layer in voltages},
            ims=voltage_ims,
            axes=voltage_axes,
        )
        weights_im = plot_weights(
            get_square_weights(C1.w, 23, 28), im=weights_im, wmin=-2, wmax=2
        )
        weights_im2 = plot_weights(C2.w, im=weights_im2, wmin=-2, wmax=2)

        plt.pause(1e-8)
    network.reset_state_variables()

# Test model with previously trained logistic regression classifier
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for s, label in test_pairs:
        logits = model(s)                       # (1, num_classes)
        predicted = logits.argmax(dim=1).item()
        total += 1
        correct += int(predicted == int(label))

print(
    "\n Accuracy of the model on %d test images: %.2f %%"
    % (n_iters, 100 * correct / total)
)
