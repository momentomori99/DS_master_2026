from framework import Framework
from readout import Readout
from data.input_data_CNN import Data as Data_CNN
from tools.pca import apply_pca
import os
import torch
from torch.utils.data import Subset


# parameters
n_neurons = 1000
n_epochs = 50
examples_train = 500
examples_test = 200
n_splits = 10
use_pca = False
n_components = 60

time = 250
dt = 1.0

intensity = 200
seed = 42

mnist_input = True
heterogeneity = False
self_tuning = False
spatial = False
convolution = False
log_normal = False

g = 5
eta = 1
sigma_input = 1
sigma_network = 1
epsilon = 0.5


def build_distinct_splits(dataset_size, samples_per_split, n_splits, seed):
    required = samples_per_split * n_splits
    if required > dataset_size:
        raise ValueError(
            f"Need {required} samples but dataset has only {dataset_size}. "
            "Reduce n_splits or samples_per_split."
        )
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(dataset_size, generator=generator)
    return [
        perm[i * samples_per_split : (i + 1) * samples_per_split].tolist()
        for i in range(n_splits)
    ]


data_CNN = Data_CNN(
    dt=dt,
    intensity=intensity,
    kernel_size=9,
    thetas_deg=(0, 45, 90, 135),
    convolution=convolution
)
train_dataset, test_dataset = data_CNN.load_MNIST()

framework = Framework(
    n_neurons=n_neurons,
    time=time,
    dt=dt,
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
)
framework.build_network()

train_split_indices = build_distinct_splits(
    dataset_size=len(train_dataset),
    samples_per_split=examples_train,
    n_splits=n_splits,
    seed=seed,
)
test_split_indices = build_distinct_splits(
    dataset_size=len(test_dataset),
    samples_per_split=examples_test,
    n_splits=n_splits,
    seed=seed + 1,
)

split_accuracies = []
all_CV = []
all_rho = []
all_rate = []

for split_id in range(n_splits):
    train_subset = Subset(train_dataset, train_split_indices[split_id])
    test_subset = Subset(test_dataset, test_split_indices[split_id])

    pairs_train, CV_train, rho_train, rate_train, *_ = framework.run_stimulation(
        train_subset, examples_train, shuffle=False
    )
    pairs_test, CV_test, rho_test, rate_test, *_ = framework.run_stimulation(
        test_subset, examples_test, shuffle=False
    )

    all_CV.extend(CV_train + CV_test)
    all_rho.extend(rho_train + rho_test)
    all_rate.extend(rate_train + rate_test)

    if use_pca:
        pairs_train, pairs_test = apply_pca(pairs_train, pairs_test, n_components)

    feature_dim = pairs_train[0][0].numel()
    readout = Readout(input_size=feature_dim, num_classes=10, seed=seed + split_id)
    readout.train_readout(pairs_train, n_epochs=n_epochs)
    acc = readout.test_readout(pairs_test)
    split_accuracies.append(acc)

    print(f"Split {split_id + 1}/{n_splits} accuracy: {acc:.2f}%")

mean_acc = sum(split_accuracies) / len(split_accuracies)
std_acc = torch.tensor(split_accuracies).std(unbiased=False).item()
print(f"\nAverage: {mean_acc:.2f}% +/- {std_acc:.2f}%")

# --- Save results to .txt ---
os.makedirs("results", exist_ok=True)
result_path = os.path.join("results", "multi_split_results.txt")

from datetime import datetime

with open(result_path, "a") as f:
    f.write("\n" + "#" * 60 + "\n")
    f.write(f"RUN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("#" * 60 + "\n\n")
    f.write("=" * 60 + "\n")
    f.write("PARAMETERS\n")
    f.write("=" * 60 + "\n")
    f.write(f"n_neurons        = {n_neurons}\n")
    f.write(f"n_epochs         = {n_epochs}\n")
    f.write(f"examples_train   = {examples_train}\n")
    f.write(f"examples_test    = {examples_test}\n")
    f.write(f"n_splits         = {n_splits}\n")
    f.write(f"use_pca          = {use_pca}\n")
    f.write(f"n_components     = {n_components}\n")
    f.write(f"time             = {time}\n")
    f.write(f"dt               = {dt}\n")
    f.write(f"intensity        = {intensity}\n")
    f.write(f"seed             = {seed}\n")
    f.write(f"mnist_input      = {mnist_input}\n")
    f.write(f"heterogeneity    = {heterogeneity}\n")
    f.write(f"self_tuning      = {self_tuning}\n")
    f.write(f"spatial          = {spatial}\n")
    f.write(f"convolution      = {convolution}\n")
    f.write(f"log_normal       = {log_normal}\n")
    f.write(f"g                = {g}\n")
    f.write(f"eta              = {eta}\n")
    f.write(f"sigma_input      = {sigma_input}\n")
    f.write(f"sigma_network    = {sigma_network}\n")
    f.write(f"epsilon          = {epsilon}\n")

    f.write("\n" + "=" * 60 + "\n")
    f.write("ACCURACIES\n")
    f.write("=" * 60 + "\n")
    f.write(f"Mean: {mean_acc:.2f}% +/- {std_acc:.2f}%\n")
    f.write(f"Per split: {split_accuracies}\n")

    f.write("\n" + "=" * 60 + "\n")
    f.write("DYNAMICS (all samples across all splits)\n")
    f.write("=" * 60 + "\n")
    f.write(f"CV   = {all_CV}\n")
    f.write(f"rho  = {all_rho}\n")
    f.write(f"rate = {all_rate}\n")

print(f"\nResults saved to {result_path}")
