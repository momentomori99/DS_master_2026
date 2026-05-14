from framework import Framework
from readout import Readout
from data.input_data_CNN import Data as Data_CNN
from tools.metrics import calculate_fisher_ratio
from tools.pca import apply_pca, plot_explained_variance
import numpy as np
import torch
import matplotlib.pyplot as plt
from visualization.visualizations_readout import plot_tsne, plot_confusion_heatmap, plot_rsa_heatmap, plot_tsne_rsa
from visualization.visualizations_spatial import plot_EI_positions, plot_outgoing_connections, plot_spikecount_grid
from visualization.visualizations import self_tuning_plot

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

intensity = 150
seed = 54

mnist_input = True
heterogeneity = True
spatial = True
convolution = True


g = 5
eta = 1.3
sigma_input = 1
sigma_network = 1
epsilon = 0.1

data_CNN = Data_CNN(dt=dt, intensity=intensity, kernel_size=9, thetas_deg=(0, 45, 90, 135), convolution=convolution)
train_dataset, test_dataset = data_CNN.load_MNIST()

framework = Framework(
    n_neurons=n_neurons,
    time=time,
    dt=dt,
    bin_ms=bin_ms,
    seed=seed,
    heterogeneity=heterogeneity,
    mnist_input=mnist_input,
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

#plot_EI_positions(framework.pos_E, framework.pos_I)
#plot_outgoing_connections(framework.mask_EE, framework.pos_E, 505)

target_label = 0
framework.run_one_sample(train_dataset, target_label)

pairs_train, CV_list, rho_mean_list, rate_list, g_list, eta_list = framework.run_stimulation(train_dataset, examples_train)
pairs_test, *_ = framework.run_stimulation(test_dataset, examples_test)




if pca:
    pairs_train, pairs_test_pca = apply_pca(pairs_train, pairs_test, n_components)
    plot_explained_variance(pairs_train)
    plot_explained_variance(pairs_test)



feature_dim = pairs_train[0][0].numel()
readout = Readout(input_size=feature_dim, num_classes=10, seed=seed)
readout.train_readout(pairs_train, n_epochs=n_epochs)
acc = readout.test_readout(pairs_test)
print(f"Accuracy: {acc:.2f}%")




plot_tsne(pairs_train, perplexity=30)
plot_rsa_heatmap(pairs_train)


