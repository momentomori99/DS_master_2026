import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def apply_pca(pairs_train, pairs_test, n_components):
    """
    Fit PCA on training features, then transform both train and test pairs.

    Returns new (feature, label) pair lists with reduced dimensionality.
    """
    features_train = torch.stack([f for f, _ in pairs_train]).numpy()
    labels_train = [l for _, l in pairs_train]

    features_test = torch.stack([f for f, _ in pairs_test]).numpy()
    labels_test = [l for _, l in pairs_test]

    pca = PCA(n_components=n_components)
    features_train_pca = pca.fit_transform(features_train)
    features_test_pca = pca.transform(features_test)

    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"PCA: {n_components} components, {explained:.1f}% variance explained")

    pairs_train_pca = [
        (torch.tensor(f, dtype=torch.float32), l)
        for f, l in zip(features_train_pca, labels_train)
    ]
    pairs_test_pca = [
        (torch.tensor(f, dtype=torch.float32), l)
        for f, l in zip(features_test_pca, labels_test)
    ]

    return pairs_train_pca, pairs_test_pca


def plot_explained_variance(pairs):
    """
    Fit full PCA on the feature matrix and plot cumulative explained variance
    as a function of the number of principal components.
    """
    features = torch.stack([f for f, _ in pairs]).numpy()
    n_components_max = min(features.shape[0], features.shape[1])

    pca = PCA(n_components=n_components_max)
    pca.fit(features)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, n_components_max + 1), cumulative_variance)
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("PCA — Explained Variance")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/pca_explained_variance.png", dpi=150, bbox_inches="tight")
    plt.show()
