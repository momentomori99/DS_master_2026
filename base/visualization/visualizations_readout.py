import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _pairs_to_xy(pairs):
    if len(pairs) == 0:
        raise ValueError("`pairs` must contain at least one (features, label) sample.")
    X = torch.stack([p[0] for p in pairs])
    y = torch.tensor([int(p[1]) for p in pairs], dtype=torch.long)
    return X, y


def plot_tsne(
    pairs,
    perplexity=30,
    max_iter=1000,
    random_state=42,
    figsize=(10, 8),
    title="t-SNE of reservoir features",
):
    """
    Plot a 2D t-SNE embedding directly from (features, label) pairs.
    """
    X, y = _pairs_to_xy(pairs)
    X_np = _to_numpy(X)
    y_np = _to_numpy(y).astype(int)

    if X_np.ndim != 2:
        raise ValueError(f"`features` must be 2D, got shape {X_np.shape}.")
    if y_np.ndim != 1:
        y_np = y_np.reshape(-1)
    if X_np.shape[0] != y_np.shape[0]:
        raise ValueError("`features` and `labels` must have the same number of samples.")
    if X_np.shape[0] < 2:
        raise ValueError("Need at least 2 samples for t-SNE.")

    effective_perplexity = min(perplexity, max(1, X_np.shape[0] - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        max_iter=max_iter,
        random_state=random_state,
    )
    X_emb = tsne.fit_transform(X_np)

    classes = np.unique(y_np)
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

    plt.figure(figsize=figsize)
    for idx, cls in enumerate(classes):
        mask = y_np == cls
        plt.scatter(
            X_emb[mask, 0],
            X_emb[mask, 1],
            s=25,
            alpha=0.8,
            color=colors[idx],
            label=f"{cls}",
        )

    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(title="Class", loc="best")
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig("tsne.png")
    plt.show(block=True)
    plt.close()

    return X_emb


def plot_confusion_heatmap(
    pairs,
    normalize=True,
    class_names=None,
    cmap="Blues",
    figsize=(8, 6),
    title=None,
):
    """
    Plot confusion heatmap directly from pairs.

    Predicted labels are built with a nearest-centroid classifier on the same pairs.
    """
    X, y_true = _pairs_to_xy(pairs)
    X_np = _to_numpy(X)
    y_true_np = _to_numpy(y_true).astype(int)

    classes = np.unique(y_true_np)
    centroids = []
    for cls in classes:
        centroids.append(X_np[y_true_np == cls].mean(axis=0))
    centroids = np.stack(centroids, axis=0)  # (C, D)

    # Predict class by nearest centroid (euclidean distance).
    dists = ((X_np[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    y_pred = classes[np.argmin(dists, axis=1)]

    cm = confusion_matrix(y_true_np, y_pred, labels=classes)
    if normalize:
        cm = cm.astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, np.maximum(row_sums, 1e-12))

    if class_names is None:
        tick_labels = [str(c) for c in classes]
    else:
        if len(class_names) != len(classes):
            raise ValueError("`class_names` length must match the number of classes.")
        tick_labels = class_names

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap=cmap,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        cbar=True,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    if title is None:
        title = "Confusion Matrix (nearest-centroid, normalized)" if normalize else "Confusion Matrix (nearest-centroid)"
    plt.title(title)
    plt.tight_layout()
    plt.show(block=True)
    plt.close()

    return cm


def plot_rsa_heatmap(
    pairs,
    class_names=None,
    cmap="RdBu_r",
    figsize=(8, 6),
    title="Representational Similarity (Pearson)",
    vmin=-1.0,
    vmax=1.0,
):
    """
    Representational Similarity Analysis: Pearson correlation between class
    centroids in feature space.

    Returns the (C, C) correlation matrix.
    """
    X, y = _pairs_to_xy(pairs)
    X_np = _to_numpy(X)
    y_np = _to_numpy(y).astype(int)

    classes = np.unique(y_np)
    centroids = np.stack(
        [X_np[y_np == cls].mean(axis=0) for cls in classes], axis=0
    )  # (C, D)

    grand_mean = centroids.mean(axis=0, keepdims=True)
    centroids = centroids - grand_mean

    corr = np.corrcoef(centroids)  # (C, C)

    if class_names is None:
        tick_labels = [str(c) for c in classes]
    else:
        if len(class_names) != len(classes):
            raise ValueError("`class_names` length must match the number of classes.")
        tick_labels = class_names

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        vmin=vmin,
        vmax=vmax,
        center=0.0,
        cbar_kws={"label": "Pearson r"},
        square=True,
    )
    plt.xlabel("Class")
    plt.ylabel("Class")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("rsa.png")
    plt.show(block=True)
    plt.close()

    return corr


def plot_tsne_rsa(
    pairs,
    class_names=None,
    perplexity=30,
    max_iter=1000,
    random_state=42,
    rsa_cmap="RdBu_r",
    figsize=(14, 5.5),
    suptitle="Reservoir Feature Analysis",
):
    """
    Side-by-side t-SNE embedding (left) and RSA heatmap (right).

    Returns (X_emb, corr).
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    })

    X, y = _pairs_to_xy(pairs)
    X_np = _to_numpy(X)
    y_np = _to_numpy(y).astype(int)
    classes = np.unique(y_np)

    if class_names is None:
        tick_labels = [str(c) for c in classes]
    else:
        if len(class_names) != len(classes):
            raise ValueError("`class_names` length must match the number of classes.")
        tick_labels = class_names

    # ── t-SNE ──
    effective_perplexity = min(perplexity, max(1, X_np.shape[0] - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        max_iter=max_iter,
        random_state=random_state,
    )
    X_emb = tsne.fit_transform(X_np)

    # ── RSA ──
    centroids = np.stack(
        [X_np[y_np == cls].mean(axis=0) for cls in classes], axis=0
    )
    grand_mean = centroids.mean(axis=0, keepdims=True)
    centroids = centroids - grand_mean
    corr = np.corrcoef(centroids)

    # ── Figure ──
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.15, 1], wspace=0.35)

    cmap_scatter = plt.cm.Set2 if len(classes) <= 8 else plt.cm.tab20
    colors = cmap_scatter(np.linspace(0, 0.9, len(classes)))

    # Left: t-SNE
    ax0 = fig.add_subplot(gs[0])
    for idx, cls in enumerate(classes):
        mask = y_np == cls
        ax0.scatter(
            X_emb[mask, 0],
            X_emb[mask, 1],
            s=20,
            alpha=0.75,
            color=colors[idx],
            edgecolors="k",
            linewidths=0.3,
            label=tick_labels[idx],
        )
    ax0.set_title("t-SNE Embedding")
    ax0.set_xlabel("Dimension 1")
    ax0.set_ylabel("Dimension 2")
    ax0.legend(
        title="Class",
        loc="best",
        frameon=True,
        fancybox=False,
        edgecolor="0.7",
        framealpha=0.9,
    )
    ax0.grid(alpha=0.2, linestyle="--", linewidth=0.5)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    # Right: RSA heatmap
    ax1 = fig.add_subplot(gs[1])
    n_cls = len(classes)
    sns.heatmap(
        corr,
        annot=False,
        cmap=rsa_cmap,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        cbar_kws={"label": "Pearson $r$", "shrink": 0.82},
        linewidths=0.5 if n_cls <= 15 else 0.0,
        linecolor="white",
        ax=ax1,
    )

    annot_texts = []
    for i in range(n_cls):
        for j in range(n_cls):
            txt = ax1.text(
                j + 0.5, i + 0.5, f"{corr[i, j]:.2f}",
                ha="center", va="center", color="black", fontsize=1,
            )
            annot_texts.append(txt)

    def _resize_annots(event=None):
        bbox = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        cell_pts = (bbox.width * 72) / max(n_cls, 1)
        fs = max(4, min(12, cell_pts * 0.45))
        for t in annot_texts:
            t.set_fontsize(fs)
        ax1.tick_params(labelsize=max(5, min(10, fs)))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("resize_event", _resize_annots)
    _resize_annots()
    ax1.set_title("Representational Similarity (RSA)")
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Class")

    if suptitle:
        fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=1.02)

    fig.tight_layout()
    plt.show(block=True)
    plt.close()

    plt.rcParams.update(plt.rcParamsDefault)

    return X_emb, corr
