# input_data.py
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from bindsnet.encoding import PoissonEncoder

from .gabor_bank import build_gabor_bank #remove the . if youre running the file directly
import matplotlib.pyplot as plt

class Data:
    def __init__(self, dt, shuffle=True, intensity=64,
                 kernel_size=9, thetas_deg=(0,45,90,135),
                 sigma=2.0, lam=4.0, gamma=0.5, psi=0.0, convolution=True):
        self.dt = dt
        self.shuffle = shuffle
        self.root = "../../data"
        self.intensity = intensity
        self.convolution = convolution

        if self.convolution:
            self.use_gabor = True
        else:
            self.use_gabor = False

        # fixed gabor weights (K,1,ks,ks)
        self.W_gabor = build_gabor_bank(
            kernel_size=kernel_size,
            thetas_deg=thetas_deg,
            sigma=sigma,
            lam=lam,
            gamma=gamma,
            psi=psi,
            device="cpu",
        )

        self.train_dataset = None
        self.test_dataset = None

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * self.intensity),  # keep your scaling
        ])

    def _to_feature_map(self, x_1x28x28: torch.Tensor) -> torch.Tensor:
        """
        x_1x28x28: (1,28,28)
        returns: feature_map (K, Hf, Wf) with ReLU applied
        """
        if not self.use_gabor:
            return x_1x28x28

        x = x_1x28x28.unsqueeze(0)  # (1,1,28,28)
        feat = F.conv2d(x, self.W_gabor, bias=None, stride=1, padding=self.W_gabor.shape[-1]//2)
        feat = torch.relu(feat)  # keep positive evidence


        # feat shape: (1, K, H, W)
        max_vals = feat.amax(dim=(2,3), keepdim=True)   # (1,K,1,1)
        feat = feat * (feat >= 0.15 * max_vals)
        return feat.squeeze(0)   # (K,Hf,Wf)

    def load_MNIST(self):
        base_train = MNIST(root=os.path.join(self.root, "MNIST"), download=True, train=True, transform=self.tf)
        base_test  = MNIST(root=os.path.join(self.root, "MNIST"), download=True, train=False, transform=self.tf)

        self.train_dataset = _FeatureDataset(base_train, self._to_feature_map)
        self.test_dataset  = _FeatureDataset(base_test,  self._to_feature_map)
        return self.train_dataset, self.test_dataset

    def plot_feature_map(self, feature_map, title="Gabor feature maps"):
        K, H, W = feature_map.shape
        cols = min(K, 4)
        rows = (K + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        axes = axes.flatten()

        for k in range(K):
            ax = axes[k]
            ax.imshow(feature_map[k].cpu(), cmap="gray")
            ax.set_title(f"kernel {k}")
            ax.axis("off")

        for k in range(K, len(axes)):
            axes[k].axis("off")

        plt.suptitle(title)
        plt.tight_layout()
        plt.show(block=True)

    

class _FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, base, feature_fn):
        self.base = base
        self.feature_fn = feature_fn

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]              # img: (1,28,28)
        feat = self.feature_fn(img)             # (K,Hf,Wf)
        return {"image": img, "label": int(label), "feature_map": feat}
