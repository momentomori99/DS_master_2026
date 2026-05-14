# Biologically Inspired Reservoir Computing Framework

**Master's Thesis — Domantas Sakalys, 2026**  
Faculty of Science and Technology, NMBU

---

## Overview

This repository contains the full implementation for the master's thesis *"Biologically Inspired Reservoir Computing Framework"*. 

---

## Repository Structure

```
DS_master_2026/
│
├── thesis.pdf                    # Full master's thesis document
├── requirements.txt              # Python package dependencies
│
└── base/                         # All source code lives here
    │
    ├── framework.py              # *** THE MAIN MODEL ***
    │                             
    │
    ├── main.py                   # Primary entry point — instantiates and runs
    │                             #   the Framework with configurable parameters,
    │                             #   trains the linear readout, and produces
    │                             #   evaluation plots (t-SNE, RSA, raster, etc.)
    │
    ├── main_g_eta_sweep.py       # Parameter sweep over (g, η) space to map
    │                             #   the Brunel phase diagram and its effect on
    │                             #   classification accuracy
    │
    ├── main_multi_split.py       # Multi-run evaluation with different train/test
    │                             #   splits for robust accuracy estimation
    │
    ├── readout.py                # Linear readout classifier (logistic regression
    │                             #   via a single-layer NN + Adam + LR scheduler)
    │
    ├── data/
    │   ├── input_data_CNN.py     # MNIST loading pipeline with optional Gabor
    │   │                         #   filter preprocessing (4-orientation bank)
    │   └── gabor_bank.py         # Gabor filter bank construction
    │
    ├── tools/
    │   ├── metrics.py            # CV, ρ (synchrony), firing rate, Fisher ratio
    │   ├── spatial_tools.py      # Toroidal EI lattice + distance-dependent
    │   │                         #   connection masks
    │   ├── build_W_in.py         # Input weight matrix construction (tiled
    │   │                         #   Gaussian and pixel-Gaussian projections)
    │   ├── feature_encoding.py   # Poisson spike encoding and binned spike counts
    │   ├── pca.py                # Optional PCA dimensionality reduction on
    │   │                         #   reservoir features
    │   └── other.py              # Per-neuron parameter sampling (heterogeneity)
    │
    ├── visualization/
    │   ├── visualizations.py          # Raster plots, rate/spike distributions,
    │   │                              #   self-tuning trajectory plots
    │   ├── visualizations_readout.py  # t-SNE, RSA heatmaps, confusion matrices
    │   └── visualizations_spatial.py  # EI position maps, connection diagrams,
    │                                  #   spike count heatmaps on the lattice
    │
    ├── basemodel/                # Iterative baseline implementations (steps 1–5)
    │   ├── 1baseline.py          # Original BindsNET example baseline
    │   ├── 2baseline_leakagefix.py
    │   ├── 3baseline_readoutfix.py
    │   ├── 4baseline_binned.py
    │   └── 5baseline_binsweep.py
    │
    └── results/                  # Saved output plots and result artefacts
```

---

## Reference

Sakalys, D. (2026). *Biologically Inspired Reservoir Computing Framework*. Master's Thesis, NMBU - Faculty of Science and Technology.
