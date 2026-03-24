# HASTE

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-orange.svg)](https://pytorch.org)

**Hybrid-Attention Streaming Topology-aware Estimator for EEG Fatigue Monitoring**

> From Electrode Geometry to Adaptive Coupling: HASTE for Streaming EEG Fatigue Monitoring

HASTE is a causal spatio-temporal model for subject-independent continuous fatigue regression from streaming EEG. It separates *where* electrode interactions are allowed (physical kNN graph) from *how strongly* they adapt (online band-aware functional priors), and injects this knowledge into spatial attention as an explicit, tunable log-bias.

---

## Architecture

```
Multi-band EEG Window X_t (C x B)
        |
        v
+---------------------------------------+
|  Causal Spectral Fusion (GRU)          |  --> band weights alpha_t
|  alpha_t = Softmax(W_s * GRU(u_t))    |
+---------------------------------------+
        |                    |
        v                    v
  Fused signal         TopK Band Gating
  x_t (C x 1)         B_t = TopK(alpha, K)
        |                    |
        |                    v
        |          Band Functional Priors
        |          P^func_{t,b} (per active band)
        |                    |
        |                    v
        |          Prior Mixing Gate g_t
        |          P_t = g_t * P^phy + (1-g_t) * P^func_t
        |                    |
        v                    v
+---------------------------------------+
|  Spatial Encoder (GAT + log-bias)      |
|  l_ij = LeakyReLU(a^T[Wh_i||Wh_j])   |
|       + beta * log(P_ij + eps)         |
+---------------------------------------+
        |
        v
+---------------------------------------+
|  Temporal Encoder (CLTA, window=L)     |
|  Causal local attention + KV cache     |
+---------------------------------------+
        |
        v
  y_hat_t = sigma(w^T h_t + b)    Fatigue score in [0, 1]
```

**Key properties**: constant-memory streaming inference (~2.8 ms/update, ~47 MB GPU on a single GPU).

---

## Installation

```bash
git clone https://github.com/<your-org>/HAST.git
cd HAST
pip install -e ".[dev]"
```

Or without packaging:

```bash
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Train on SEED-VIG (fold 0)
python train.py --config configs/seedvig.yaml --data-root ./data_seed_vig --fold 0

# Train on SADT
python train.py --config configs/sadt.yaml --data-root ./data_sadt --fold 0

# Evaluate a checkpoint
python train.py --config configs/seedvig.yaml --eval-only --checkpoint checkpoints/hast_seedvig_fold0.pth
```

---

## Datasets

HASTE is evaluated on three public EEG fatigue datasets:

| Dataset | Channels | Subjects | Sessions | Label Source | k (kNN) |
|---------|----------|----------|----------|--------------|---------|
| **SEED-VIG** | 17 | 23 | 23 | PERCLOS | 5 |
| **SADT** | 30 | 27 | 62 | RT -> DI | 7 |
| **MPD-DF** | 32 | 50 | 50 | Physician levels | 7 |

### SEED-VIG

Uses the publicly released 5-band DE features. Place data as:

```
data_seed_vig/
  EEG_Feature_5Bands/
    1_20151124_*.mat    # contains 'de_movingAve' (17, T, 5)
    ...
  PERCLOS/
    1_20151124_*.mat    # contains 'PERCLOS' (T,)
    ...
```

### SADT

Raw EEG at 500 Hz from 30 scalp channels. Pre-process to 5-band DE features using `data/preprocessing.py`, then place as:

```
data_sadt/
  DE_features/
    sub01_sess01.mat    # contains 'de_features' (30, T, 5)
    ...
  labels/
    sub01_sess01.mat    # contains 'drowsiness_index' (T,) or 'rt' (T,)
    ...
```

### MPD-DF

32-channel EEG at 500 Hz with physician-annotated fatigue levels (1-5). Pre-process similarly:

```
data_mpddf/
  DE_features/
    sub01.mat           # contains 'de_features' (32, T, 5)
    ...
  labels/
    sub01.mat           # contains 'fatigue_level' (T,) in {1,2,3,4,5}
    ...
```

---

## Training

### Single Fold

```bash
bash scripts/train_seedvig.sh --fold 0
bash scripts/train_sadt.sh --fold 0
bash scripts/train_mpddf.sh --fold 0
```

### Full LOSO (all folds)

```bash
bash scripts/run_all_folds.sh configs/seedvig.yaml
bash scripts/run_all_folds.sh configs/sadt.yaml
bash scripts/run_all_folds.sh configs/mpddf.yaml
```

### Custom Hyperparameters

CLI arguments override YAML config values:

```bash
python train.py --config configs/seedvig.yaml --lr 5e-4 --batch-size 16 --epochs 50
```

### TensorBoard

Training logs are written to `runs/`. View with:

```bash
tensorboard --logdir runs/
```

---

## Evaluation

```bash
# Evaluate a single checkpoint
bash scripts/evaluate.sh --config configs/seedvig.yaml \
    --checkpoint checkpoints/hast_seedvig_fold0.pth --fold 0
```

---

## Project Structure

```
HAST/
├── configs/                    # YAML configs per dataset
│   ├── seedvig.yaml
│   ├── sadt.yaml
│   └── mpddf.yaml
├── data/                       # Dataset loaders
│   ├── __init__.py
│   ├── seedvig.py              # SEED-VIG DE feature loader
│   ├── sadt.py                 # SADT loader (RT -> DI mapping)
│   ├── mpddf.py                # MPD-DF loader (fatigue levels -> [0,1])
│   └── preprocessing.py        # Raw EEG -> 5-band DE pipeline (MNE)
├── models/                     # Model architecture
│   ├── __init__.py
│   ├── hast.py                 # HASTModel, loss function
│   ├── blocks.py               # CausalSpectralFusion, BandFunctionalPrior,
│   │                           # PriorMixingGate, MultiHeadGraphAttention,
│   │                           # SpatialTGALayer, TemporalBlock
│   └── adjacency.py            # kNN graph construction, row normalization
├── scripts/                    # Shell scripts
│   ├── train_seedvig.sh
│   ├── train_sadt.sh
│   ├── train_mpddf.sh
│   ├── evaluate.sh
│   └── run_all_folds.sh
├── tests/                      # Pytest test suite
│   ├── conftest.py
│   ├── test_model.py
│   ├── test_data.py
│   └── test_config.py
├── train.py                    # Training / evaluation CLI
├── pyproject.toml              # Package metadata
├── requirements.txt            # Dependencies
├── LICENSE                     # MIT
├── CONTRIBUTING.md             # Contribution guidelines
└── README.md
```

---

## Hyperparameters

Default hyperparameters (from the paper, Table 2):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d` (spatial) | 64 | Spatial model dimension |
| `d_model` (temporal) | 128 | Temporal model dimension |
| `d_r` | 32 | GRU hidden dim (spectral fusion) |
| `d_s` | 16 | Low-dim similarity for functional prior |
| `K` (TopK) | 3 | Active bands for gating |
| `L` (look-back) | 10 | CLTA window size |
| `H_s` | 4 | Spatial attention heads |
| `L_s` | 2 | Spatial layers |
| `H_t` | 4 | Temporal attention heads |
| Optimizer | AdamW | lr=1e-3, wd=1e-4 |
| Batch size | 32 | |
| Max epochs | 100 | with early stopping |
| `lambda_pred` | 0.1 | Prediction smoothness weight |
| `lambda_band` | 0.01 | Band drift regularization |
| `lambda_graph` | 0.01 | Graph/gate drift regularization |

---

## Testing

```bash
pytest tests/ -v
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{niu2025haste,
  title   = {From Electrode Geometry to Adaptive Coupling: {HASTE} for Streaming {EEG} Fatigue Monitoring},
  author  = {Niu, Qiyu and Xie, Yi and Liu, Siao and Wang, Shouyan and Nie, Yingnan},
  journal = {Knowledge-Based Systems},
  year    = {2025}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
