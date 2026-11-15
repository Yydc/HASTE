"""
HASTE training and evaluation CLI.

Supports SEED-VIG, SADT, and MPD-DF datasets via YAML config files.

Usage:
    python train.py --config configs/seedvig.yaml --data-root ./data_seed_vig
    python train.py --config configs/sadt.yaml --fold 0
    python train.py --config configs/mpddf.yaml --eval-only --checkpoint checkpoints/hast_mpddf_fold0_seed2025.pth
"""

import argparse
import copy
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data import (
    SEEDVIGSequenceDataset,
    build_file_pairs,
    extract_subject_id,
    SADTDataset,
    build_sadt_file_pairs,
    extract_subject_id_sadt,
    MPDDFDataset,
    build_mpddf_file_pairs,
    extract_subject_id_mpddf,
)
from models import (
    HASTModel,
    build_full_adjacency,
    build_knn_adjacency,
    coords_from_channel_names,
    hast_loss_fn,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_corr(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Pearson correlation between flattened tensors."""
    if y_true.numel() <= 1 or y_pred.numel() <= 1:
        return 0.0
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    vx = y_true - y_true.mean()
    vy = y_pred - y_pred.mean()
    denom = torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + 1e-8
    return ((vx * vy).sum() / denom).item()


def crop_collate(batch):
    """Batch variable-length sessions (SADT / MPD-DF) by cropping to the shortest one."""
    T = min(x.shape[-1] for x, _ in batch)
    return (torch.stack([x[..., :T] for x, _ in batch]),
            torch.stack([y[:T] for _, y in batch]))


class RandomCropDataset(Dataset):
    """
    Training on random fixed-length windows of whole sessions.

    Sessions are first z-scored over their full length by the base dataset (the
    paper's sequence-level normalisation), then a random crop of `crop_len`
    windows is taken. Each epoch draws ceil(T_i / (crop_len - burn_in)) crops
    from session i, so every window is covered about once per epoch. Sessions
    shorter than `crop_len` are returned whole.
    """

    def __init__(self, sessions: Dataset, crop_len: int, burn_in: int = 0):
        if crop_len <= burn_in + 1:
            raise ValueError("train_crop_len must exceed burn_in + 1")
        self.sessions = sessions
        self.crop_len = crop_len
        lengths = [sessions[i][1].shape[0] for i in range(len(sessions))]
        stride = crop_len - burn_in
        self.index = [i for i, T in enumerate(lengths) for _ in range(max(1, math.ceil(T / stride)))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, j):
        x, y = self.sessions[self.index[j]]
        T = y.shape[0]
        if T <= self.crop_len:
            return x, y
        s = int(torch.randint(0, T - self.crop_len + 1, (1,)))
        return x[..., s:s + self.crop_len], y[s:s + self.crop_len]


def load_coords(path: str) -> np.ndarray:
    coords = np.loadtxt(path, delimiter=None)
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise ValueError("coords must have shape (C, 2) or (C, 3)")
    return coords


def load_yaml_config(path: str) -> dict:
    """Load YAML config file."""
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, args):
    if args.warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    if args.warmup_epochs >= args.epochs:
        return LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.epochs)
    warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs))
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def get_subject_split(pairs, extract_id_fn, n_folds=5, fold_idx=0, seed=2025, val_fraction=0.0):
    """
    Split data by subject ID (leave-one-subject-out).
    Returns train / inner-validation / test pairs and the test subject IDs.
    The test subject(s) of the fold are never used for training or model selection;
    early stopping and checkpoint selection use ceil(val_fraction * #train subjects)
    other training subjects. With val_fraction=0 selection falls back to the test
    subject (optimistic; kept only for comparison with earlier runs).
    """
    subj_map: Dict[int, List[Tuple[str, str]]] = {}
    for p in pairs:
        sid = extract_id_fn(p[0])
        subj_map.setdefault(sid, []).append(p)

    unique_subjs = sorted(subj_map.keys())
    print(f"Found {len(unique_subjs)} unique subjects: {unique_subjs}")
    if n_folds > len(unique_subjs):
        raise ValueError(
            f"n_folds={n_folds} > {len(unique_subjs)} subjects found; some folds would be empty. "
            f"For LOSO set --n-folds {len(unique_subjs)} (check the subject-ID parsing of the file names)."
        )

    rng = np.random.RandomState(seed)
    rng.shuffle(unique_subjs)

    folds = np.array_split(unique_subjs, n_folds)
    test_subjs = sorted(folds[fold_idx % n_folds].tolist())
    train_subjs = [s for s in unique_subjs if s not in test_subjs]

    n_val = int(np.ceil(val_fraction * len(train_subjs))) if val_fraction > 0 else 0
    if n_val >= len(train_subjs):
        raise ValueError(f"val_fraction={val_fraction} leaves no training subjects")
    if n_val > 0:
        val_rng = np.random.RandomState(seed + 1 + fold_idx)
        val_subjs = sorted(val_rng.choice(train_subjs, n_val, replace=False).tolist())
        train_subjs = [s for s in train_subjs if s not in val_subjs]
    else:
        val_subjs = test_subjs
        print("WARNING: val_fraction=0 -> checkpoint selection on the test subject (optimistic).")

    print(f"Fold {fold_idx}/{n_folds} - Test subjects: {test_subjs} | "
          f"Inner-val subjects: {val_subjs} | Train subjects: {len(train_subjs)}")

    def to_pairs(subjs):
        return [p for s in subjs for p in subj_map[s]]

    return to_pairs(train_subjs), to_pairs(val_subjs), to_pairs(test_subjs), test_subjs


def build_datasets(args, channel_indices):
    """Build train, inner-validation and test datasets based on args.dataset."""
    dataset_name = getattr(args, "dataset", "seedvig")

    if dataset_name == "seedvig":
        eeg_dir = os.path.join(args.data_root, "EEG_Feature_5Bands")
        perclos_dir = os.path.join(args.data_root, "PERCLOS")
        pairs = build_file_pairs(eeg_dir, perclos_dir)
        assert len(pairs) > 0, "No EEG-PERCLOS file pairs found."
        train_pairs, val_pairs, test_pairs, test_subjs = get_subject_split(
            pairs, extract_subject_id,
            n_folds=args.n_folds, fold_idx=args.fold, seed=args.seed,
            val_fraction=args.val_fraction,
        )
        train_ds = SEEDVIGSequenceDataset(
            train_pairs, normalize=True, cache=args.cache,
            augment=args.augment, channel_indices=channel_indices,
        )
        val_ds = SEEDVIGSequenceDataset(
            val_pairs, normalize=True, cache=args.cache,
            augment=False, channel_indices=channel_indices,
        )
        test_ds = SEEDVIGSequenceDataset(
            test_pairs, normalize=True, cache=args.cache,
            augment=False, channel_indices=channel_indices,
        )

    elif dataset_name == "sadt":
        pairs = build_sadt_file_pairs(args.data_root)
        assert len(pairs) > 0, "No SADT feature-label pairs found."
        train_pairs, val_pairs, test_pairs, test_subjs = get_subject_split(
            pairs, extract_subject_id_sadt,
            n_folds=args.n_folds, fold_idx=args.fold, seed=args.seed,
            val_fraction=args.val_fraction,
        )
        train_ds = SADTDataset(
            train_pairs, normalize=True, cache=args.cache,
            augment=args.augment, channel_indices=channel_indices,
        )
        val_ds = SADTDataset(
            val_pairs, normalize=True, cache=args.cache,
            augment=False, channel_indices=channel_indices,
        )
        test_ds = SADTDataset(
            test_pairs, normalize=True, cache=args.cache,
            augment=False, channel_indices=channel_indices,
        )

    elif dataset_name == "mpddf":
        pairs = build_mpddf_file_pairs(args.data_root)
        assert len(pairs) > 0, "No MPD-DF feature-label pairs found."
        train_pairs, val_pairs, test_pairs, test_subjs = get_subject_split(
            pairs, extract_subject_id_mpddf,
            n_folds=args.n_folds, fold_idx=args.fold, seed=args.seed,
            val_fraction=args.val_fraction,
        )
        train_ds = MPDDFDataset(
            train_pairs, normalize=True, cache=args.cache,
            augment=args.augment, channel_indices=channel_indices,
        )
        val_ds = MPDDFDataset(
            val_pairs, normalize=True, cache=args.cache,
            augment=False, channel_indices=channel_indices,
        )
        test_ds = MPDDFDataset(
            test_pairs, normalize=True, cache=args.cache,
            augment=False, channel_indices=channel_indices,
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: seedvig, sadt, mpddf")

    return train_ds, val_ds, test_ds, test_subjs


# ---------------------------------------------------------------------------
# Adjacency
# ---------------------------------------------------------------------------

def parse_channel_names(value):
    """Channel names from YAML (list) or CLI (comma-separated string)."""
    if not value:
        return None
    if isinstance(value, str):
        value = [v for v in value.split(",") if v.strip()]
    return [str(v).strip() for v in value]


def build_adjacency(args, num_channels, channel_indices):
    """
    Physical kNN graph on electrode coordinates (Sec 4.2). Coordinates come from
    --coords-path, or from --channel-names via the standard 10-05 montage.
    --dense-graph gives the fully connected 'w/o Topo (dense)' ablation.
    """
    if args.dense_graph:
        print("Using full adjacency (complete graph): 'w/o Topo (dense)' ablation.")
        return build_full_adjacency(num_channels)

    channel_names = parse_channel_names(args.channel_names)
    if args.coords_path:
        coords = load_coords(args.coords_path)
        source = args.coords_path
    elif channel_names:
        coords = coords_from_channel_names(channel_names)
        source = "standard_1005 montage: " + ",".join(
            channel_names if channel_indices is None else [channel_names[i] for i in channel_indices]
        )
    else:
        raise ValueError(
            "HASTE needs electrode positions for the physical kNN graph: set channel_names "
            "in the config (or --channel-names / --coords-path). Use --dense-graph only for "
            "the 'w/o Topo (dense)' ablation."
        )

    if channel_indices is not None:
        coords = coords[channel_indices]
    if coords.shape[0] != num_channels:
        raise ValueError(f"got {coords.shape[0]} electrode positions but num_channels={num_channels}")

    adj = build_knn_adjacency(coords, k=args.adj_k, self_loop=True)
    avg_deg = (adj.sum() - num_channels).item() / num_channels
    print(f"Physical kNN graph (k={args.adj_k}, symmetrized + self-loops, "
          f"avg degree={avg_deg:.2f}) from {source}")
    return adj


# ---------------------------------------------------------------------------
# TensorBoard
# ---------------------------------------------------------------------------

def get_writer(args):
    """Create TensorBoard SummaryWriter if available."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        dataset_name = getattr(args, "dataset", "seedvig")
        log_dir = os.path.join("runs", f"{dataset_name}_fold{args.fold}_seed{args.seed}")
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logging to: {log_dir}")
        return writer
    except ImportError:
        print("TensorBoard not available, skipping logging.")
        return None


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    lambda_pred: float,
    lambda_band: float,
    lambda_graph: float,
    device: torch.device,
) -> Tuple[float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_abs = 0.0
    total_sq = 0.0
    n_points = 0
    y_true_all = []
    y_pred_all = []
    session_corrs = []
    n = 0

    with torch.no_grad():
        for X_val, y_val in loader:
            X_val = X_val.to(device)
            y_val = y_val.to(device)

            y_final, y_seq, alpha, P_func, g, P_t = model(X_val)
            loss_total, _, _, _, _ = hast_loss_fn(
                y_true=y_val, y_final=y_final, y_seq=y_seq,
                alpha=alpha, P_func=P_func, g=g,
                lambda_pred=lambda_pred,
                lambda_band=lambda_band,
                lambda_graph=lambda_graph,
            )

            bs = X_val.size(0)
            total_loss += loss_total.item() * bs
            n += bs

            # Streaming regression: MAE / RMSE over every update y_hat_t (Sec. 5.2)
            err = y_seq - y_val
            total_abs += err.abs().sum().item()
            total_sq += (err ** 2).sum().item()
            n_points += err.numel()

            # PCC per session, then averaged (per-subject PCC; SADT averages sessions)
            for i in range(bs):
                session_corrs.append(compute_corr(y_val[i], y_seq[i]))

            y_true_all.append(y_val[:, -1].detach().cpu())
            y_pred_all.append(y_final.detach().cpu())

    loss_avg = total_loss / max(n, 1)
    mae_avg = total_abs / max(n_points, 1)
    rmse_avg = (total_sq / max(n_points, 1)) ** 0.5

    corr_final = compute_corr(torch.cat(y_true_all), torch.cat(y_pred_all))
    corr_seq = float(np.mean(session_corrs)) if session_corrs else 0.0

    return loss_avg, mae_avg, rmse_avg, corr_final, corr_seq


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def train(model, train_loader, val_loader, args, device, writer=None):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args)
    # torch.amp.GradScaler does not exist before torch 2.3 and takes `device`, not
    # `device_type`; loss scaling is only needed for CUDA fp16, so use the CUDA scaler.
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    burn_in = args.burn_in if args.train_crop_len > 0 else 0

    higher_is_better = args.checkpoint_metric in ("corr_seq", "corr_final")
    best_metric = -float("inf") if higher_is_better else float("inf")
    best_state = None
    epochs_no_improve = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        n_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False)
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                y_final, y_seq, alpha, P_func, g, P_t = model(X_batch)
                if 0 < burn_in < y_seq.shape[1] - 1:
                    # First windows of a crop only warm up the GRU state / CLTA window
                    y_seq, y_batch = y_seq[:, burn_in:], y_batch[:, burn_in:]
                    alpha, P_func, g = alpha[:, burn_in:], P_func[:, burn_in:], g[:, burn_in:]
                loss_total, loss_reg, loss_pred, loss_band, loss_graph = hast_loss_fn(
                    y_true=y_batch, y_final=y_final, y_seq=y_seq,
                    alpha=alpha, P_func=P_func, g=g,
                    lambda_pred=args.lambda_pred,
                    lambda_band=args.lambda_band,
                    lambda_graph=args.lambda_graph,
                )
            scaler.scale(loss_total).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            bs = X_batch.size(0)
            total_train_loss += loss_total.item() * bs
            n_train += bs
            global_step += 1

            pbar.set_postfix(loss=f"{loss_total.item():.4f}")

            if writer is not None:
                writer.add_scalar("train/loss_step", loss_total.item(), global_step)

        train_loss_epoch = total_train_loss / max(n_train, 1)

        val_loss, val_mae, val_rmse, corr_final, corr_seq = evaluate(
            model, val_loader,
            args.lambda_pred, args.lambda_band, args.lambda_graph, device,
        )

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss_epoch:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"MAE={val_mae:.4f} | "
            f"RMSE={val_rmse:.4f} | "
            f"PCC_final={corr_final:.4f} | "
            f"PCC_seq={corr_seq:.4f} | "
            f"lr={current_lr:.2e}"
        )

        # TensorBoard logging
        if writer is not None:
            writer.add_scalar("train/loss_epoch", train_loss_epoch, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/mae", val_mae, epoch)
            writer.add_scalar("val/rmse", val_rmse, epoch)
            writer.add_scalar("val/pcc_final", corr_final, epoch)
            writer.add_scalar("val/pcc_seq", corr_seq, epoch)
            writer.add_scalar("lr", current_lr, epoch)

        # Checkpoint logic
        metric_map = {
            "mae": (val_mae, False),
            "loss": (val_loss, False),
            "rmse": (val_rmse, False),
            "corr_seq": (corr_seq, True),
            "corr_final": (corr_final, True),
        }
        metric, higher = metric_map.get(args.checkpoint_metric, (corr_seq, True))
        better = metric > best_metric if higher else metric < best_metric

        if better:
            best_metric = metric
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if writer is not None:
        writer.close()

    return model, best_metric


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/evaluate HASTE for streaming EEG fatigue monitoring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file. CLI args override config values.")

    # Dataset
    parser.add_argument("--dataset", type=str, default="seedvig",
                        choices=["seedvig", "sadt", "mpddf"])
    parser.add_argument("--data-root", type=str, default="./data_seed_vig")

    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-pred", type=float, default=0.1)
    parser.add_argument("--lambda-band", type=float, default=0.01)
    parser.add_argument("--lambda-graph", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--checkpoint-metric", type=str, default="corr_seq",
                        choices=["mae", "loss", "rmse", "corr_seq", "corr_final"])
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of training subjects held out per fold for early stopping / "
                             "checkpoint selection (0 = select on the test subject, optimistic).")
    parser.add_argument("--train-crop-len", type=int, default=128,
                        help="Train on random crops of this many windows (0 = whole sessions).")
    parser.add_argument("--burn-in", type=int, default=16,
                        help="Leading windows of each crop excluded from the loss (state warm-up).")

    # Model architecture
    parser.add_argument("--num-channels", type=int, default=17)
    parser.add_argument("--num-bands", type=int, default=5)
    parser.add_argument("--spatial-d-model", type=int, default=64)
    parser.add_argument("--spatial-heads", type=int, default=4)
    parser.add_argument("--spatial-layers", type=int, default=2)
    parser.add_argument("--temporal-d-model", type=int, default=128)
    parser.add_argument("--temporal-heads", type=int, default=4)
    parser.add_argument("--temporal-layers", type=int, default=1)
    parser.add_argument("--temporal-window", type=int, default=10)
    parser.add_argument("--temporal-ff-dim", type=int, default=512)
    parser.add_argument("--d-r", type=int, default=32)
    parser.add_argument("--d-embed", type=int, default=64)
    parser.add_argument("--d-s", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Adjacency
    parser.add_argument("--channel-names", default=None,
                        help="Comma-separated 10-20 channel names in data order (kNN graph from the "
                             "standard_1005 montage). Usually set in the YAML config.")
    parser.add_argument("--coords-path", type=str, default=None,
                        help="Text file with (C, 2|3) electrode coordinates; overrides --channel-names.")
    parser.add_argument("--adj-k", type=int, default=5, help="Physical kNN degree k.")
    parser.add_argument("--dense-graph", action="store_true",
                        help="Fully connected graph ('w/o Topo (dense)' ablation).")

    # Evaluation
    parser.add_argument("--n-folds", type=int, default=23)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; load checkpoint and evaluate.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for --eval-only mode.")

    # Runtime
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--augment", dest="augment", action="store_true", default=False,
                        help="Feature-noise augmentation (not part of the paper protocol).")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--cache", dest="cache", action="store_true", default=True)
    parser.add_argument("--no-cache", dest="cache", action="store_false")
    parser.add_argument("--channel-keep", type=str, default=None,
                        help="Comma-separated channel indices to keep (1-based)")

    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints")

    # --- Load YAML config as defaults, then parse CLI ---
    # First pass: get --config path
    temp_args, _ = parser.parse_known_args()
    if temp_args.config:
        yaml_cfg = load_yaml_config(temp_args.config)
        # Convert YAML keys: underscores to match argparse dest names
        yaml_defaults = {}
        for k, v in yaml_cfg.items():
            yaml_defaults[k.replace("-", "_")] = v
        parser.set_defaults(**yaml_defaults)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    # Channel selection
    base_channels = args.num_channels
    channel_indices = None
    if args.channel_keep:
        channel_indices = [int(x.strip()) - 1 for x in args.channel_keep.split(",") if x.strip()]
        if not channel_indices:
            channel_indices = None
        elif min(channel_indices) < 0 or max(channel_indices) >= base_channels:
            raise ValueError(f"channel_keep indices must be in range 1..{base_channels}")
    num_channels = len(channel_indices) if channel_indices is not None else base_channels

    # Device
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(f"Device: {device}")

    # Dataset
    dataset_name = getattr(args, "dataset", "seedvig")
    print(f"Dataset: {dataset_name} | Fold: {args.fold}/{args.n_folds}")

    train_ds, val_ds, test_ds, test_subjs = build_datasets(args, channel_indices)
    if args.train_crop_len > 0:
        train_ds = RandomCropDataset(train_ds, args.train_crop_len, args.burn_in)
        print(f"Training on random crops of {args.train_crop_len} windows "
              f"(burn-in {args.burn_in}): {len(train_ds)} crops/epoch")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=crop_collate,
    )
    # One session per batch so held-out sessions are evaluated over their full length
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # Adjacency
    adj = build_adjacency(args, num_channels, channel_indices)

    # Model
    model = HASTModel(
        num_channels=num_channels,
        num_bands=args.num_bands,
        spatial_d_model=args.spatial_d_model,
        spatial_heads=args.spatial_heads,
        spatial_layers=args.spatial_layers,
        temporal_d_model=args.temporal_d_model,
        temporal_heads=args.temporal_heads,
        temporal_layers=args.temporal_layers,
        temporal_window=args.temporal_window,
        temporal_ff_dim=args.temporal_ff_dim,
        d_r=args.d_r,
        d_embed=args.d_embed,
        d_s=args.d_s,
        top_k=args.top_k,
        adjacency=adj,
        dropout=args.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Eval-only mode
    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("--eval-only requires --checkpoint <path>")
        print(f"Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.to(device)
        loss, mae, rmse, pcc_final, pcc_seq = evaluate(
            model, test_loader,
            args.lambda_pred, args.lambda_band, args.lambda_graph, device,
        )
        print(f"Test results (subjects {test_subjs}):")
        print(f"  Loss={loss:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f} | "
              f"PCC_final={pcc_final:.4f} | PCC_seq={pcc_seq:.4f}")
        return

    # Train
    writer = get_writer(args)
    print(f"Training HASTE ({dataset_name}, fold {args.fold})...")
    model, best_metric = train(model, train_loader, val_loader, args, device, writer)

    metric_names = {
        "mae": "MAE", "loss": "Loss", "rmse": "RMSE",
        "corr_seq": "PCC_seq", "corr_final": "PCC_final",
    }
    print(f"Training complete. Best {metric_names.get(args.checkpoint_metric, 'metric')}: {best_metric:.4f}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"hast_{dataset_name}_fold{args.fold}_seed{args.seed}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Best model saved: {ckpt_path}")

    # Test-subject metrics of the selected model
    _, mae, rmse, _, pcc = evaluate(
        model, test_loader, args.lambda_pred, args.lambda_band, args.lambda_graph, device,
    )
    print(f"Test subjects {test_subjs}: PCC={pcc:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f}")

if __name__ == "__main__":
    main()
