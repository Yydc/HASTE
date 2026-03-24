"""
HASTE training and evaluation CLI.

Supports SEED-VIG, SADT, and MPD-DF datasets via YAML config files.

Usage:
    python train.py --config configs/seedvig.yaml --data-root ./data_seed_vig
    python train.py --config configs/sadt.yaml --fold 0
    python train.py --config configs/mpddf.yaml --eval-only --checkpoint checkpoints/hast_mpddf_fold0.pth
"""

import argparse
import copy
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
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
from models import HASTModel, build_full_adjacency, build_knn_gaussian, hast_loss_fn


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

def get_subject_split(pairs, extract_id_fn, n_folds=5, fold_idx=0, seed=2025):
    """
    Split data by subject ID (leave-one-subject-out).
    A subject appears ONLY in train OR val, never both.
    """
    subj_map: Dict[int, List[Tuple[str, str]]] = {}
    for p in pairs:
        sid = extract_id_fn(p[0])
        subj_map.setdefault(sid, []).append(p)

    unique_subjs = sorted(subj_map.keys())
    print(f"Found {len(unique_subjs)} unique subjects: {unique_subjs}")

    rng = np.random.RandomState(seed)
    rng.shuffle(unique_subjs)

    folds = np.array_split(unique_subjs, n_folds)
    val_subjs = set(folds[fold_idx % n_folds])
    train_subjs = [s for s in unique_subjs if s not in val_subjs]

    print(f"Fold {fold_idx}/{n_folds} - Val subjects: {sorted(val_subjs)}")

    train_pairs = [p for s in train_subjs for p in subj_map[s]]
    val_pairs = [p for s in val_subjs for p in subj_map[s]]
    return train_pairs, val_pairs


def build_datasets(args, channel_indices):
    """Build train and val datasets based on args.dataset."""
    dataset_name = getattr(args, "dataset", "seedvig")

    if dataset_name == "seedvig":
        eeg_dir = os.path.join(args.data_root, "EEG_Feature_5Bands")
        perclos_dir = os.path.join(args.data_root, "PERCLOS")
        pairs = build_file_pairs(eeg_dir, perclos_dir)
        assert len(pairs) > 0, "No EEG-PERCLOS file pairs found."
        train_pairs, val_pairs = get_subject_split(
            pairs, extract_subject_id,
            n_folds=args.n_folds, fold_idx=args.fold, seed=args.seed,
        )
        train_ds = SEEDVIGSequenceDataset(
            train_pairs, normalize=True, cache=args.cache,
            augment=args.augment, channel_indices=channel_indices,
        )
        val_ds = SEEDVIGSequenceDataset(
            val_pairs, normalize=True, cache=args.cache,
            augment=False, channel_indices=channel_indices,
        )

    elif dataset_name == "sadt":
        pairs = build_sadt_file_pairs(args.data_root)
        assert len(pairs) > 0, "No SADT feature-label pairs found."
        train_pairs, val_pairs = get_subject_split(
            pairs, extract_subject_id_sadt,
            n_folds=args.n_folds, fold_idx=args.fold, seed=args.seed,
        )
        train_ds = SADTDataset(
            train_pairs, normalize=True, cache=args.cache,
            augment=args.augment, channel_indices=channel_indices,
        )
        val_ds = SADTDataset(
            val_pairs, normalize=True, cache=args.cache,
            augment=False, channel_indices=channel_indices,
        )

    elif dataset_name == "mpddf":
        pairs = build_mpddf_file_pairs(args.data_root)
        assert len(pairs) > 0, "No MPD-DF feature-label pairs found."
        train_pairs, val_pairs = get_subject_split(
            pairs, extract_subject_id_mpddf,
            n_folds=args.n_folds, fold_idx=args.fold, seed=args.seed,
        )
        train_ds = MPDDFDataset(
            train_pairs, normalize=True, cache=args.cache,
            augment=args.augment, channel_indices=channel_indices,
        )
        val_ds = MPDDFDataset(
            val_pairs, normalize=True, cache=args.cache,
            augment=False, channel_indices=channel_indices,
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: seedvig, sadt, mpddf")

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Adjacency
# ---------------------------------------------------------------------------

def build_adjacency(args, num_channels, channel_indices):
    """Build adjacency matrix from coords or default to full graph."""
    if args.coords_path:
        coords = load_coords(args.coords_path)
        if coords.shape[0] < num_channels:
            raise ValueError(
                f"coords has {coords.shape[0]} rows but need at least {num_channels} channels"
            )
        if channel_indices is not None:
            coords = coords[channel_indices]
        elif coords.shape[0] != num_channels:
            raise ValueError(
                f"coords has {coords.shape[0]} rows but num_channels={num_channels}"
            )
        if args.coords_normalize:
            mean = coords.mean(axis=0, keepdims=True)
            std = coords.std() + 1e-6
            coords = (coords - mean) / std
        adj = build_knn_gaussian(coords, k=args.adj_k, sigma=args.adj_sigma, self_loop=True)
        print(f"k-NN adjacency (k={args.adj_k}, sigma={args.adj_sigma}, "
              f"adj stats: min={adj.min():.4f}, max={adj.max():.4f}, mean={adj.mean():.4f})")
    else:
        adj = build_full_adjacency(num_channels)
        print("Using full adjacency (complete graph).")
    return adj


# ---------------------------------------------------------------------------
# TensorBoard
# ---------------------------------------------------------------------------

def get_writer(args):
    """Create TensorBoard SummaryWriter if available."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        dataset_name = getattr(args, "dataset", "seedvig")
        log_dir = os.path.join("runs", f"{dataset_name}_fold{args.fold}")
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
    total_mae = 0.0
    total_rmse = 0.0
    y_true_all = []
    y_pred_all = []
    y_true_seq_all = []
    y_pred_seq_all = []
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
            y_val_final = y_val[:, -1] if y_val.dim() > 1 else y_val
            total_mae += torch.abs(y_final - y_val_final).sum().item()
            total_rmse += torch.sqrt(torch.mean((y_final - y_val_final) ** 2)).item() * bs
            n += bs

            y_true_all.append(y_val.detach().cpu())
            y_pred_all.append(y_final.detach().cpu())
            y_true_seq_all.append(y_val.detach().cpu())
            y_pred_seq_all.append(y_seq.detach().cpu())

    loss_avg = total_loss / max(n, 1)
    mae_avg = total_mae / max(n, 1)
    rmse_avg = total_rmse / max(n, 1)

    y_true_cat = torch.cat(y_true_all, dim=0)
    y_pred_cat = torch.cat(y_pred_all, dim=0)
    y_true_final = y_true_cat[:, -1] if y_true_cat.dim() > 1 else y_true_cat
    corr_final = compute_corr(y_true_final, y_pred_cat)

    y_true_seq_cat = torch.cat(y_true_seq_all, dim=0)
    y_pred_seq_cat = torch.cat(y_pred_seq_all, dim=0)
    corr_seq = compute_corr(y_true_seq_cat, y_pred_seq_cat)

    return loss_avg, mae_avg, rmse_avg, corr_final, corr_seq


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def train(model, train_loader, val_loader, args, device, writer=None):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args)
    scaler = torch.amp.GradScaler(
        device_type="cuda" if device.type == "cuda" else "cpu",
        enabled=args.amp,
    )

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
        metric, higher = metric_map.get(args.checkpoint_metric, (corr_final, True))
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
    parser.add_argument("--checkpoint-metric", type=str, default="corr_final",
                        choices=["mae", "loss", "rmse", "corr_seq", "corr_final"])

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
    parser.add_argument("--coords-path", type=str, default=None)
    parser.add_argument("--coords-normalize", action="store_true", dest="coords_normalize", default=True)
    parser.add_argument("--no-coords-normalize", dest="coords_normalize", action="store_false")
    parser.add_argument("--adj-k", type=int, default=5)
    parser.add_argument("--adj-sigma", type=float, default=1.0)

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
    parser.add_argument("--augment", dest="augment", action="store_true", default=True)
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

    train_ds, val_ds = build_datasets(args, channel_indices)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

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
            model, val_loader,
            args.lambda_pred, args.lambda_band, args.lambda_graph, device,
        )
        print(f"Evaluation results:")
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
    ckpt_path = os.path.join(args.output_dir, f"hast_{dataset_name}_fold{args.fold}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Best model saved: {ckpt_path}")


if __name__ == "__main__":
    main()
