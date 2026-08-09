"""
train_forward_tcn.py
Training pipeline for the DirectSequenceTCN forward model.

Loss composition (Torun TMTT 2020 + Liu Micromachines 2025 informed):
   L_total = 1.0 * L1(Re, Im)              # complex-valued reconstruction
           + 0.1 * Huber(clamped_dB)       # passband shape, magnitude-aware
           + 0.5 * weighted_L2(Re, Im)     # element-weighted, attends to weak modes
           + 50.0 * passivity_loss(S)      # soft Torun-PEL, on all 401 freqs
           + 0.05 * causality_loss(S)      # soft Torun-CEL on diagonal reflections

Schedule: AdamW + Cosine Annealing with Warm Restarts (SGDR).
   Reference: Loshchilov & Hutter, ICLR 2017, arXiv:1608.03983.
   Periodic LR restarts help escape local minima in non-convex loss landscapes.

Inference: SVD passivity projection is applied at inference time so all returned
predictions are guaranteed passive (Grivet-Talocia & Gustavsen 2016, eq. 11.86).
We report metrics BOTH before and after this projection so we can see how much
the projection costs us in accuracy (typically 0.05-0.20 dB).

Logging style (per request): each epoch prints on its own line with all metrics.
No dynamic single-line progress bar; this makes log files much easier to grep
and lets the user watch a long-running training without losing the history.

Run:
   python train_forward_tcn.py --dry-run   # smoke test (3 epochs, 200 train)
   python train_forward_tcn.py             # full training
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # No display required
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forward_direct_sequence_resnet_v2 import (  # noqa: E402
    DirectSequenceTCN,
    passivity_loss,
    causality_loss,
    passivity_project_svd,
)

# CUDA / TF32 settings: enable speedup on Ampere+ GPUs without precision loss
# inside the conv backbone. The rational layer is float64/complex128 regardless.
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path.home() / "mece_project_inverse_model" / "Generative_Inverse_Design_of_High-Speed_Interconnects"
DATA_PT = PROJECT_ROOT / "data" / "processed" / "Universal-Diff-SI-Array" / "diff_pair_dataset.pt"
SPLITS_PT = PROJECT_ROOT / "sandbox_v1" / "data" / "splits.pt"
RUNS_DIR = PROJECT_ROOT / "sandbox_v1" / "models" / "forward_runs"

# Noise floor for dB metrics (Hillebrecht IEEE TEMC 2024)
NOISE_FLOOR_DB = -55.0


# -----------------------------------------------------------------------------
# Dataset wrapper
# -----------------------------------------------------------------------------
class DiffPairDataset(Dataset):
    """One differential pair per item.  Inputs are pre-z-scored in the .pt."""

    def __init__(self, payload: dict, indices: np.ndarray):
        self.x_local = payload["X_local"][indices]
        self.x_global = payload["X_global"][indices]
        self.x_context = payload["X_context"][indices]
        self.y_real = payload["Y_real"][indices]
        self.y_imag = payload["Y_imag"][indices]
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        S = torch.complex(self.y_real[i].to(torch.float64),
                          self.y_imag[i].to(torch.float64))
        return self.x_local[i], self.x_global[i], self.x_context[i], S


# -----------------------------------------------------------------------------
# Element weights for mode-conversion attention
# -----------------------------------------------------------------------------
def compute_element_weights(payload: dict, train_idx: np.ndarray,
                             n_sample: int = 500, seed: int = 0,
                             floor: float = 1e-3):
    """
    Per-(i,j) loss weights inversely proportional to mean |S_ij|^2.
    Normalized so mean weight = 1.0 (keeps overall loss scale interpretable).

    Without this weighting the dominant Sdd11, Sdd21 dwarf the weak mode-
    conversion elements (Sdc, Scd), and the model never bothers fitting them.
    Yield-aware inverse design needs accurate mode conversion (EMC compliance).
    """
    rng = np.random.default_rng(seed)
    pick = rng.choice(train_idx, size=min(n_sample, len(train_idx)), replace=False)
    yr = payload["Y_real"][pick].to(torch.float64)
    yi = payload["Y_imag"][pick].to(torch.float64)
    mag_sq = yr ** 2 + yi ** 2
    mean_per_elem = mag_sq.mean(dim=(0, 1))
    weight = 1.0 / mean_per_elem.clamp_min(floor)
    weight = weight * (16.0 / weight.sum())
    return weight, mean_per_elem


# -----------------------------------------------------------------------------
# Composite loss
# -----------------------------------------------------------------------------
def composite_loss(pred: torch.Tensor, target: torch.Tensor,
                    weight44: torch.Tensor,
                    w_pass: float = 50.0,
                    w_caus: float = 0.05) -> dict:
    """
    L_total = L1(Re,Im) + 0.1 * Huber(clamped_dB) + 0.5 * weighted_L2
              + w_pass * passivity_loss + w_caus * causality_loss

    Default passivity weight is 50 (vs the user's prior 5).  Liu Micromachines 2025
    reports this magnitude is needed for max sigma to converge near 1.0.
    """
    eps = 1e-12

    # ---- L1 on real and imaginary parts ----
    # Robust to outlier samples; gentler gradients than L2 on dominant elements.
    l_linear = (pred.real - target.real).abs().mean() + \
               (pred.imag - target.imag).abs().mean()

    # ---- dB-domain Huber loss with noise-floor clamp ----
    # Below the noise floor (-55 dB), simulator mesh noise dominates; we don't
    # want the model wasting capacity fitting that.
    p_db = 20.0 * torch.log10(pred.abs().clamp_min(eps))
    t_db = 20.0 * torch.log10(target.abs().clamp_min(eps))
    p_db_c = p_db.clamp_min(NOISE_FLOOR_DB)
    t_db_c = t_db.clamp_min(NOISE_FLOOR_DB)
    l_db = torch.nn.functional.huber_loss(p_db_c, t_db_c, delta=2.0)

    # ---- Element-weighted L2 (mode conversion attention) ----
    diff_sq = (pred.real - target.real) ** 2 + (pred.imag - target.imag) ** 2
    l_weighted = (diff_sq * weight44.view(1, 1, 4, 4)).mean()

    # ---- Soft passivity (Torun PEL surrogate, all frequencies) ----
    l_pass = passivity_loss(pred)

    # ---- Soft causality (Torun CEL surrogate, diagonal elements only) ----
    l_caus = causality_loss(pred)

    total = (1.0 * l_linear
             + 0.1 * l_db
             + 0.5 * l_weighted
             + w_pass * l_pass
             + w_caus * l_caus)

    return {
        "loss": total,
        "l_linear": l_linear.detach().item(),
        "l_db": l_db.detach().item(),
        "l_weighted": l_weighted.detach().item(),
        "l_pass": l_pass.detach().item(),
        "l_caus": l_caus.detach().item(),
    }


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
@torch.no_grad()
def passband_mae_db(pred, target, i, j, floor_db=NOISE_FLOOR_DB):
    """
    Mean absolute dB error on element (i, j), masked to frequencies where the
    TRUE response is above the noise floor.  Reported as our primary metric
    because it cleanly excludes regions where the dataset itself is noise.
    """
    eps = 1e-12
    p_db = 20.0 * torch.log10(pred[..., i, j].abs() + eps)
    t_db = 20.0 * torch.log10(target[..., i, j].abs() + eps)
    mask = t_db > floor_db
    if mask.sum() == 0:
        return float("nan")
    return (p_db[mask] - t_db[mask]).abs().mean().item()


@torch.no_grad()
def passivity_max_sigma(pred):
    """Maximum singular value across all (batch, freq).  Should be <= 1."""
    sv = torch.linalg.svdvals(pred)
    return sv.max().item()


# -----------------------------------------------------------------------------
# Per-line logger (replaces single-line progress bar per user's request)
# -----------------------------------------------------------------------------
class EpochLogger:
    """
    Emits one line per epoch with all key metrics.  Format:

      [Epoch 042/200] tr=0.842 vl=0.612 | S11=1.87dB S21=2.41dB avg=2.14dB | sig=1.04 caus=4.3e-04 | lr=3.5e-04 t=8.4s  [BEST]

    Each line is timestamped to a CSV for downstream analysis.
    """

    def __init__(self, total_epochs: int):
        self.total = total_epochs

    def log(self, epoch: int,
            train_stats: dict, val_stats: dict,
            cur_lr: float, dt: float,
            is_best: bool):
        avg_db = 0.5 * (val_stats["mae_sdd11_db"] + val_stats["mae_sdd21_db"])
        avg_db_proj = 0.5 * (val_stats["mae_sdd11_db_proj"]
                              + val_stats["mae_sdd21_db_proj"])
        marker = "  [BEST]" if is_best else ""
        line = (
            f"[Epoch {epoch:03d}/{self.total}] "
            f"tr={train_stats['loss']:.3f} vl={val_stats['loss']:.3f}"
            f" | S11={val_stats['mae_sdd11_db']:5.2f}dB"
            f" S21={val_stats['mae_sdd21_db']:5.2f}dB"
            f" avg={avg_db:5.2f}dB"
            f" (proj avg={avg_db_proj:5.2f}dB)"
            f" | sig={val_stats['sigma_max']:.3f}"
            f" caus={val_stats['l_caus']:.1e}"
            f" | lr={cur_lr:.1e} t={dt:.1f}s"
            f"{marker}"
        )
        print(line, flush=True)


# -----------------------------------------------------------------------------
# Train and eval epoch loops
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, device, weight44, optimizer,
                     w_pass: float, w_caus: float):
    model.train()
    acc = {"loss": 0.0, "l_linear": 0.0, "l_db": 0.0, "l_weighted": 0.0,
           "l_pass": 0.0, "l_caus": 0.0,
           "mae_sdd11_db": 0.0, "mae_sdd21_db": 0.0, "n": 0}

    for xl, xg, xc, S_tgt in loader:
        xl = xl.to(device, non_blocking=True)
        xg = xg.to(device, non_blocking=True)
        xc = xc.to(device, non_blocking=True)
        S_tgt = S_tgt.to(device, non_blocking=True)
        B = xl.shape[0]

        pred = model(xl, xg, xc)
        ld = composite_loss(pred, S_tgt, weight44, w_pass=w_pass, w_caus=w_caus)

        optimizer.zero_grad(set_to_none=True)
        ld["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        acc["loss"] += ld["loss"].detach().item() * B
        acc["l_linear"] += ld["l_linear"] * B
        acc["l_db"] += ld["l_db"] * B
        acc["l_weighted"] += ld["l_weighted"] * B
        acc["l_pass"] += ld["l_pass"] * B
        acc["l_caus"] += ld["l_caus"] * B
        acc["mae_sdd11_db"] += passband_mae_db(pred, S_tgt, 0, 0) * B
        acc["mae_sdd21_db"] += passband_mae_db(pred, S_tgt, 1, 0) * B
        acc["n"] += B

    n = acc["n"]
    return {k: (v / n if k != "n" else v) for k, v in acc.items()}


@torch.no_grad()
def eval_one_epoch(model, loader, device, weight44,
                    w_pass: float, w_caus: float):
    """
    Validation loop.  Reports passband MAE both BEFORE and AFTER SVD passivity
    projection.  The projected metrics are what we'd actually deliver to the
    inverse model; the unprojected ones tell us how big the projection cost is.
    """
    model.eval()
    acc = {"loss": 0.0, "l_pass": 0.0, "l_caus": 0.0,
           "mae_sdd11_db": 0.0, "mae_sdd21_db": 0.0,
           "mae_sdd11_db_proj": 0.0, "mae_sdd21_db_proj": 0.0,
           "sigma_max": 0.0, "n": 0}

    for xl, xg, xc, S_tgt in loader:
        xl = xl.to(device, non_blocking=True)
        xg = xg.to(device, non_blocking=True)
        xc = xc.to(device, non_blocking=True)
        S_tgt = S_tgt.to(device, non_blocking=True)
        B = xl.shape[0]

        pred = model(xl, xg, xc)
        ld = composite_loss(pred, S_tgt, weight44, w_pass=w_pass, w_caus=w_caus)

        # Pre-projection metrics
        mae11 = passband_mae_db(pred, S_tgt, 0, 0)
        mae21 = passband_mae_db(pred, S_tgt, 1, 0)
        sig_max = passivity_max_sigma(pred)

        # Post-projection metrics: what the inverse model will actually see
        pred_proj = passivity_project_svd(pred)
        mae11_proj = passband_mae_db(pred_proj, S_tgt, 0, 0)
        mae21_proj = passband_mae_db(pred_proj, S_tgt, 1, 0)

        acc["loss"] += ld["loss"].item() * B
        acc["l_pass"] += ld["l_pass"] * B
        acc["l_caus"] += ld["l_caus"] * B
        acc["mae_sdd11_db"] += mae11 * B
        acc["mae_sdd21_db"] += mae21 * B
        acc["mae_sdd11_db_proj"] += mae11_proj * B
        acc["mae_sdd21_db_proj"] += mae21_proj * B
        acc["sigma_max"] = max(acc["sigma_max"], sig_max)
        acc["n"] += B

    n = acc["n"]
    return {k: (v / n if k not in ("sigma_max", "n") else v)
            for k, v in acc.items()}


# -----------------------------------------------------------------------------
# Diagnostic plots
# -----------------------------------------------------------------------------
@torch.no_grad()
def plot_predictions(model, dataset, device, freqs_hz, out_path,
                      n_show: int = 6, seed: int = 0):
    """Six val samples, Sdd11 + Sdd21 panels, target vs prediction in dB."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(dataset), size=min(n_show, len(dataset)), replace=False)
    model.eval()
    fig, axes = plt.subplots(2, n_show, figsize=(3.4 * n_show, 6), sharex=True)
    f_ghz = freqs_hz / 1e9

    for col, idx in enumerate(pick):
        xl, xg, xc, S_tgt = dataset[int(idx)]
        xl = xl.unsqueeze(0).to(device)
        xg = xg.unsqueeze(0).to(device)
        xc = xc.unsqueeze(0).to(device)
        S_pred = model(xl, xg, xc).squeeze(0)
        # Apply SVD passivity projection for the displayed predictions
        S_pred_proj = passivity_project_svd(S_pred.unsqueeze(0)).squeeze(0).cpu()

        for row, (i, j, label) in enumerate([(0, 0, "Sdd11"), (1, 0, "Sdd21")]):
            ax = axes[row, col]
            ax.plot(f_ghz,
                     20 * np.log10(S_tgt[:, i, j].abs().numpy() + 1e-12),
                     "b-", linewidth=1,
                     label="target" if (row == 0 and col == 0) else None)
            ax.plot(f_ghz,
                     20 * np.log10(S_pred_proj[:, i, j].abs().numpy() + 1e-12),
                     "r--", linewidth=1,
                     label="TCN+SVDproj" if (row == 0 and col == 0) else None)
            ax.axhline(NOISE_FLOOR_DB, color="gray", linestyle=":", linewidth=0.6)
            ax.set_ylabel(f"|{label}| [dB]")
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(f"sample idx {int(idx)}", fontsize=9)
                if col == 0:
                    ax.legend(fontsize=8)
            else:
                ax.set_xlabel("Frequency [GHz]")

    plt.suptitle("Validation: TCN + SVD passivity projection vs target "
                  "(dotted = noise floor -55 dB)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_training_curves(history: dict, out_path: Path):
    """Loss, MAE, passivity, and component-loss breakdown."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Loss curves
    axes[0, 0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0, 0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("composite loss")
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_title("Loss")

    # MAE curves (pre- and post-projection)
    axes[0, 1].plot(history["epoch"], history["val_sdd11"], label="Sdd11 (raw)")
    axes[0, 1].plot(history["epoch"], history["val_sdd21"], label="Sdd21 (raw)")
    axes[0, 1].plot(history["epoch"], history["val_sdd11_proj"],
                     label="Sdd11 (proj)", linestyle="--")
    axes[0, 1].plot(history["epoch"], history["val_sdd21_proj"],
                     label="Sdd21 (proj)", linestyle="--")
    axes[0, 1].axhline(2.0, color="green", linestyle=":", linewidth=0.7,
                       label="2 dB target")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("passband MAE [dB]")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_title("Validation passband MAE (raw vs SVD-projected)")

    # Passivity tracking
    axes[1, 0].plot(history["epoch"], history["val_sigma_max"], color="r")
    axes[1, 0].axhline(1.0, color="black", linestyle="--",
                       linewidth=0.7, label="passivity bound")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel("max singular value (validation)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_title("Passivity: max sigma over val set "
                          "(soft-enforced during training)")

    # Causality residual
    axes[1, 1].plot(history["epoch"], history["val_caus"], color="purple")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_ylabel("causality residual (val)")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title("Causality: non-causal energy in diagonal impulse response")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="3 epochs on subsampled data; smoke test")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w-pass", type=float, default=50.0,
                        help="Soft passivity loss weight (Liu 2025 uses high values)")
    parser.add_argument("--w-caus", type=float, default=0.05,
                        help="Soft causality loss weight (Torun 2019/2020 form)")
    parser.add_argument("--restart-period", type=int, default=40,
                        help="SGDR T_0: epochs in first cosine cycle")
    parser.add_argument("--restart-mult", type=int, default=2,
                        help="SGDR T_mult: each cycle this much longer than prior")
    parser.add_argument("--hidden-dim", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ----- Load data -----
    print(f"Loading dataset:    {DATA_PT}")
    payload = torch.load(DATA_PT, weights_only=False)
    print(f"Loading splits:     {SPLITS_PT}")
    splits = torch.load(SPLITS_PT, weights_only=False)

    train_idx = splits["train_idx"].numpy()
    val_idx = splits["val_idx"].numpy()
    test_idx = splits["test_idx"].numpy()

    if args.dry_run:
        rng = np.random.default_rng(args.seed)
        train_idx = rng.choice(train_idx, size=200, replace=False)
        val_idx = rng.choice(val_idx, size=50, replace=False)
        args.epochs = 3
        print("DRY RUN MODE: 200 train, 50 val, 3 epochs")

    print(f"  train pairs: {len(train_idx)}")
    print(f"  val pairs:   {len(val_idx)}")
    print(f"  test pairs:  {len(test_idx)}  (held back for final eval)")

    # ----- Element weights for mode-conversion attention -----
    weight44, _ = compute_element_weights(payload, train_idx,
                                            n_sample=500, seed=args.seed)
    weight44 = weight44.to(device)
    np.set_printoptions(precision=2, suppress=True)
    print(f"\nElement weights (4x4, normalized to mean=1):")
    print(weight44.cpu().numpy())

    # ----- Datasets and loaders -----
    train_ds = DiffPairDataset(payload, train_idx)
    val_ds = DiffPairDataset(payload, val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers,
                               pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=(device == "cuda"))

    # ----- Build model -----
    freqs_hz = payload["frequencies"].to(torch.float64)
    model = DirectSequenceTCN(
        freqs_hz=freqs_hz,
        d_local=payload["X_local"].shape[1],
        d_global=payload["X_global"].shape[1],
        d_context=payload["X_context"].shape[1],
        hidden_dim=args.hidden_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    rf = 1 + (5 - 1) * 2 * sum(model.dilations)
    print(f"\nModel: DirectSequenceTCN")
    print(f"  parameters:       {n_params:,}")
    print(f"  hidden_dim:       {args.hidden_dim}")
    print(f"  dilations:        {model.dilations}")
    print(f"  receptive field:  {rf} samples (full-band: 401 freq points)")

    # ----- Optimizer + SGDR scheduler -----
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=args.lr,
                                   weight_decay=args.weight_decay)
    # SGDR: Loshchilov-Hutter ICLR 2017.  T_0 epochs in first cycle, each
    # subsequent cycle multiplies length by T_mult.  Periodic restarts help
    # escape narrow local minima typical in physics-regularized losses.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.restart_period, T_mult=args.restart_mult,
        eta_min=1e-6,
    )

    # ----- Run directory -----
    run_id = datetime.now().strftime("run_%Y-%m-%d_%H%M%S_tcn")
    if args.dry_run:
        run_id += "_dryrun"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRun dir: {run_dir}")

    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "args": vars(args),
            "model": "DirectSequenceTCN",
            "n_params": int(n_params),
            "dilations": list(model.dilations),
            "receptive_field": int(rf),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "device": device,
            "noise_floor_db": NOISE_FLOOR_DB,
        }, f, indent=2)

    # CSV log
    log_path = run_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "val_loss",
            "train_sdd11_db", "train_sdd21_db",
            "val_sdd11_db", "val_sdd21_db",
            "val_sdd11_db_proj", "val_sdd21_db_proj",
            "val_sigma_max", "val_l_caus",
            "lr", "epoch_time_s",
        ])

    print(f"\nStarting training: {args.epochs} epochs, batch {args.batch_size},"
           f" lr {args.lr}, w_pass {args.w_pass}, w_caus {args.w_caus}")
    print(f"SGDR cosine restarts: T_0={args.restart_period} T_mult={args.restart_mult}\n")

    logger = EpochLogger(args.epochs)
    best_avg = float("inf")
    epochs_no_improve = 0
    history = {k: [] for k in [
        "epoch", "train_loss", "val_loss",
        "val_sdd11", "val_sdd21",
        "val_sdd11_proj", "val_sdd21_proj",
        "val_sigma_max", "val_caus",
    ]}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, device, weight44,
                                        optimizer,
                                        w_pass=args.w_pass, w_caus=args.w_caus)
        val_stats = eval_one_epoch(model, val_loader, device, weight44,
                                     w_pass=args.w_pass, w_caus=args.w_caus)
        scheduler.step()
        dt = time.time() - t0

        cur_lr = optimizer.param_groups[0]["lr"]
        # Use the SVD-projected MAE as the "production" metric (this is what
        # the inverse model will see).  Average across Sdd11 and Sdd21.
        avg_db_proj = 0.5 * (val_stats["mae_sdd11_db_proj"]
                              + val_stats["mae_sdd21_db_proj"])

        is_best = avg_db_proj < best_avg
        if is_best:
            best_avg = avg_db_proj
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Per-line log
        logger.log(epoch, train_stats, val_stats, cur_lr, dt, is_best)

        # Save metrics history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["val_sdd11"].append(val_stats["mae_sdd11_db"])
        history["val_sdd21"].append(val_stats["mae_sdd21_db"])
        history["val_sdd11_proj"].append(val_stats["mae_sdd11_db_proj"])
        history["val_sdd21_proj"].append(val_stats["mae_sdd21_db_proj"])
        history["val_sigma_max"].append(val_stats["sigma_max"])
        history["val_caus"].append(val_stats["l_caus"])

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, train_stats["loss"], val_stats["loss"],
                train_stats["mae_sdd11_db"], train_stats["mae_sdd21_db"],
                val_stats["mae_sdd11_db"], val_stats["mae_sdd21_db"],
                val_stats["mae_sdd11_db_proj"], val_stats["mae_sdd21_db_proj"],
                val_stats["sigma_max"], val_stats["l_caus"],
                cur_lr, dt,
            ])

        # Save checkpoints
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_stats["loss"],
            "val_mae_sdd11_db": val_stats["mae_sdd11_db"],
            "val_mae_sdd21_db": val_stats["mae_sdd21_db"],
            "val_mae_sdd11_db_proj": val_stats["mae_sdd11_db_proj"],
            "val_mae_sdd21_db_proj": val_stats["mae_sdd21_db_proj"],
            "val_sigma_max": val_stats["sigma_max"],
            "weight44": weight44.cpu(),
            "config": vars(args),
        }
        torch.save(ckpt, run_dir / "checkpoint_last.pt")
        if is_best:
            torch.save(ckpt, run_dir / "checkpoint_best.pt")

        if not args.dry_run and epochs_no_improve >= args.patience:
            print(f"\n[EARLY STOP] No improvement in projected avg MAE for "
                  f"{args.patience} epochs.")
            break

    # ----- Diagnostics from best checkpoint -----
    print(f"\nLoading best checkpoint for diagnostics")
    best_ckpt = torch.load(run_dir / "checkpoint_best.pt", weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])

    plot_predictions(model, val_ds, device, freqs_hz.numpy(),
                      run_dir / "val_predictions.png", n_show=6, seed=args.seed)
    print(f"Saved: {run_dir / 'val_predictions.png'}")

    plot_training_curves(history, run_dir / "training_curves.png")
    print(f"Saved: {run_dir / 'training_curves.png'}")

    print(f"\n{'='*70}")
    print(f"Training complete.")
    print(f"  Best epoch: {best_ckpt['epoch']}")
    print(f"  Best val Sdd11 MAE (raw):  {best_ckpt['val_mae_sdd11_db']:.2f} dB")
    print(f"  Best val Sdd21 MAE (raw):  {best_ckpt['val_mae_sdd21_db']:.2f} dB")
    print(f"  Best val Sdd11 MAE (proj): {best_ckpt['val_mae_sdd11_db_proj']:.2f} dB"
          f"  <-- inverse model will see this")
    print(f"  Best val Sdd21 MAE (proj): {best_ckpt['val_mae_sdd21_db_proj']:.2f} dB"
          f"  <-- inverse model will see this")
    print(f"  Best avg (proj): {best_avg:.2f} dB  (target: 2 dB)")
    print(f"  Best val max sigma: {best_ckpt['val_sigma_max']:.3f}"
          f" (becomes <=1.0 after SVD projection)")
    print(f"  Run dir: {run_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()