"""
train_forward_v4.py
Training pipeline for the v4 parallel-decomposition rational forward model
on Universal-Diff-SI-Array.

Same loss, metrics, and training recipe as v3:
  - Composite loss: L1 on (Re,Im) + Huber on noise-floor-clamped dB + element-weighted L2
  - Noise floor at -55 dB
  - Passband MAE (above noise floor) as primary metric
  - AdamW + cosine LR decay + gradient clipping
  - Dynamic progress bar with live metrics

What's different:
  - Imports rational_forward_v4 (parallel decomposition)
  - Logs per-sub-model delta_scale for diagnostic visibility
  - Adds a delta_scale plot to the diagnostic figures
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rational_forward_v5 import RationalForwardModel  # noqa: E402


PROJECT_ROOT = Path.home() / "mece_project_inverse_model" / "Generative_Inverse_Design_of_High-Speed_Interconnects"
DATA_PT = PROJECT_ROOT / "data" / "processed" / "Universal-Diff-SI-Array" / "diff_pair_dataset.pt"
SPLITS_PT = PROJECT_ROOT / "sandbox_v1" / "data" / "splits.pt"
POLE_BASIS_PT = PROJECT_ROOT / "sandbox_v1" / "models" / "pole_basis" / "pole_basis.pt"
RUNS_DIR = PROJECT_ROOT / "sandbox_v1" / "models" / "forward_runs"

# Noise floor for the dataset (Hillebrecht 2024 / TUHH CONMLS solver)
NOISE_FLOOR_DB = -55.0


class DiffPairDataset(Dataset):
    """One diff pair per item: z-scored inputs and complex128 target S."""

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


def compute_element_weights(payload: dict, train_idx: np.ndarray,
                             n_sample: int = 500, seed: int = 0,
                             floor: float = 1e-3):
    """
    Per-(i,j) loss weights inversely proportional to mean |S_ij|^2.
    Normalized so mean weight = 1.0.
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


def composite_loss(pred, target, weight44):
    """
    L = 1.0 * L1(Re,Im) + 0.1 * Huber(clamped_dB) + 0.5 * weighted_L2(Re,Im)
    """
    eps = 1e-12
    l_linear = (pred.real - target.real).abs().mean() + \
               (pred.imag - target.imag).abs().mean()

    p_db = 20.0 * torch.log10(pred.abs().clamp_min(eps))
    t_db = 20.0 * torch.log10(target.abs().clamp_min(eps))
    p_db_c = p_db.clamp_min(NOISE_FLOOR_DB)
    t_db_c = t_db.clamp_min(NOISE_FLOOR_DB)
    l_db = torch.nn.functional.huber_loss(p_db_c, t_db_c, delta=2.0)

    diff_sq = (pred.real - target.real) ** 2 + (pred.imag - target.imag) ** 2
    l_weighted = (diff_sq * weight44.view(1, 1, 4, 4)).mean()

    total = 1.0 * l_linear + 0.1 * l_db + 0.5 * l_weighted
    return {
        "loss": total,
        "l_linear": l_linear.detach().item(),
        "l_db": l_db.detach().item(),
        "l_weighted": l_weighted.detach().item(),
    }


@torch.no_grad()
def passband_mae_db(pred, target, i, j, floor_db=NOISE_FLOOR_DB):
    """
    Mean absolute dB error on element (i,j), masked to frequencies where the
    TRUE response is above the noise floor.
    """
    eps = 1e-12
    p_db = 20.0 * torch.log10(pred[..., i, j].abs() + eps)
    t_db = 20.0 * torch.log10(target[..., i, j].abs() + eps)
    mask = t_db > floor_db
    if mask.sum() == 0:
        return float("nan")
    return (p_db[mask] - t_db[mask]).abs().mean().item()


@torch.no_grad()
def reciprocity_error(pred):
    return (pred - pred.transpose(-1, -2)).abs().max().item()


@torch.no_grad()
def passivity_max_sigma(pred):
    s_sub = pred[:, ::20]
    sv = torch.linalg.svdvals(s_sub)
    return sv.max().item()


class ProgressBar:
    """Single-line dynamic progress bar with custom metrics string."""

    def __init__(self, total_epochs: int, bar_len: int = 22):
        self.total = total_epochs
        self.bar_len = bar_len

    def render(self, epoch: int, metrics: str):
        frac = epoch / self.total
        filled = int(round(self.bar_len * frac))
        bar = "=" * filled + "-" * (self.bar_len - filled)
        pct = int(100 * frac)
        line = f"Epoch {epoch:03d}/{self.total} [{bar}] {pct:3d}% | {metrics}"
        sys.stdout.write("\r" + line.ljust(180))
        sys.stdout.flush()

    def finish(self, final_line: str = ""):
        sys.stdout.write("\n")
        if final_line:
            sys.stdout.write(final_line + "\n")
        sys.stdout.flush()


def train_one_epoch(model, loader, device, weight44, optimizer):
    model.train()
    acc = {"loss": 0.0, "l_linear": 0.0, "l_db": 0.0, "l_weighted": 0.0,
           "mae_sdd11_db": 0.0, "mae_sdd21_db": 0.0, "n": 0}
    for xl, xg, xc, S_tgt in loader:
        xl = xl.to(device, non_blocking=True)
        xg = xg.to(device, non_blocking=True)
        xc = xc.to(device, non_blocking=True)
        S_tgt = S_tgt.to(device, non_blocking=True)
        B = xl.shape[0]

        pred = model(xl, xg, xc)
        ld = composite_loss(pred, S_tgt, weight44)

        optimizer.zero_grad(set_to_none=True)
        ld["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        acc["loss"] += ld["loss"].detach().item() * B
        acc["l_linear"] += ld["l_linear"] * B
        acc["l_db"] += ld["l_db"] * B
        acc["l_weighted"] += ld["l_weighted"] * B
        acc["mae_sdd11_db"] += passband_mae_db(pred, S_tgt, 0, 0) * B
        acc["mae_sdd21_db"] += passband_mae_db(pred, S_tgt, 1, 0) * B
        acc["n"] += B

    n = acc["n"]
    return {k: (v / n if k != "n" else v) for k, v in acc.items()}


@torch.no_grad()
def eval_one_epoch(model, loader, device, weight44):
    model.eval()
    acc = {"loss": 0.0, "mae_sdd11_db": 0.0, "mae_sdd21_db": 0.0,
           "recip": 0.0, "passivity": 0.0, "n": 0}
    for xl, xg, xc, S_tgt in loader:
        xl = xl.to(device, non_blocking=True)
        xg = xg.to(device, non_blocking=True)
        xc = xc.to(device, non_blocking=True)
        S_tgt = S_tgt.to(device, non_blocking=True)
        B = xl.shape[0]
        pred = model(xl, xg, xc)
        ld = composite_loss(pred, S_tgt, weight44)
        acc["loss"] += ld["loss"].item() * B
        acc["mae_sdd11_db"] += passband_mae_db(pred, S_tgt, 0, 0) * B
        acc["mae_sdd21_db"] += passband_mae_db(pred, S_tgt, 1, 0) * B
        acc["recip"] = max(acc["recip"], reciprocity_error(pred))
        acc["passivity"] = max(acc["passivity"], passivity_max_sigma(pred))
        acc["n"] += B
    n = acc["n"]
    return {
        "loss": acc["loss"] / n,
        "mae_sdd11_db": acc["mae_sdd11_db"] / n,
        "mae_sdd21_db": acc["mae_sdd21_db"] / n,
        "recip_max": acc["recip"],
        "passivity_max_sigma": acc["passivity"],
    }


@torch.no_grad()
def plot_predictions(model, dataset, device, freqs_hz, out_path,
                      n_show: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(dataset), size=min(n_show, len(dataset)), replace=False)
    model.eval()
    fig, axes = plt.subplots(2, n_show, figsize=(3.2 * n_show, 6), sharex=True)
    f_ghz = freqs_hz / 1e9
    for col, idx in enumerate(pick):
        xl, xg, xc, S_tgt = dataset[int(idx)]
        xl = xl.unsqueeze(0).to(device)
        xg = xg.unsqueeze(0).to(device)
        xc = xc.unsqueeze(0).to(device)
        S_pred = model(xl, xg, xc).squeeze(0).cpu()
        for row, (i, j, label) in enumerate([(0, 0, "Sdd11"), (1, 0, "Sdd21")]):
            ax = axes[row, col]
            ax.plot(f_ghz, 20 * np.log10(S_tgt[:, i, j].abs().numpy() + 1e-12),
                    "b-", linewidth=1, label="target" if (row == 0 and col == 0) else None)
            ax.plot(f_ghz, 20 * np.log10(S_pred[:, i, j].abs().numpy() + 1e-12),
                    "r--", linewidth=1, label="pred" if (row == 0 and col == 0) else None)
            ax.axhline(NOISE_FLOOR_DB, color="gray", linestyle=":", linewidth=0.6)
            ax.set_ylabel(f"|{label}| [dB]")
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(f"sample idx {int(idx)}", fontsize=9)
                if col == 0:
                    ax.legend(fontsize=8)
            if row == 1:
                ax.set_xlabel("Frequency [GHz]")
    plt.suptitle("Val predictions vs targets (dotted line = noise floor -55 dB)",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--M", type=int, default=5,
                        help="Number of parallel sub-models in the decomposition")
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
    print(f"Loading pole basis: {POLE_BASIS_PT}")
    basis = torch.load(POLE_BASIS_PT, weights_only=False)

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
    print(f"  test pairs:  {len(test_idx)}")

    # ----- Element weights -----
    weight44, _ = compute_element_weights(payload, train_idx,
                                            n_sample=500, seed=args.seed)
    weight44 = weight44.to(device)
    np.set_printoptions(precision=2, suppress=True)
    print(f"\nElement weights (4x4, mean=1):")
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

    # ----- Build v4 model -----
    freqs_hz = payload["frequencies"].to(torch.float64)
    model = RationalForwardModel(
        freqs_hz=freqs_hz,
        init_poles_real=basis["poles_real"].to(torch.float64),
        init_poles_cmplx=basis["poles_cmplx_uhp"].to(torch.complex128),
        M=args.M,
        d_local=payload["X_local"].shape[1],
        d_global=payload["X_global"].shape[1],
        d_context=payload["X_context"].shape[1],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: rational_forward_v4, M={args.M} ({n_params:,} parameters)")
    print(f"  Sub-model pole budget:")
    for i in range(args.M):
        print(f"    sub {i}: {model.sub_n_real[i]} real + {model.sub_n_cmplx[i]} complex"
              f" -> {model.sub_n_real[i] + 2*model.sub_n_cmplx[i]} layer poles")

    # ----- Optimizer + scheduler -----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ----- Run directory -----
    run_id = datetime.now().strftime("run_%Y-%m-%d_%H%M%S_v4")
    if args.dry_run:
        run_id += "_dryrun"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRun dir: {run_dir}")

    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "args": vars(args),
            "model": "rational_forward_v4",
            "M": args.M,
            "n_params": int(n_params),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "device": device,
            "noise_floor_db": NOISE_FLOOR_DB,
        }, f, indent=2)

    log_path = run_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        header = ["epoch", "train_loss", "val_loss",
                  "train_mae_sdd11_db", "train_mae_sdd21_db",
                  "val_mae_sdd11_db", "val_mae_sdd21_db",
                  "val_recip_max", "val_passivity_max_sigma",
                  "lr", "epoch_time_s"]
        # Per-sub-model delta_scale columns
        for i in range(args.M):
            header.append(f"delta_sub{i}")
        csv.writer(f).writerow(header)

    print(f"\nStarting training: {args.epochs} epochs, batch {args.batch_size}, lr {args.lr}")
    print(f"Noise floor for dB metrics: {NOISE_FLOOR_DB} dB\n")

    bar = ProgressBar(args.epochs)
    best_avg = float("inf")
    epochs_no_improve = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "val_sdd11": [], "val_sdd21": [],
               "delta_scales": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, device, weight44, optimizer)
        val_stats = eval_one_epoch(model, val_loader, device, weight44)
        scheduler.step()
        dt = time.time() - t0

        cur_lr = optimizer.param_groups[0]["lr"]
        deltas = model.get_delta_scales()
        avg_db = 0.5 * (val_stats["mae_sdd11_db"] + val_stats["mae_sdd21_db"])

        is_best = avg_db < best_avg
        marker = "*" if is_best else " "
        if is_best:
            best_avg = avg_db
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Progress bar metrics; show min/max delta to keep line compact
        d_min, d_max = min(deltas), max(deltas)
        metrics_str = (
            f"train:{train_stats['loss']:.3f} val:{val_stats['loss']:.3f}"
            f" Sdd11:{val_stats['mae_sdd11_db']:5.2f}dB"
            f" Sdd21:{val_stats['mae_sdd21_db']:5.2f}dB"
            f" avg:{avg_db:5.2f}dB delta:[{d_min:.2f}-{d_max:.2f}]"
            f" lr:{cur_lr:.1e} {marker}"
        )
        bar.render(epoch, metrics_str)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["val_sdd11"].append(val_stats["mae_sdd11_db"])
        history["val_sdd21"].append(val_stats["mae_sdd21_db"])
        history["delta_scales"].append(deltas)

        row = [epoch, train_stats["loss"], val_stats["loss"],
                train_stats["mae_sdd11_db"], train_stats["mae_sdd21_db"],
                val_stats["mae_sdd11_db"], val_stats["mae_sdd21_db"],
                val_stats["recip_max"], val_stats["passivity_max_sigma"],
                cur_lr, dt] + list(deltas)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_stats["loss"],
            "val_mae_sdd11_db": val_stats["mae_sdd11_db"],
            "val_mae_sdd21_db": val_stats["mae_sdd21_db"],
            "weight44": weight44.cpu(),
            "config": vars(args),
        }
        torch.save(ckpt, run_dir / "checkpoint_last.pt")
        if is_best:
            torch.save(ckpt, run_dir / "checkpoint_best.pt")

        if not args.dry_run and epochs_no_improve >= args.patience:
            bar.finish(f"\n[EARLY STOP] No improvement for {args.patience} epochs.")
            break

    bar.finish()

    # ----- Diagnostics -----
    print(f"\nLoading best checkpoint for diagnostics")
    best_ckpt = torch.load(run_dir / "checkpoint_best.pt", weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    plot_predictions(model, val_ds, device, freqs_hz.numpy(),
                      run_dir / "val_predictions.png", n_show=6, seed=args.seed)
    print(f"Saved: {run_dir / 'val_predictions.png'}")

    # Loss/MAE/delta curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("composite loss")
    axes[0].set_yscale("log"); axes[0].grid(True, alpha=0.3); axes[0].legend()
    axes[1].plot(history["epoch"], history["val_sdd11"], label="Sdd11")
    axes[1].plot(history["epoch"], history["val_sdd21"], label="Sdd21")
    axes[1].axhline(2.0, color="green", linestyle=":", linewidth=0.7, label="target 2 dB")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val passband MAE [dB]")
    axes[1].grid(True, alpha=0.3); axes[1].legend()
    # Per-sub-model delta curves
    deltas_arr = np.array(history["delta_scales"])  # (n_epochs, M)
    for i in range(args.M):
        axes[2].plot(history["epoch"], deltas_arr[:, i], label=f"sub {i}")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("delta_scale per sub-model")
    axes[2].grid(True, alpha=0.3); axes[2].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curves.png", dpi=110)
    plt.close()
    print(f"Saved: {run_dir / 'loss_curves.png'}")

    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"  Best epoch: {best_ckpt['epoch']}")
    print(f"  Best val passband Sdd11 MAE: {best_ckpt['val_mae_sdd11_db']:.2f} dB")
    print(f"  Best val passband Sdd21 MAE: {best_ckpt['val_mae_sdd21_db']:.2f} dB")
    print(f"  Best avg: {best_avg:.2f} dB (target: 2 dB)")
    print(f"  Final delta_scales: {[f'{d:.2f}' for d in history['delta_scales'][-1]]}")
    print(f"  Run dir: {run_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()