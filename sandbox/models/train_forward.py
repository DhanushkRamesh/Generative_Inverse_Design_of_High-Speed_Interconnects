"""
train_forward.py — train the per-sample-pole rational forward model on
the Universal-Diff-SI-Array dataset.

Reads:
  data/processed/Universal-Diff-SI-Array/diff_pair_dataset.pt
  sandbox_v1/data/splits.pt
  sandbox_v1/models/pole_basis/pole_basis.pt  (used to init the pole head)

Writes:
  sandbox_v1/models/forward_runs/run_<timestamp>/
      config.json
      train_log.csv
      checkpoint_best.pt
      checkpoint_last.pt
      loss_curves.png

Two modes:
  --dry-run   2 epochs, 100 train samples, 50 val samples. ~30 seconds.
              Goal: confirm nothing crashes, loss decreases.
  (no flag)   full training: all train pairs, val every epoch,
              early stopping with patience 8 on val loss.

Loss: MSE on (Re(S), Im(S)) over all (freq, port, port) entries.
Optimizer: AdamW with cosine LR decay.
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

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Allow `python train_forward.py` from sandbox_v1/models/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rational_net_forward import RationalForwardModel  # noqa: E402


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = Path.home() / "mece_project_inverse_model" / "Generative_Inverse_Design_of_High-Speed_Interconnects"
DATA_PT = PROJECT_ROOT / "data" / "processed" / "Universal-Diff-SI-Array" / "diff_pair_dataset.pt"
SPLITS_PT = PROJECT_ROOT / "sandbox_v1" / "data" / "splits.pt"
POLE_BASIS_PT = PROJECT_ROOT / "sandbox_v1" / "models" / "pole_basis" / "pole_basis.pt"

RUNS_DIR = PROJECT_ROOT / "sandbox_v1" / "models" / "forward_runs"


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class DiffPairDataset(Dataset):
    """
    Returns one diff pair per __getitem__:
        x_local, x_global, x_context  (float32 tensors, already z-scored)
        S_target                       (complex128, shape (F, 4, 4))
    """

    def __init__(self, payload: dict, indices: np.ndarray):
        self.x_local = payload["X_local"][indices]
        self.x_global = payload["X_global"][indices]
        self.x_context = payload["X_context"][indices]
        # Y_real/Y_imag are float32; combine into complex128 lazily per item to save RAM
        self.y_real = payload["Y_real"][indices]
        self.y_imag = payload["Y_imag"][indices]
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        S = torch.complex(self.y_real[i].to(torch.float64),
                          self.y_imag[i].to(torch.float64))
        return self.x_local[i], self.x_global[i], self.x_context[i], S


# ----------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------
def mse_complex(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE on real and imaginary parts, averaged over all entries."""
    return ((pred.real - target.real) ** 2 + (pred.imag - target.imag) ** 2).mean()


def db_mae_sdd(pred: torch.Tensor, target: torch.Tensor, i: int, j: int) -> float:
    """Mean absolute error in dB on a single S-element (e.g. Sdd11 = (0,0), Sdd21 = (1,0))."""
    eps = 1e-12
    p_db = 20 * torch.log10(pred[..., i, j].abs() + eps)
    t_db = 20 * torch.log10(target[..., i, j].abs() + eps)
    return (p_db - t_db).abs().mean().item()


# ----------------------------------------------------------------------
# Train / eval loops
# ----------------------------------------------------------------------
def run_epoch(model, loader, device, optimizer=None):
    """Single pass. If optimizer is None, runs in eval mode (no grad)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_n = 0
    total_mae_sdd11 = 0.0
    total_mae_sdd21 = 0.0

    cm = torch.enable_grad() if is_train else torch.no_grad()
    with cm:
        for xl, xg, xc, S_target in loader:
            xl = xl.to(device, non_blocking=True)
            xg = xg.to(device, non_blocking=True)
            xc = xc.to(device, non_blocking=True)
            S_target = S_target.to(device, non_blocking=True)
            B = xl.shape[0]

            pred = model(xl, xg, xc)
            loss = mse_complex(pred, S_target)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

            total_loss += loss.item() * B
            total_mae_sdd11 += db_mae_sdd(pred, S_target, 0, 0) * B
            total_mae_sdd21 += db_mae_sdd(pred, S_target, 1, 0) * B
            total_n += B

    return {
        "loss": total_loss / total_n,
        "mae_sdd11_db": total_mae_sdd11 / total_n,
        "mae_sdd21_db": total_mae_sdd21 / total_n,
    }


# ----------------------------------------------------------------------
# Diagnostic plot — pred vs target on 6 random val samples
# ----------------------------------------------------------------------
def plot_predictions(model, dataset, device, freqs_hz, out_path, n_show=6, seed=0):
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(dataset), size=min(n_show, len(dataset)), replace=False)

    model.eval()
    fig, axes = plt.subplots(2, n_show, figsize=(3.2 * n_show, 6), sharex=True)
    with torch.no_grad():
        for col, idx in enumerate(pick):
            xl, xg, xc, S_tgt = dataset[int(idx)]
            xl = xl.unsqueeze(0).to(device)
            xg = xg.unsqueeze(0).to(device)
            xc = xc.unsqueeze(0).to(device)
            S_pred = model(xl, xg, xc).squeeze(0).cpu()
            f_ghz = freqs_hz / 1e9
            # Sdd11
            ax = axes[0, col]
            ax.plot(f_ghz, 20 * np.log10(S_tgt[:, 0, 0].abs() + 1e-12), "b-", linewidth=1, label="target")
            ax.plot(f_ghz, 20 * np.log10(S_pred[:, 0, 0].abs() + 1e-12), "r--", linewidth=1, label="pred")
            ax.set_ylabel("|Sdd11| [dB]")
            ax.set_title(f"sample idx {int(idx)}", fontsize=9)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.legend(fontsize=8)
            # Sdd21
            ax = axes[1, col]
            ax.plot(f_ghz, 20 * np.log10(S_tgt[:, 1, 0].abs() + 1e-12), "b-", linewidth=1)
            ax.plot(f_ghz, 20 * np.log10(S_pred[:, 1, 0].abs() + 1e-12), "r--", linewidth=1)
            ax.set_ylabel("|Sdd21| [dB]")
            ax.set_xlabel("Frequency [GHz]")
            ax.grid(True, alpha=0.3)
    plt.suptitle("Val predictions vs targets (random sample)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick smoke test: 2 epochs, 100 train + 50 val samples")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ----- load data -----
    print(f"Loading dataset:  {DATA_PT}")
    payload = torch.load(DATA_PT, weights_only=False)
    print(f"Loading splits:   {SPLITS_PT}")
    splits = torch.load(SPLITS_PT, weights_only=False)
    print(f"Loading pole basis: {POLE_BASIS_PT}")
    basis = torch.load(POLE_BASIS_PT, weights_only=False)

    train_idx = splits["train_idx"].numpy()
    val_idx = splits["val_idx"].numpy()
    test_idx = splits["test_idx"].numpy()

    if args.dry_run:
        rng = np.random.default_rng(args.seed)
        train_idx = rng.choice(train_idx, size=100, replace=False)
        val_idx = rng.choice(val_idx, size=50, replace=False)
        args.epochs = 2
        print("DRY RUN: 100 train, 50 val, 2 epochs")

    print(f"  train pairs: {len(train_idx)}")
    print(f"  val pairs:   {len(val_idx)}")
    print(f"  test pairs:  {len(test_idx)}  (not used during training)")

    train_ds = DiffPairDataset(payload, train_idx)
    val_ds = DiffPairDataset(payload, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device == "cuda"))

    # ----- build model -----
    freqs_hz = payload["frequencies"].to(torch.float64)
    model = RationalForwardModel(
        freqs_hz=freqs_hz,
        init_poles_real=basis["poles_real"].to(torch.float64),
        init_poles_cmplx=basis["poles_cmplx_uhp"].to(torch.complex128),
        d_local=payload["X_local"].shape[1],
        d_global=payload["X_global"].shape[1],
        d_context=payload["X_context"].shape[1],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ----- run directory -----
    run_id = datetime.now().strftime("run_%Y-%m-%d_%H%M%S") + ("_dryrun" if args.dry_run else "")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"  run dir: {run_dir}")

    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "args": vars(args),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "n_params": int(n_params),
            "device": device,
            "pole_basis": str(POLE_BASIS_PT),
        }, f, indent=2)

    log_path = run_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss",
                                "train_mae_sdd11_db", "train_mae_sdd21_db",
                                "val_mae_sdd11_db", "val_mae_sdd21_db",
                                "lr", "epoch_time_s"])

    # ----- training loop -----
    best_val = float("inf")
    epochs_no_improve = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "val_mae_sdd11": [], "val_mae_sdd21": []}

    for epoch in range(args.epochs):
        t0 = time.time()
        train_stats = run_epoch(model, train_loader, device, optimizer)
        val_stats = run_epoch(model, val_loader, device, None)
        scheduler.step()
        dt = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]

        print(f"epoch {epoch+1:3d}/{args.epochs}  "
              f"train_loss={train_stats['loss']:.4e}  "
              f"val_loss={val_stats['loss']:.4e}  "
              f"val_Sdd11={val_stats['mae_sdd11_db']:5.2f}dB  "
              f"val_Sdd21={val_stats['mae_sdd21_db']:5.2f}dB  "
              f"lr={cur_lr:.2e}  ({dt:.1f}s)")

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["val_mae_sdd11"].append(val_stats["mae_sdd11_db"])
        history["val_mae_sdd21"].append(val_stats["mae_sdd21_db"])

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, train_stats["loss"], val_stats["loss"],
                train_stats["mae_sdd11_db"], train_stats["mae_sdd21_db"],
                val_stats["mae_sdd11_db"], val_stats["mae_sdd21_db"],
                cur_lr, dt,
            ])

        # Checkpoint
        ckpt = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_stats["loss"],
            "val_mae_sdd11_db": val_stats["mae_sdd11_db"],
            "val_mae_sdd21_db": val_stats["mae_sdd21_db"],
            "config": vars(args),
        }
        torch.save(ckpt, run_dir / "checkpoint_last.pt")
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            epochs_no_improve = 0
            torch.save(ckpt, run_dir / "checkpoint_best.pt")
        else:
            epochs_no_improve += 1
            if not args.dry_run and epochs_no_improve >= args.patience:
                print(f"Early stopping: val loss did not improve for {args.patience} epochs")
                break

    # ----- final diagnostics -----
    print(f"\nLoading best checkpoint for diagnostics: {run_dir / 'checkpoint_best.pt'}")
    best_ckpt = torch.load(run_dir / "checkpoint_best.pt", weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])

    plot_predictions(model, val_ds, device, freqs_hz.numpy(),
                     run_dir / "val_predictions.png", n_show=6, seed=args.seed)
    print(f"  Saved val_predictions.png")

    # Loss curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss (MSE on Re,Im)")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(history["epoch"], history["val_mae_sdd11"], label="Sdd11")
    axes[1].plot(history["epoch"], history["val_mae_sdd21"], label="Sdd21")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val MAE [dB]")
    axes[1].axhline(1.5, color="k", linestyle=":", linewidth=0.7, label="target 1.5 dB")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curves.png", dpi=110)
    plt.close()
    print(f"  Saved loss_curves.png")

    print(f"\nBest val loss: {best_val:.4e}")
    print(f"Best val Sdd11 MAE: {best_ckpt['val_mae_sdd11_db']:.2f} dB")
    print(f"Best val Sdd21 MAE: {best_ckpt['val_mae_sdd21_db']:.2f} dB")
    print(f"\nDone. Run dir: {run_dir}")


if __name__ == "__main__":
    main()