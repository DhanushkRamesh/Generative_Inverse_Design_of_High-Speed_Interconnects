"""
train_forward_v2.py — training loop for rational_forward_v2 model.

Changes from v1:
  * Imports rational_forward_v2 (multi-scale Fourier, residual encoder, freer pole head)
  * Per-element loss weighting: each (i, j) S-element's loss is divided by
    its mean squared magnitude on the training set. Computed once at startup
    from a sample of train pairs, then frozen. Forces the model to fit weak
    elements (mode conversion) proportionally instead of ignoring them.
  * Logs the pole-head delta_scale every epoch so we can see whether the
    model is using its pole-freedom.
  * Default lr 5e-4 (was 1e-3) — bigger model needs gentler optimization.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rational_net_forward_v2 import RationalForwardModel  # noqa: E402


PROJECT_ROOT = Path.home() / "mece_project_inverse_model" / "Generative_Inverse_Design_of_High-Speed_Interconnects"
DATA_PT = PROJECT_ROOT / "data" / "processed" / "Universal-Diff-SI-Array" / "diff_pair_dataset.pt"
SPLITS_PT = PROJECT_ROOT / "sandbox_v1" / "data" / "splits.pt"
POLE_BASIS_PT = PROJECT_ROOT / "sandbox_v1" / "models" / "pole_basis" / "pole_basis.pt"
RUNS_DIR = PROJECT_ROOT / "sandbox_v1" / "models" / "forward_runs"


class DiffPairDataset(Dataset):
    def __init__(self, payload, indices):
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


def compute_element_weights(payload, train_idx, n_sample=500, seed=0, floor=1e-3):
    """
    Compute per-(i,j) loss weights = 1 / max(mean(|S_ij|^2), floor).
    Higher weight for weaker elements so MSE doesn't ignore them.
    """
    rng = np.random.default_rng(seed)
    pick = rng.choice(train_idx, size=min(n_sample, len(train_idx)), replace=False)
    yr = payload["Y_real"][pick].to(torch.float64)
    yi = payload["Y_imag"][pick].to(torch.float64)
    mag_sq = (yr ** 2 + yi ** 2)             # (N, F, 4, 4)
    mean_per_elem = mag_sq.mean(dim=(0, 1))  # (4, 4)
    weight = 1.0 / mean_per_elem.clamp_min(floor)
    # Normalize so the mean weight is 1 (keeps loss scale comparable to unweighted)
    weight = weight * (16.0 / weight.sum())
    return weight, mean_per_elem


def weighted_mse_complex(pred, target, weight44):
    """MSE on (Re, Im), weighted per-element."""
    diff2 = (pred.real - target.real) ** 2 + (pred.imag - target.imag) ** 2  # (B, F, 4, 4)
    # weight44 shape (4, 4) → broadcast over B, F
    weighted = diff2 * weight44.view(1, 1, 4, 4)
    return weighted.mean()


def db_mae_sdd(pred, target, i, j):
    eps = 1e-12
    p_db = 20 * torch.log10(pred[..., i, j].abs() + eps)
    t_db = 20 * torch.log10(target[..., i, j].abs() + eps)
    return (p_db - t_db).abs().mean().item()


def run_epoch(model, loader, device, weight44, optimizer=None):
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
            loss = weighted_mse_complex(pred, S_target, weight44)

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
            ax = axes[0, col]
            ax.plot(f_ghz, 20 * np.log10(S_tgt[:, 0, 0].abs().numpy() + 1e-12), "b-", linewidth=1, label="target")
            ax.plot(f_ghz, 20 * np.log10(S_pred[:, 0, 0].abs().numpy() + 1e-12), "r--", linewidth=1, label="pred")
            ax.set_ylabel("|Sdd11| [dB]")
            ax.set_title(f"sample idx {int(idx)}", fontsize=9)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.legend(fontsize=8)
            ax = axes[1, col]
            ax.plot(f_ghz, 20 * np.log10(S_tgt[:, 1, 0].abs().numpy() + 1e-12), "b-", linewidth=1)
            ax.plot(f_ghz, 20 * np.log10(S_pred[:, 1, 0].abs().numpy() + 1e-12), "r--", linewidth=1)
            ax.set_ylabel("|Sdd21| [dB]")
            ax.set_xlabel("Frequency [GHz]")
            ax.grid(True, alpha=0.3)
    plt.suptitle("Val predictions vs targets (random sample)", fontsize=10)
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
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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
        train_idx = rng.choice(train_idx, size=200, replace=False)
        val_idx = rng.choice(val_idx, size=50, replace=False)
        args.epochs = 3
        print("DRY RUN: 200 train, 50 val, 3 epochs")

    print(f"  train pairs: {len(train_idx)}")
    print(f"  val pairs:   {len(val_idx)}")
    print(f"  test pairs:  {len(test_idx)}")

    weight44, mean_per_elem = compute_element_weights(payload, train_idx, n_sample=500, seed=args.seed)
    weight44 = weight44.to(device)
    print(f"\nElement-wise loss weights (normalized, mean=1):")
    print(weight44.cpu().numpy())
    print(f"Mean |S_ij|^2 on a sample of 500 train pairs:")
    print(mean_per_elem.cpu().numpy())

    train_ds = DiffPairDataset(payload, train_idx)
    val_ds = DiffPairDataset(payload, val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device == "cuda"))

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
    print(f"\nmodel parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_id = datetime.now().strftime("run_%Y-%m-%d_%H%M%S") + ("_dryrun" if args.dry_run else "") + "_v2"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {run_dir}")

    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "args": vars(args), "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)), "n_test": int(len(test_idx)),
            "n_params": int(n_params), "device": device,
            "pole_basis": str(POLE_BASIS_PT), "model": "rational_forward_v2",
        }, f, indent=2)

    log_path = run_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss",
                                "train_mae_sdd11_db", "train_mae_sdd21_db",
                                "val_mae_sdd11_db", "val_mae_sdd21_db",
                                "lr", "delta_scale", "epoch_time_s"])

    best_val = float("inf")
    epochs_no_improve = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "val_mae_sdd11": [], "val_mae_sdd21": [], "delta_scale": []}

    for epoch in range(args.epochs):
        t0 = time.time()
        train_stats = run_epoch(model, train_loader, device, weight44, optimizer)
        val_stats = run_epoch(model, val_loader, device, weight44, None)
        scheduler.step()
        dt = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]
        dscale = model.pole_head.delta_scale.item()

        print(f"epoch {epoch+1:3d}/{args.epochs}  "
              f"train={train_stats['loss']:.3e}  val={val_stats['loss']:.3e}  "
              f"Sdd11={val_stats['mae_sdd11_db']:5.2f}dB  Sdd21={val_stats['mae_sdd21_db']:5.2f}dB  "
              f"δ={dscale:.3f}  lr={cur_lr:.2e}  ({dt:.1f}s)")

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["val_mae_sdd11"].append(val_stats["mae_sdd11_db"])
        history["val_mae_sdd21"].append(val_stats["mae_sdd21_db"])
        history["delta_scale"].append(dscale)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, train_stats["loss"], val_stats["loss"],
                train_stats["mae_sdd11_db"], train_stats["mae_sdd21_db"],
                val_stats["mae_sdd11_db"], val_stats["mae_sdd21_db"],
                cur_lr, dscale, dt,
            ])

        ckpt = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_stats["loss"],
            "val_mae_sdd11_db": val_stats["mae_sdd11_db"],
            "val_mae_sdd21_db": val_stats["mae_sdd21_db"],
            "weight44": weight44.cpu(),
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
                print(f"Early stopping: no improvement for {args.patience} epochs")
                break

    print(f"\nLoading best checkpoint")
    best_ckpt = torch.load(run_dir / "checkpoint_best.pt", weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    plot_predictions(model, val_ds, device, freqs_hz.numpy(), run_dir / "val_predictions.png", n_show=6, seed=args.seed)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_yscale("log"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history["epoch"], history["val_mae_sdd11"], label="Sdd11")
    axes[1].plot(history["epoch"], history["val_mae_sdd21"], label="Sdd21")
    axes[1].axhline(1.5, color="k", linestyle=":", linewidth=0.7)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val MAE [dB]")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].plot(history["epoch"], history["delta_scale"])
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("pole delta_scale")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curves.png", dpi=110)
    plt.close()

    print(f"\nBest val loss: {best_val:.4e}")
    print(f"Best val Sdd11 MAE: {best_ckpt['val_mae_sdd11_db']:.2f} dB")
    print(f"Best val Sdd21 MAE: {best_ckpt['val_mae_sdd21_db']:.2f} dB")
    print(f"Final delta_scale: {history['delta_scale'][-1]:.3f}")
    print(f"\nDone. Run dir: {run_dir}")


if __name__ == "__main__":
    main()