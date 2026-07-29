"""
train_forward_model_v7.py
Training pipeline for the v7 Implicit Neural Representation (INR) Surrogate.
Enforces physical Passivity directly via Singular Value penalties.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rational_forward_v7 import FrequencyConditionedResNet

PROJECT_ROOT = Path.home() / "mece_project_inverse_model" / "Generative_Inverse_Design_of_High-Speed_Interconnects"
DATA_PT = PROJECT_ROOT / "data" / "processed" / "Universal-Diff-SI-Array" / "diff_pair_dataset.pt"
SPLITS_PT = PROJECT_ROOT / "sandbox_v1" / "data" / "splits.pt"
RUNS_DIR = PROJECT_ROOT / "sandbox_v1" / "models" / "forward_runs"

NOISE_FLOOR_DB = -55.0

class DiffPairDataset(Dataset):
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
    """INR Physics Loss: L1 Phase + Huber Magnitude + SVD Passivity"""
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

    # Passivity Penalty (Sample 10 random frequencies per batch to keep it fast)
    F_len = pred.shape[1]
    f_idx = torch.randint(0, F_len, (10,), device=pred.device)
    sv = torch.linalg.svdvals(pred[:, f_idx, :, :])
    # Penalty activates only if Singular Value > 1.0
    l_passivity = F.relu(sv - 1.0).mean()

    # Total formulation
    total = 1.0 * l_linear + 0.1 * l_db + 0.5 * l_weighted + 5.0 * l_passivity

    return {
        "loss": total,
        "l_linear": l_linear.detach().item(),
        "l_db": l_db.detach().item(),
        "l_passivity": l_passivity.detach().item(),
    }

@torch.no_grad()
def passband_mae_db(pred, target, i, j, floor_db=NOISE_FLOOR_DB):
    eps = 1e-12
    p_db = 20.0 * torch.log10(pred[..., i, j].abs() + eps)
    t_db = 20.0 * torch.log10(target[..., i, j].abs() + eps)
    mask = t_db > floor_db
    if mask.sum() == 0:
        return float("nan")
    return (p_db[mask] - t_db[mask]).abs().mean().item()

class ProgressBar:
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

def train_one_epoch(model, loader, freqs_hz, device, weight44, optimizer):
    model.train()
    acc = {"loss": 0.0, "l_passivity": 0.0, "mae_sdd11_db": 0.0, "mae_sdd21_db": 0.0, "n": 0}
    for xl, xg, xc, S_tgt in loader:
        xl = xl.to(device, non_blocking=True)
        xg = xg.to(device, non_blocking=True)
        xc = xc.to(device, non_blocking=True)
        S_tgt = S_tgt.to(device, non_blocking=True)
        B = xl.shape[0]

        pred = model(xl, xg, xc, freqs_hz)
        ld = composite_loss(pred, S_tgt, weight44)

        optimizer.zero_grad(set_to_none=True)
        ld["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        acc["loss"] += ld["loss"].detach().item() * B
        acc["l_passivity"] += ld["l_passivity"] * B
        acc["mae_sdd11_db"] += passband_mae_db(pred, S_tgt, 0, 0) * B
        acc["mae_sdd21_db"] += passband_mae_db(pred, S_tgt, 1, 0) * B
        acc["n"] += B

    n = acc["n"]
    return {k: (v / n if k != "n" else v) for k, v in acc.items()}

@torch.no_grad()
def eval_one_epoch(model, loader, freqs_hz, device, weight44):
    model.eval()
    acc = {"loss": 0.0, "mae_sdd11_db": 0.0, "mae_sdd21_db": 0.0, "n": 0}
    for xl, xg, xc, S_tgt in loader:
        xl = xl.to(device, non_blocking=True)
        xg = xg.to(device, non_blocking=True)
        xc = xc.to(device, non_blocking=True)
        S_tgt = S_tgt.to(device, non_blocking=True)
        B = xl.shape[0]
        
        pred = model(xl, xg, xc, freqs_hz)
        ld = composite_loss(pred, S_tgt, weight44)
        
        acc["loss"] += ld["loss"].item() * B
        acc["mae_sdd11_db"] += passband_mae_db(pred, S_tgt, 0, 0) * B
        acc["mae_sdd21_db"] += passband_mae_db(pred, S_tgt, 1, 0) * B
        acc["n"] += B
        
    n = acc["n"]
    return {
        "loss": acc["loss"] / n,
        "mae_sdd11_db": acc["mae_sdd11_db"] / n,
        "mae_sdd21_db": acc["mae_sdd21_db"] / n,
    }

@torch.no_grad()
def plot_predictions(model, dataset, freqs_hz, device, out_path, n_show: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(dataset), size=min(n_show, len(dataset)), replace=False)
    model.eval()
    fig, axes = plt.subplots(2, n_show, figsize=(3.2 * n_show, 6), sharex=True)
    f_ghz = freqs_hz.cpu().numpy() / 1e9
    
    for col, idx in enumerate(pick):
        xl, xg, xc, S_tgt = dataset[int(idx)]
        xl = xl.unsqueeze(0).to(device)
        xg = xg.unsqueeze(0).to(device)
        xc = xc.unsqueeze(0).to(device)
        S_pred = model(xl, xg, xc, freqs_hz).squeeze(0).cpu()
        
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
                
    plt.suptitle("INR Predictions vs Targets", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64) 
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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

    weight44, _ = compute_element_weights(payload, train_idx, n_sample=500, seed=args.seed)
    weight44 = weight44.to(device)

    train_ds = DiffPairDataset(payload, train_idx)
    val_ds = DiffPairDataset(payload, val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device == "cuda"))

    freqs_hz = payload["frequencies"].to(device)

    model = FrequencyConditionedResNet(
        d_local=payload["X_local"].shape[1],
        d_global=payload["X_global"].shape[1],
        d_context=payload["X_context"].shape[1],
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: INR Forward V7 ({n_params:,} parameters)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    run_id = datetime.now().strftime("run_%Y-%m-%d_%H%M%S_v7_inr")
    if args.dry_run:
        run_id += "_dryrun"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    with open(run_dir / "config.json", "w") as f:
        json.dump({"args": vars(args), "model": "v7_inr"}, f, indent=2)

    print(f"\nStarting INR training: {args.epochs} epochs")
    bar = ProgressBar(args.epochs)
    best_avg = float("inf")
    epochs_no_improve = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_sdd11": [], "val_sdd21": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, freqs_hz, device, weight44, optimizer)
        val_stats = eval_one_epoch(model, val_loader, freqs_hz, device, weight44)
        
        avg_db = 0.5 * (val_stats["mae_sdd11_db"] + val_stats["mae_sdd21_db"])
        scheduler.step(avg_db)

        is_best = avg_db < best_avg
        marker = "*" if is_best else " "
        if is_best:
            best_avg = avg_db
            epochs_no_improve = 0
            torch.save(model.state_dict(), run_dir / "checkpoint_best.pt")
        else:
            epochs_no_improve += 1

        metrics_str = (
            f"tL:{train_stats['loss']:.2f} vL:{val_stats['loss']:.2f} "
            f"PassPen:{train_stats['l_passivity']:.4f} "
            f"S11:{val_stats['mae_sdd11_db']:5.2f}dB S21:{val_stats['mae_sdd21_db']:5.2f}dB "
            f"avg:{avg_db:5.2f}dB {marker}"
        )
        bar.render(epoch, metrics_str)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["val_sdd11"].append(val_stats["mae_sdd11_db"])
        history["val_sdd21"].append(val_stats["mae_sdd21_db"])

        if not args.dry_run and epochs_no_improve >= args.patience:
            bar.finish(f"\n[EARLY STOP] No improvement for {args.patience} epochs.")
            break

    bar.finish()

    print(f"\nLoading best checkpoint for diagnostics")
    model.load_state_dict(torch.load(run_dir / "checkpoint_best.pt", weights_only=True))
    plot_predictions(model, val_ds, freqs_hz, device, run_dir / "val_predictions.png", seed=args.seed)
    print(f"Saved: {run_dir / 'val_predictions.png'}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].set_yscale("log"); axes[0].legend()
    axes[1].plot(history["epoch"], history["val_sdd11"], label="Sdd11")
    axes[1].plot(history["epoch"], history["val_sdd21"], label="Sdd21")
    axes[1].axhline(2.0, color="green", linestyle=":", label="target 2 dB")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("MAE [dB]"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curves.png", dpi=110)

if __name__ == "__main__":
    main()