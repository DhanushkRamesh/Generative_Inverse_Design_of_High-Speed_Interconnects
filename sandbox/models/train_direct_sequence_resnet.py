"""
train_direct_sequence_resnet.py
--------------------------------------------------
Direct Sequence 1D-Convolutional ResNet for S-Parameter Generation.
Bypasses the "Zero-Residue Trap" of Rational networks by predicting the 
frequency sequence directly, while enforcing Reciprocity and Passivity physically.

Automatically adapts to both Array (21D) and Link (23D) datasets.

Author: Lead ML/EM Researcher
"""

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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# =============================================================================
# HARDWARE & PATH CONFIGURATION
# =============================================================================
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

PROJECT_ROOT = Path.home() / "mece_project_inverse_model" / "Generative_Inverse_Design_of_High-Speed_Interconnects"
SPLITS_PT = PROJECT_ROOT / "sandbox_v1" / "data" / "splits.pt"
RUNS_DIR = PROJECT_ROOT / "sandbox_v1" / "models" / "forward_runs"

# Goldilocks Noise Floor: -55 dB fits too much HFSS numerical mesh noise. 
# -45 dB perfectly isolates the deep crosstalk nulls without overfitting to static.
NOISE_FLOOR_DB = -45.0

# =============================================================================
# DATASET AND WEIGHTS
# =============================================================================
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
                             n_sample: int = 500, seed: int = 0, floor: float = 1e-3):
    rng = np.random.default_rng(seed)
    pick = rng.choice(train_idx, size=min(n_sample, len(train_idx)), replace=False)
    yr = payload["Y_real"][pick].to(torch.float64)
    yi = payload["Y_imag"][pick].to(torch.float64)
    mag_sq = yr ** 2 + yi ** 2
    mean_per_elem = mag_sq.mean(dim=(0, 1))
    weight = 1.0 / mean_per_elem.clamp_min(floor)
    weight = weight * (16.0 / weight.sum())
    return weight, mean_per_elem

# =============================================================================
# MODEL ARCHITECTURE: 1D-CONV RESNET
# =============================================================================
class Conv1DResBlock(nn.Module):
    """1D Convolutional Residual Block ensuring frequency-domain smoothness."""
    def __init__(self, channels: int, dropout: float = 0.10):
        super().__init__()
        # Using replicate padding to avoid edge artifacts at 0 Hz and 100 GHz
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2, padding_mode='replicate')
        self.norm1 = nn.GroupNorm(4, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2, padding_mode='replicate')
        self.norm2 = nn.GroupNorm(4, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(self.conv1(h))
        h = self.drop(h)
        h = self.norm2(h)
        h = self.conv2(h)
        return x + h

class DirectSequenceResNet(nn.Module):
    """
    Maps Geometry -> 401-point sequence directly via 1D Convolutions.
    Structurally guarantees S = S^T.
    Automatically adapts its input layer to Array (21 features) or Link (23 features).
    """
    UPPER_R = (0, 0, 0, 0, 1, 1, 1, 2, 2, 3)
    UPPER_C = (0, 1, 2, 3, 1, 2, 3, 2, 3, 3)

    def __init__(self, d_local=8, d_global=6, d_context=7, F_len=401, hidden_dim=384, n_blocks=8):
        super().__init__()
        self.F_len = F_len
        in_dim = d_local + d_global + d_context
        
        # Geometry Latent Encoder
        self.geom_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # 1D Sequence Processor
        # Channels: hidden_dim (geometry) + 1 (frequency positional encoding)
        self.proj_in = nn.Conv1d(hidden_dim + 1, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([Conv1DResBlock(hidden_dim, dropout=0.10) for _ in range(n_blocks)])
        self.proj_out = nn.Conv1d(hidden_dim, 20, kernel_size=1) # 10 Real, 10 Imag
        
        # Initialize output close to 0 for stability
        nn.init.normal_(self.proj_out.weight, std=1e-3)
        nn.init.zeros_(self.proj_out.bias)

        self.register_buffer("upper_r", torch.tensor(self.UPPER_R, dtype=torch.long))
        self.register_buffer("upper_c", torch.tensor(self.UPPER_C, dtype=torch.long))

    def _scatter_symmetric(self, vec_real: torch.Tensor, vec_imag: torch.Tensor, B: int) -> torch.Tensor:
        """Transforms (B, 10, F) back to symmetric (B, F, 4, 4) complex matrices."""
        mat = torch.zeros((B, 4, 4, self.F_len), dtype=torch.float64, device=vec_real.device)
        mat_i = torch.zeros((B, 4, 4, self.F_len), dtype=torch.float64, device=vec_real.device)
        
        mat[:, self.upper_r, self.upper_c, :] = vec_real.to(torch.float64)
        mat[:, self.upper_c, self.upper_r, :] = vec_real.to(torch.float64)
        
        mat_i[:, self.upper_r, self.upper_c, :] = vec_imag.to(torch.float64)
        mat_i[:, self.upper_c, self.upper_r, :] = vec_imag.to(torch.float64)
        
        S = torch.complex(mat, mat_i)
        return S.permute(0, 3, 1, 2) # Returns (B, F, 4, 4)

    def forward(self, x_local, x_global, x_context, freqs_hz):
        B = x_local.shape[0]
        
        # 1. Encode Geometry
        x = torch.cat([x_local, x_global, x_context], dim=-1)
        h_geom = self.geom_mlp(x) # (B, hidden_dim)
        
        # 2. Expand to Sequence and inject frequency coordinates [0 to 1]
        h_seq = h_geom.unsqueeze(-1).expand(-1, -1, self.F_len) # (B, hidden_dim, F)
        f_norm = (freqs_hz / freqs_hz.max()).view(1, 1, self.F_len).expand(B, 1, -1).to(h_seq.dtype)
        
        # 3. Process Sequence
        h = torch.cat([h_seq, f_norm], dim=1) # (B, hidden_dim + 1, F)
        h = self.proj_in(h)
        for block in self.blocks:
            h = block(h)
        out = self.proj_out(h) # (B, 20, F)
        
        vec_real = out[:, :10, :]
        vec_imag = out[:, 10:, :]
        
        return self._scatter_symmetric(vec_real, vec_imag, B)

# =============================================================================
# PHYSICS-INFORMED LOSS
# =============================================================================
def composite_loss(pred, target, weight44):
    """Direct Sequence Loss: L1 Phase + Huber Magnitude + WL2 + SVD Passivity"""
    eps = 1e-12
    l_linear = (pred.real - target.real).abs().mean() + (pred.imag - target.imag).abs().mean()

    p_db = 20.0 * torch.log10(pred.abs().clamp_min(eps))
    t_db = 20.0 * torch.log10(target.abs().clamp_min(eps))
    p_db_c = p_db.clamp_min(NOISE_FLOOR_DB)
    t_db_c = t_db.clamp_min(NOISE_FLOOR_DB)
    l_db = F.huber_loss(p_db_c, t_db_c, delta=2.0)

    diff_sq = (pred.real - target.real) ** 2 + (pred.imag - target.imag) ** 2
    l_weighted = (diff_sq * weight44.view(1, 1, 4, 4)).mean()

    # Fast Passivity Penalty (Evaluate on 10 random frequencies per batch)
    F_len = pred.shape[1]
    f_idx = torch.randint(0, F_len, (10,), device=pred.device)
    sv = torch.linalg.svdvals(pred[:, f_idx, :, :])
    
    # Passivity multiplier set to 10.0 (Goldilocks zone between 5.0 and 20.0)
    l_passivity = F.relu(sv - 1.0).mean()

    total = 1.0 * l_linear + 0.1 * l_db + 0.5 * l_weighted + 10.0 * l_passivity

    return {
        "loss": total,
        "l_linear": l_linear.detach().item(),
        "l_db": l_db.detach().item(),
        "l_passivity": l_passivity.detach().item()
    }

@torch.no_grad()
def passband_mae_db(pred, target, i, j):
    eps = 1e-12
    p_db = 20.0 * torch.log10(pred[..., i, j].abs() + eps)
    t_db = 20.0 * torch.log10(target[..., i, j].abs() + eps)
    mask = t_db > NOISE_FLOOR_DB
    if mask.sum() == 0: return float("nan")
    return (p_db[mask] - t_db[mask]).abs().mean().item()

@torch.no_grad()
def passivity_max_sigma(pred):
    s_sub = pred[:, ::20] # Downsample for speed
    sv = torch.linalg.svdvals(s_sub)
    return sv.max().item()

# =============================================================================
# PIPELINE UTILS
# =============================================================================
def train_one_epoch(model, loader, freqs_hz, device, weight44, optimizer):
    model.train()
    acc = {"loss": 0.0, "l_pass": 0.0, "sdd11": 0.0, "sdd21": 0.0, "n": 0}
    for xl, xg, xc, S_tgt in loader:
        xl, xg, xc = xl.to(device), xg.to(device), xc.to(device)
        S_tgt = S_tgt.to(device)
        B = xl.shape[0]

        pred = model(xl, xg, xc, freqs_hz)
        ld = composite_loss(pred, S_tgt, weight44)

        optimizer.zero_grad(set_to_none=True)
        ld["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        acc["loss"] += ld["loss"].detach().item() * B
        acc["l_pass"] += ld["l_passivity"] * B
        acc["sdd11"] += passband_mae_db(pred, S_tgt, 0, 0) * B
        acc["sdd21"] += passband_mae_db(pred, S_tgt, 1, 0) * B
        acc["n"] += B
    n = acc["n"]
    return {k: v/n for k, v in acc.items() if k != "n"}

@torch.no_grad()
def eval_one_epoch(model, loader, freqs_hz, device, weight44):
    model.eval()
    acc = {"loss": 0.0, "sdd11": 0.0, "sdd21": 0.0, "pass_max": 0.0, "n": 0}
    for xl, xg, xc, S_tgt in loader:
        xl, xg, xc = xl.to(device), xg.to(device), xc.to(device)
        S_tgt = S_tgt.to(device)
        B = xl.shape[0]
        
        pred = model(xl, xg, xc, freqs_hz)
        ld = composite_loss(pred, S_tgt, weight44)
        
        acc["loss"] += ld["loss"].item() * B
        acc["sdd11"] += passband_mae_db(pred, S_tgt, 0, 0) * B
        acc["sdd21"] += passband_mae_db(pred, S_tgt, 1, 0) * B
        acc["pass_max"] = max(acc["pass_max"], passivity_max_sigma(pred))
        acc["n"] += B
    n = acc["n"]
    return {k: (v/n if k != "pass_max" else v) for k, v in acc.items() if k != "n"}

@torch.no_grad()
def plot_predictions(model, dataset, freqs_hz, device, out_path, seed=0):
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(dataset), size=6, replace=False)
    model.eval()
    fig, axes = plt.subplots(2, 6, figsize=(20, 6), sharex=True)
    f_ghz = freqs_hz.cpu().numpy() / 1e9
    
    for col, idx in enumerate(pick):
        xl, xg, xc, S_tgt = dataset[int(idx)]
        xl, xg, xc = xl.unsqueeze(0).to(device), xg.unsqueeze(0).to(device), xc.unsqueeze(0).to(device)
        S_pred = model(xl, xg, xc, freqs_hz).squeeze(0).cpu()
        
        for row, (i, j, lbl) in enumerate([(0, 0, "Sdd11"), (1, 0, "Sdd21")]):
            ax = axes[row, col]
            ax.plot(f_ghz, 20 * np.log10(S_tgt[:, i, j].abs().numpy() + 1e-12), "b-", lw=1, label="Target" if col==0 and row==0 else "")
            ax.plot(f_ghz, 20 * np.log10(S_pred[:, i, j].abs().numpy() + 1e-12), "r--", lw=1, label="CNN Pred" if col==0 and row==0 else "")
            ax.axhline(NOISE_FLOOR_DB, color="gray", ls=":", lw=0.6)
            ax.set_ylabel(f"|{lbl}| [dB]")
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(f"Idx {int(idx)}", fontsize=9)
                if col == 0: ax.legend(fontsize=8)
            else:
                ax.set_xlabel("Freq [GHz]")
                
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    # ADDED CLI ARGUMENT TO ROUTE BETWEEN DATASETS
    parser.add_argument("--dataset", type=str, required=True, choices=["Array", "Link"], help="Which dataset to train on")
    
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=35)
    
    # GOLDILOCKS ARCHITECTURE DEFAULTS
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--n-blocks", type=int, default=8)
    
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # DYNAMIC DATA PATH ROUTING
    DATA_PT = PROJECT_ROOT / "data" / "processed" / f"Universal-Diff-SI-{args.dataset}" / "diff_pair_dataset.pt"
    
    print(f"Loading dataset: {DATA_PT}")
    payload = torch.load(DATA_PT, weights_only=False)
    
    # DYNAMIC, LEAKAGE-FREE SPLIT LOGIC
    sim_ids = np.array(payload["sim_ids"])
    unique_sims = np.unique(sim_ids)
    
    # Shuffle unique simulations deterministically using the seed
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_sims)
    
    # 85% Train / 15% Val split (at the simulation level)
    n_train = int(0.85 * len(unique_sims))
    train_sims = set(unique_sims[:n_train])
    
    # Map back to the individual pair indices
    train_idx = np.array([i for i, sid in enumerate(sim_ids) if sid in train_sims])
    val_idx = np.array([i for i, sid in enumerate(sim_ids) if sid not in train_sims])
    
    print(f"Data Split: {len(train_idx)} train pairs, {len(val_idx)} val pairs")
    
    weight44, _ = compute_element_weights(payload, train_idx, seed=args.seed)
    weight44 = weight44.to(device)

    train_ds = DiffPairDataset(payload, train_idx)
    val_ds = DiffPairDataset(payload, val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    freqs_hz = payload["frequencies"].to(device)

    # DYNAMIC INITIALIZATION: Automatically adapts to 21D (Array) or 23D (Link)
    model = DirectSequenceResNet(
        d_local=payload["X_local"].shape[1],
        d_global=payload["X_global"].shape[1],
        d_context=payload["X_context"].shape[1],
        F_len=len(freqs_hz),
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks
    ).to(device)
    
    print(f"\nModel: Direct Sequence 1D-ResNet ({sum(p.numel() for p in model.parameters()):,} parameters)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    run_dir = RUNS_DIR / datetime.now().strftime(f"run_%Y-%m-%d_%H%M%S_direct_resnet_{args.dataset.lower()}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting Training for {args.dataset}: {args.epochs} epochs")
    best_avg = float("inf")
    epochs_no_improve = 0
    history = {"epoch": [], "t_loss": [], "v_loss": [], "v_s11": [], "v_s21": [], "pass_max": []}

    for epoch in range(1, args.epochs + 1):
        t_stats = train_one_epoch(model, train_loader, freqs_hz, device, weight44, optimizer)
        v_stats = eval_one_epoch(model, val_loader, freqs_hz, device, weight44)
        
        avg_db = 0.5 * (v_stats["sdd11"] + v_stats["sdd21"])
        scheduler.step(avg_db)

        is_best = avg_db < best_avg
        marker = "[NEW BEST]" if is_best else ""
        if is_best:
            best_avg = avg_db
            epochs_no_improve = 0
            torch.save(model.state_dict(), run_dir / "checkpoint_best.pt")
        else:
            epochs_no_improve += 1

        lr = optimizer.param_groups[0]["lr"]
        
        # LINE-BY-LINE CONSOLE OUTPUT
        metrics = (
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"tL:{t_stats['loss']:.2f} vL:{v_stats['loss']:.2f} | "
            f"PassPen:{t_stats['l_pass']:.4f} | "
            f"S11:{v_stats['sdd11']:5.2f}dB S21:{v_stats['sdd21']:5.2f}dB | "
            f"Avg:{avg_db:5.2f}dB | LR:{lr:.1e} {marker}"
        )
        print(metrics)

        history["epoch"].append(epoch); history["t_loss"].append(t_stats["loss"])
        history["v_loss"].append(v_stats["loss"]); history["v_s11"].append(v_stats["sdd11"])
        history["v_s21"].append(v_stats["sdd21"]); history["pass_max"].append(v_stats["pass_max"])

        if epochs_no_improve >= args.patience:
            print(f"\n[EARLY STOP] No improvement for {args.patience} epochs.")
            break
            
    print(f"\nLoading best checkpoint & Generating Plots...")
    model.load_state_dict(torch.load(run_dir / "checkpoint_best.pt", weights_only=True))
    plot_predictions(model, val_ds, freqs_hz, device, run_dir / "val_predictions.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["epoch"], history["t_loss"], label="Train")
    axes[0].plot(history["epoch"], history["v_loss"], label="Val")
    axes[0].set_yscale("log"); axes[0].set_title("Loss"); axes[0].grid(alpha=0.3); axes[0].legend()
    
    axes[1].plot(history["epoch"], history["v_s11"], label="Sdd11")
    axes[1].plot(history["epoch"], history["v_s21"], label="Sdd21")
    axes[1].axhline(2.0, color="g", ls=":", label="2 dB Target")
    axes[1].set_title("Validation MAE [dB]"); axes[1].grid(alpha=0.3); axes[1].legend()

    axes[2].plot(history["epoch"], history["pass_max"], color="r")
    axes[2].axhline(1.0, color="k", ls="--", label="Passivity Limit")
    axes[2].set_title("Max Singular Value (Passivity)"); axes[2].grid(alpha=0.3); axes[2].legend()

    plt.tight_layout()
    plt.savefig(run_dir / "loss_curves.png", dpi=120)
    plt.close()

    print(f"Run Directory: {run_dir}")
    print(f"Best Validation Avg MAE: {best_avg:.2f} dB")

if __name__ == "__main__":
    main()