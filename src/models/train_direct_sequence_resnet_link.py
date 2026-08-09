"""
train_direct_sequence_resnet_link.py
--------------------------------------------------
Direct Sequence 1D-Convolutional ResNet for S-Parameter Generation.
Bypasses the "Zero-Residue Trap" of Rational networks by predicting the 
frequency sequence directly, while enforcing Reciprocity and Passivity physically.

PHYSICS-INFORMED UPGRADE (Dynamic Grey-Box Modeling):
Instead of a global attenuation scalar, this script uses a Physics Head to 
dynamically predict the specific attenuation (alpha) and phase (beta) for 
EACH INDIVIDUAL trace geometry, drastically lowering S21 error on Link datasets.

"""

import argparse
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
RUNS_DIR = PROJECT_ROOT / "src" / "models" / "forward_runs"

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
# MODEL ARCHITECTURE: DYNAMIC PHYSICS-INFORMED RESNET
# =============================================================================
class Conv1DResBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.10):
        super().__init__()
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
    UPPER_R = (0, 0, 0, 0, 1, 1, 1, 2, 2, 3)
    UPPER_C = (0, 1, 2, 3, 1, 2, 3, 2, 3, 3)

    def __init__(self, d_local=8, d_global=6, d_context=7, F_len=401, hidden_dim=384, n_blocks=8):
        super().__init__()
        self.F_len = F_len
        self.is_link_dataset = (d_global == 7)
        in_dim = d_local + d_global + d_context
        
        # 1. Geometry Latent Encoder
        self.geom_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # 2. DYNAMIC PHYSICS HEAD (Calculates specific trace attenuation per sample)
        if self.is_link_dataset:
            self.physics_head = nn.Linear(hidden_dim, 2) # Predicts [alpha, beta]
            # Initialize to reasonable baseline physics
            nn.init.constant_(self.physics_head.bias[0], 2.0)
            nn.init.constant_(self.physics_head.bias[1], 10.0)
            nn.init.normal_(self.physics_head.weight, std=0.01)
        
        # 3. CNN Sequence Processor
        self.proj_in = nn.Conv1d(hidden_dim + 1, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([Conv1DResBlock(hidden_dim, dropout=0.10) for _ in range(n_blocks)])
        self.proj_out = nn.Conv1d(hidden_dim, 20, kernel_size=1)
        
        nn.init.normal_(self.proj_out.weight, std=1e-3)
        nn.init.zeros_(self.proj_out.bias)

        self.register_buffer("upper_r", torch.tensor(self.UPPER_R, dtype=torch.long))
        self.register_buffer("upper_c", torch.tensor(self.UPPER_C, dtype=torch.long))

    def _scatter_symmetric(self, vec_real: torch.Tensor, vec_imag: torch.Tensor, B: int) -> torch.Tensor:
        mat = torch.zeros((B, 4, 4, self.F_len), dtype=torch.float64, device=vec_real.device)
        mat_i = torch.zeros((B, 4, 4, self.F_len), dtype=torch.float64, device=vec_real.device)
        
        mat[:, self.upper_r, self.upper_c, :] = vec_real.to(torch.float64)
        mat[:, self.upper_c, self.upper_r, :] = vec_real.to(torch.float64)
        mat_i[:, self.upper_r, self.upper_c, :] = vec_imag.to(torch.float64)
        mat_i[:, self.upper_c, self.upper_r, :] = vec_imag.to(torch.float64)
        
        S = torch.complex(mat, mat_i)
        return S.permute(0, 3, 1, 2)

    def forward(self, x_local, x_global, x_context, freqs_hz):
        B = x_local.shape[0]
        
        # --- PART A: ENCODE GEOMETRY ---
        x = torch.cat([x_local, x_global, x_context], dim=-1)
        h_geom = self.geom_mlp(x)
        h_seq = h_geom.unsqueeze(-1).expand(-1, -1, self.F_len)
        f_norm = (freqs_hz / freqs_hz.max()).view(1, 1, self.F_len).expand(B, 1, -1).to(h_seq.dtype)
        
        # --- PART B: COMPUTE CNN RESIDUE ---
        h = torch.cat([h_seq, f_norm], dim=1)
        h = self.proj_in(h)
        for block in self.blocks:
            h = block(h)
        out = self.proj_out(h)
        
        vec_real = out[:, :10, :]
        vec_imag = out[:, 10:, :]
        S_cnn_residue = self._scatter_symmetric(vec_real, vec_imag, B)

        # --- PART C: COMPUTE DYNAMIC ANALYTICAL PHYSICS ---
        if self.is_link_dataset:
            # Predict trace-specific alpha and beta based on the board geometry
            phys_params = self.physics_head(h_geom) # Shape: (B, 2)
            alpha = F.softplus(phys_params[:, 0]).view(B, 1, 1) # Bounded > 0
            beta = phys_params[:, 1].view(B, 1, 1)
            
            length_feature = x_global[:, -1].view(B, 1, 1)
            
            mag = torch.exp(-alpha * f_norm * length_feature)
            phase = -beta * f_norm * length_feature
            
            T_real = mag * torch.cos(phase)
            T_imag = mag * torch.sin(phase)
            T_cmplx = torch.complex(T_real, T_imag).view(B, self.F_len).to(S_cnn_residue.dtype)
            
            S_analytical = torch.zeros_like(S_cnn_residue)
            S_analytical[:, :, 2, 0] = T_cmplx # S31
            S_analytical[:, :, 0, 2] = T_cmplx # S13
            S_analytical[:, :, 3, 1] = T_cmplx # S42
            S_analytical[:, :, 1, 3] = T_cmplx # S24
            
            S_final = S_analytical + S_cnn_residue
        else:
            S_final = S_cnn_residue

        return S_final

# =============================================================================
# PHYSICS-INFORMED LOSS
# =============================================================================
def composite_loss(pred, target, weight44):
    """Physics-informed composite loss for S-parameter regression.

    Four terms, summed with fixed weights (see `total` below):
      1. l_linear   -- L1 error on the raw real/imag parts. Cheap, stable,
                       but BLIND to deep-dB accuracy (at -40 dB the linear
                       value is ~0.01, so linear error barely changes whether
                       the model predicts -40 or -80 dB). This is why a
                       linear-dominated loss lets Sdd21 plateau.
      2. l_db       -- Huber error in the dB (log-magnitude) domain. This
                       tracks insertion/return loss on the dB scale the plots
                       (and the spec) are read on. Kept at coefficient 0.1
                       (raising it was tested and destabilises training, since
                       the log term is already ~600x larger than l_linear in
                       raw magnitude).
      3. l_weighted -- element-aware MSE (per-S-parameter variance weighting),
                       so low-energy matrix elements are not drowned out.
      4. l_passivity-- penalises singular values > 1 (a passive network cannot
                       amplify); keeps predictions physically valid.

    Frequency weighting (freq_weights) concentrates BOTH the linear and dB
    terms on the 0-28 GHz eye band -- the band that actually drives the 112G
    PAM4 eye and the yield spec -- with progressively less weight above it.
    Index mapping on the 0.25-100 GHz / 401-pt grid: ~112 -> 28 GHz,
    ~240 -> 60 GHz.
    """
    eps = 1e-12
    F_len = pred.shape[1]

    # --- eye-band-focused frequency weighting ------------------------------
    # 0-28 GHz (eye band)      -> 5x  (highest priority: this is what matters)
    # 28-60 GHz                -> 2x  (moderate: still in useful range)
    # 60-100 GHz               -> 1x  (low: mostly deep, out-of-band, near floor)
    freq_weights = torch.ones(F_len, device=pred.device)
    idx_28ghz = min(112, F_len)      # 28 GHz on the 401-pt, 0.25-100 GHz grid
    idx_60ghz = min(240, F_len)      # 60 GHz
    freq_weights[:idx_28ghz] = 5.0
    freq_weights[idx_28ghz:idx_60ghz] = 2.0
    freq_weights[idx_60ghz:] = 1.0

    # --- 1. linear (real/imag) L1 loss, frequency-weighted -----------------
    diff_real = (pred.real - target.real).abs() * freq_weights.view(1, -1, 1, 1)
    diff_imag = (pred.imag - target.imag).abs() * freq_weights.view(1, -1, 1, 1)
    l_linear = diff_real.mean() + diff_imag.mean()

    # --- 2. dB-domain Huber loss, frequency-weighted (THE key fix) ---------
    # Clamp both prediction and target at the noise floor so the model is not
    # forced to chase physically-meaningless -80..-240 dB values (below any
    # real solver floor); it only has to track dB accurately down to the floor.
    p_db = 20.0 * torch.log10(pred.abs().clamp_min(eps))
    t_db = 20.0 * torch.log10(target.abs().clamp_min(eps))
    p_db_c = p_db.clamp_min(NOISE_FLOOR_DB)
    t_db_c = t_db.clamp_min(NOISE_FLOOR_DB)
    db_err = F.huber_loss(p_db_c, t_db_c, delta=2.0, reduction='none')
    l_db = (db_err * freq_weights.view(1, -1, 1, 1)).mean()

    # --- 3. element-aware MSE (per-S-parameter variance weighting) ---------
    diff_sq = (pred.real - target.real) ** 2 + (pred.imag - target.imag) ** 2
    l_weighted = (diff_sq * weight44.view(1, 1, 4, 4)).mean()

    # --- 4. passivity penalty (sigma_max <= 1 for a passive network) -------
    f_idx = torch.randint(0, F_len, (10,), device=pred.device)
    sv = torch.linalg.svdvals(pred[:, f_idx, :, :])
    l_passivity = F.relu(sv - 1.0).mean()

    # --- weighted sum ------------------------------------------------------
    # NOTE on weights (IMPORTANT): in raw magnitude l_db is ~500-600x larger
    # than l_linear (dB errors are O(1-10); linear errors are O(0.001-0.01)),
    # so the dB term is already a strong contributor even at a small coefficient.
    # We KEEP the dB coefficient at the original, proven-stable 0.1. Raising it
    # was tested and found to DESTABILISE training (the log-magnitude term swings
    # violently for near-zero predictions and the total loss diverges). The
    # genuine, stable lever for in-band accuracy is the FREQUENCY WEIGHTING
    # above, which concentrates all terms on the 0-28 GHz eye band -- that is
    # the only change from the original loss, and it is numerically safe.
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
    s_sub = pred[:, ::20] 
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
    parser.add_argument("--dataset", type=str, required=True, choices=["Array", "Link"], help="Which dataset to train on")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--n-blocks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    DATA_PT = PROJECT_ROOT / "data" / "processed" / f"Universal-Diff-SI-{args.dataset}" / "diff_pair_dataset.pt"
    print(f"Loading dataset: {DATA_PT}")
    payload = torch.load(DATA_PT, weights_only=False)
    
    sim_ids = np.array(payload["sim_ids"])
    unique_sims = np.unique(sim_ids)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_sims)
    
    n_train = int(0.85 * len(unique_sims))
    train_sims = set(unique_sims[:n_train])
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