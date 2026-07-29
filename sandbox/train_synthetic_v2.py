"""
==========================================================================
TRAINING SCRIPT: PRNet on Realistic Synthetic Benchmark
==========================================================================
This is the FINAL training script that validates whether your Rational Layer
architecture can learn a nonlinear geometry -> S-parameter mapping.

Key improvements over previous training scripts:
  1. Network sized for 8 poles (matching oracle max)
  2. Deeper MLP with residual connections (learns nonlinear mapping)
  3. Multi-objective loss: complex MSE + dB Huber + phase penalty
  4. Comprehensive evaluation with per-sample breakdown
  5. Publishable diagnostic plots

Success Criterion: Val MAE < 0.5 dB
  -> If achieved: Rational Layer is validated, move to TUHH
  -> If NOT achieved: architecture needs rework before TUHH attempt
==========================================================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import math


# =====================================================================
# 1. Dataset Loader (unchanged — compatible with your pipeline)
# =====================================================================
class SyntheticPoleDataset(Dataset):
    def __init__(self, pt_file_path):
        data = torch.load(pt_file_path)
        self.X = torch.cat([data['X_global'], data['X_local']], dim=1)
        
        y_r = data['Y_real'][:, :, :, 0] if data['Y_real'].dim() == 4 else data['Y_real']
        y_i = data['Y_imag'][:, :, :, 0] if data['Y_imag'].dim() == 4 else data['Y_imag']
        self.Y = torch.complex(y_r, y_i)
        
        self.freqs_ghz = data['frequencies'].numpy() / 1e9
        
        # Store ground truth pole counts if available
        self.gt_num_poles = data.get('gt_num_poles', None)
        
        print(f"Loaded {len(self.X)} samples | "
              f"Freq: {self.freqs_ghz[0]:.2f}-{self.freqs_ghz[-1]:.2f} GHz | "
              f"Points: {len(self.freqs_ghz)}")

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]


# =====================================================================
# 2. PRODUCTION PRNet — Sized for Real Problems
# =====================================================================
# This is still YOUR Rational Layer (S(s) = sum R/(s-P) + D).
# The only change is proper capacity: deeper MLP, residual blocks,
# and 8 poles to match the oracle's maximum.
# =====================================================================

class ResidualBlock(nn.Module):
    """Pre-activation residual block with LayerNorm."""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
    
    def forward(self, x):
        return x + self.block(x)


class ProductionPRNet(nn.Module):
    """
    Physics-Constrained Rational Layer with production-grade backbone.
    
    Architecture:
      Input (10 features) -> Stem (10->512) -> 3x ResBlocks -> Head (512->out)
      
    Output parameterization (per target):
      - alpha:  sigmoid -> [-3.0, -0.05]   (damping, guarantees stability)
      - f_res:  sigmoid -> [0, 100] GHz     (resonance frequency)
      - c_re:   sigmoid -> [-40, 40]        (residue real part)
      - c_im:   sigmoid -> [-40, 40]        (residue imaginary part)
      - d_re:   unbounded                   (direct term real)
      - d_im:   unbounded                   (direct term imag)
      - gamma:  softplus -> [0, inf)        (loss envelope coefficient)
    
    Total output: (4*num_poles + 2) * num_targets + 1 = (4*8+2)*2 + 1 = 69
    """
    def __init__(self, input_dim=10, num_poles=8, num_targets=2, hidden_dim=512):
        super().__init__()
        self.num_targets = num_targets
        self.num_poles = num_poles
        
        self.params_per_target = (4 * num_poles) + 2  # 34 per target
        total_out = (self.params_per_target * num_targets) + 1  # +1 for gamma
        
        # Stem: project input to hidden dimension
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Body: 3 residual blocks (deep enough for nonlinear mapping)
        self.body = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )
        
        # Head: project to output parameters
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, total_out),
        )
        
        # --- SPREADING INITIALIZATION ---
        # Spread poles evenly across 0-100 GHz band
        target_f = torch.linspace(0.05, 0.95, num_poles)
        inv_sig = torch.log(target_f / (1.0 - target_f))
        
        with torch.no_grad():
            final_layer = self.head[-1]
            for t in range(num_targets):
                offset = t * self.params_per_target
                # Initialize f_res bias to spread poles
                final_layer.bias[offset + num_poles : offset + 2*num_poles] = inv_sig
                # Initialize residues to non-zero
                final_layer.bias[offset + 2*num_poles : offset + 4*num_poles] = 0.5
            # Initialize gamma to small value
            final_layer.bias[-1] = 0.0
    
    def forward(self, x, s_tensor, freqs_ghz):
        batch = x.shape[0]
        num_freqs = s_tensor.shape[0]
        
        # MLP backbone
        h = self.stem(x)
        h = self.body(h)
        raw_out = self.head(h)
        
        # Split into per-target params and gamma
        pr_params = raw_out[:, :-1].view(batch, self.num_targets, self.params_per_target)
        gamma_raw = raw_out[:, -1]
        
        # === PHYSICS-CONSTRAINED ACTIVATION FUNCTIONS ===
        
        # Alpha (damping): Must be negative for stability
        # Minimum damping of -0.3 to match generator and prevent near-singular poles
        alpha = -(torch.sigmoid(pr_params[:, :, :self.num_poles]) * 2.7 + 0.3)
        
        # Resonance frequency: 0-100 GHz
        f_res = torch.sigmoid(pr_params[:, :, self.num_poles:2*self.num_poles]) * 100.0
        beta = 2 * math.pi * f_res
        
        # Residues: Moderate range (matched to generator)
        c_re = (torch.sigmoid(pr_params[:, :, 2*self.num_poles:3*self.num_poles]) - 0.5) * 40.0
        c_im = (torch.sigmoid(pr_params[:, :, 3*self.num_poles:4*self.num_poles]) - 0.5) * 40.0
        
        # Direct term
        d_re = pr_params[:, :, -2].unsqueeze(-1)
        d_im = pr_params[:, :, -1].unsqueeze(-1)
        
        # === POLE-RESIDUE SYNTHESIS (Your Rational Layer Equation) ===
        # S(s) = sum_n [ c_n/(s-p_n) + conj(c_n)/(s-conj(p_n)) ] + d
        
        p = torch.complex(alpha, beta).unsqueeze(-1)           # (B, T, P, 1)
        c = torch.complex(c_re, c_im).unsqueeze(-1)            # (B, T, P, 1)
        d = torch.complex(d_re, d_im)                          # (B, T, 1)
        s_view = s_tensor.view(1, 1, 1, num_freqs)             # (1, 1, 1, F)
        
        # NUMERICAL SAFETY: Add small epsilon to denominator to prevent inf
        # When s ≈ p (frequency near pole), |s-p| can be tiny -> division explodes
        denom1 = s_view - p
        denom2 = s_view - torch.conj(p)
        # Clamp denominator magnitude away from zero
        eps = 1e-6
        safe_denom1 = denom1 + eps * torch.sign(denom1.real + 1e-10)
        safe_denom2 = denom2 + eps * torch.sign(denom2.real + 1e-10)
        
        term1 = c / safe_denom1
        term2 = torch.conj(c) / safe_denom2
        H_s = torch.sum(term1 + term2, dim=2) + d              # (B, T, F)
        
        # CLAMP output magnitude to prevent log10(inf) in loss
        H_s = torch.clamp(H_s.real, -100.0, 100.0) + 1j * torch.clamp(H_s.imag, -100.0, 100.0)
        
        # === LOSS ENVELOPE ON Sdd21 ===
        gamma = torch.nn.functional.softplus(gamma_raw).unsqueeze(-1)  # (B, 1)
        f_tensor = freqs_ghz.view(1, num_freqs)                       # (1, F)
        exp_decay = torch.exp(-gamma * f_tensor).to(dtype=torch.complex64)
        
        H_s11 = H_s[:, 0, :]
        H_s21 = H_s[:, 1, :] * exp_decay
        H_out = torch.stack([H_s11, H_s21], dim=1)
        
        return H_out.transpose(1, 2)  # (B, F, T)


# =====================================================================
# 3. LOSS FUNCTION — Multi-Objective for SI
# =====================================================================

def si_loss(pred, target):
    """
    Combined loss that captures both fine structure and envelope.
    All operations are numerically hardened against inf/nan.
    """
    # Term 1: Complex MSE (safe — no division or log involved)
    mse_complex = (nn.functional.mse_loss(pred.real, target.real) +
                   nn.functional.mse_loss(pred.imag, target.imag))
    
    # Term 2: dB Huber Loss
    # Use larger epsilon and clamp to prevent -inf from log10
    pred_mag = torch.clamp(torch.abs(pred), min=1e-7)
    target_mag = torch.clamp(torch.abs(target), min=1e-7)
    p_db = 20 * torch.log10(pred_mag)
    t_db = 20 * torch.log10(target_mag)
    # Clamp dB values to reasonable range
    p_db = torch.clamp(p_db, -100.0, 40.0)
    t_db = torch.clamp(t_db, -100.0, 40.0)
    db_huber = nn.functional.smooth_l1_loss(p_db, t_db)
    
    # Term 3: Phase penalty (cosine similarity — more numerically stable than atan2)
    # cos_sim = Re(pred * conj(target)) / (|pred| * |target|)
    dot_real = pred.real * target.real + pred.imag * target.imag
    phase_loss = 1.0 - (dot_real / (pred_mag * target_mag + 1e-8)).mean()
    
    return mse_complex + 0.1 * db_huber + 0.01 * phase_loss


# =====================================================================
# 4. TRAINING LOOP
# =====================================================================

def train(data_path, results_dir="realistic_benchmark_results", epochs=600):
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    ds = SyntheticPoleDataset(data_path)
    
    # 80/10/10 split
    n = len(ds)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    train_set, val_set, test_set = random_split(ds, [n_train, n_val, n_test])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=32)
    
    print(f"Split: {n_train} train / {n_val} val / {n_test} test")
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProductionPRNet(input_dim=10, num_poles=8, num_targets=2).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters on {device}")
    
    # Optimizer with warmup
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Frequency tensors — construct on device with explicit dtype
    f_ghz = torch.tensor(ds.freqs_ghz, dtype=torch.float32, device=device)
    # Build s = j*2*pi*f using real and imaginary parts separately (avoids Python complex->torch issues)
    omega = 2 * math.pi * f_ghz
    s_tensor = torch.complex(torch.zeros_like(omega), omega)  # s = j*omega
    
    # Training history
    history = {'train_loss': [], 'val_mae_db': [], 'lr': []}
    best_val_mae = float('inf')
    
    print(f"\n{'='*60}")
    print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Val MAE (dB)':>12} | {'Best':>8} | {'LR':>10}")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x, s_tensor, f_ghz)
            loss = si_loss(pred, y)
            
            # NaN guard — skip corrupted batches instead of destroying the model
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [WARNING] NaN/Inf loss at epoch {epoch+1}, skipping batch")
                optimizer.zero_grad()
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # --- VALIDATE ---
        model.eval()
        val_db_error = 0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                pred = model(xv, s_tensor, f_ghz)
                p_db = 20 * torch.log10(torch.clamp(torch.abs(pred), min=1e-7))
                t_db = 20 * torch.log10(torch.clamp(torch.abs(yv), min=1e-7))
                p_db = torch.clamp(p_db, -100.0, 40.0)
                t_db = torch.clamp(t_db, -100.0, 40.0)
                val_db_error += torch.abs(p_db - t_db).mean().item()
        
        avg_train = train_loss / len(train_loader)
        avg_val_db = val_db_error / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(avg_train)
        history['val_mae_db'].append(avg_val_db)
        history['lr'].append(current_lr)
        
        scheduler.step()
        
        # Checkpoint best model
        if avg_val_db < best_val_mae:
            best_val_mae = avg_val_db
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
        
        # Print progress
        if (epoch + 1) % 50 == 0 or epoch == 0:
            marker = " ***" if avg_val_db <= best_val_mae else ""
            print(f"{epoch+1:6d} | {avg_train:11.6f} | {avg_val_db:12.2f} | "
                  f"{best_val_mae:8.2f} | {current_lr:10.2e}{marker}")
    
    print(f"{'='*60}")
    print(f"Training complete. Best Val MAE: {best_val_mae:.2f} dB")
    
    # =====================================================================
    # 5. FINAL EVALUATION ON TEST SET
    # =====================================================================
    model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth")))
    model.eval()
    
    all_pred = []
    all_target = []
    with torch.no_grad():
        for xv, yv in test_loader:
            xv, yv = xv.to(device), yv.to(device)
            pred = model(xv, s_tensor, f_ghz)
            all_pred.append(pred.cpu())
            all_target.append(yv.cpu())
    
    all_pred = torch.cat(all_pred, dim=0)
    all_target = torch.cat(all_target, dim=0)
    
    pred_db = 20 * torch.log10(torch.clamp(torch.abs(all_pred), min=1e-7))
    target_db = 20 * torch.log10(torch.clamp(torch.abs(all_target), min=1e-7))
    pred_db = torch.clamp(pred_db, -100.0, 40.0)
    target_db = torch.clamp(target_db, -100.0, 40.0)
    per_sample_mae = torch.abs(pred_db - target_db).mean(dim=(1, 2))
    
    test_mae = per_sample_mae.mean().item()
    test_p95 = torch.quantile(per_sample_mae, 0.95).item()
    
    print(f"\n{'='*60}")
    print(f"TEST SET RESULTS ({n_test} samples)")
    print(f"{'='*60}")
    print(f"  Mean MAE:   {test_mae:.2f} dB")
    print(f"  95th %ile:  {test_p95:.2f} dB")
    print(f"  {'PASS' if test_mae < 0.5 else 'NEEDS WORK'}: "
          f"{'< 0.5 dB target achieved!' if test_mae < 0.5 else f'Target is < 0.5 dB, got {test_mae:.2f} dB'}")
    
    # =====================================================================
    # 6. PUBLICATION PLOTS
    # =====================================================================
    _plot_results(all_pred, all_target, ds.freqs_ghz, per_sample_mae,
                  history, best_val_mae, test_mae, results_dir)
    
    return model, history, test_mae


def _plot_results(pred, target, freqs, per_sample_mae, history, best_val, test_mae, save_dir):
    """Generate publication-quality result plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Rational Layer Benchmark — Test MAE: {test_mae:.2f} dB", 
                 fontsize=14, fontweight='bold')
    
    # --- Plot 1: Best test sample (Sdd21) ---
    best_idx = torch.argmin(per_sample_mae).item()
    ax = axes[0, 0]
    t_db = 20 * np.log10(np.abs(target[best_idx, :, 1].numpy()) + 1e-12)
    p_db = 20 * np.log10(np.abs(pred[best_idx, :, 1].numpy()) + 1e-12)
    ax.plot(freqs, t_db, 'b-', linewidth=2, label='Ground Truth')
    ax.plot(freqs, p_db, 'r--', linewidth=2, label='PRNet')
    ax.set_title(f"Best Sample (MAE: {per_sample_mae[best_idx]:.2f} dB)", fontsize=10)
    ax.set_xlabel("Frequency (GHz)"); ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, alpha=0.3); ax.legend()
    
    # --- Plot 2: Median test sample (Sdd21) ---
    median_idx = torch.argsort(per_sample_mae)[len(per_sample_mae)//2].item()
    ax = axes[0, 1]
    t_db = 20 * np.log10(np.abs(target[median_idx, :, 1].numpy()) + 1e-12)
    p_db = 20 * np.log10(np.abs(pred[median_idx, :, 1].numpy()) + 1e-12)
    ax.plot(freqs, t_db, 'b-', linewidth=2, label='Ground Truth')
    ax.plot(freqs, p_db, 'r--', linewidth=2, label='PRNet')
    ax.set_title(f"Median Sample (MAE: {per_sample_mae[median_idx]:.2f} dB)", fontsize=10)
    ax.set_xlabel("Frequency (GHz)"); ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, alpha=0.3); ax.legend()
    
    # --- Plot 3: Worst test sample (Sdd21) ---
    worst_idx = torch.argmax(per_sample_mae).item()
    ax = axes[0, 2]
    t_db = 20 * np.log10(np.abs(target[worst_idx, :, 1].numpy()) + 1e-12)
    p_db = 20 * np.log10(np.abs(pred[worst_idx, :, 1].numpy()) + 1e-12)
    ax.plot(freqs, t_db, 'b-', linewidth=2, label='Ground Truth')
    ax.plot(freqs, p_db, 'r--', linewidth=2, label='PRNet')
    ax.set_title(f"Worst Sample (MAE: {per_sample_mae[worst_idx]:.2f} dB)", fontsize=10)
    ax.set_xlabel("Frequency (GHz)"); ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, alpha=0.3); ax.legend()
    
    # --- Plot 4: Training convergence ---
    ax = axes[1, 0]
    ax.plot(history['val_mae_db'], 'b-', linewidth=1.5, label='Val MAE (dB)')
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Target: 0.5 dB')
    ax.set_title("Training Convergence", fontsize=10)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val MAE (dB)")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3); ax.legend()
    
    # --- Plot 5: Per-sample MAE distribution ---
    ax = axes[1, 1]
    ax.hist(per_sample_mae.numpy(), bins=30, color='steelblue', edgecolor='black', alpha=0.8)
    ax.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Target: 0.5 dB')
    ax.axvline(x=per_sample_mae.mean().item(), color='red', linestyle='-', linewidth=2, label=f'Mean: {per_sample_mae.mean():.2f} dB')
    ax.set_title("Per-Sample MAE Distribution", fontsize=10)
    ax.set_xlabel("MAE (dB)"); ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3); ax.legend()
    
    # --- Plot 6: Best sample Sdd11 ---
    ax = axes[1, 2]
    t_db = 20 * np.log10(np.abs(target[best_idx, :, 0].numpy()) + 1e-12)
    p_db = 20 * np.log10(np.abs(pred[best_idx, :, 0].numpy()) + 1e-12)
    ax.plot(freqs, t_db, 'b-', linewidth=2, label='Ground Truth')
    ax.plot(freqs, p_db, 'r--', linewidth=2, label='PRNet')
    ax.set_title(f"Best Sample Sdd11 (MAE: {per_sample_mae[best_idx]:.2f} dB)", fontsize=10)
    ax.set_xlabel("Frequency (GHz)"); ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, alpha=0.3); ax.legend()
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "benchmark_results.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Results saved to: {save_path}")


# =====================================================================
# 7. MAIN
# =====================================================================
if __name__ == "__main__":
    data_path = os.path.expanduser(
        "~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects"
        "/data/processed/Synthetic-Link/synthetic_poles_dataset.pt"
    )
    
    model, history, test_mae = train(
        data_path=data_path,
        results_dir="realistic_benchmark_results",
        epochs=600
    )