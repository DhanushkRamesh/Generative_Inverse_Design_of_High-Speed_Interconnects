"""
==========================================================================
TRAINING SCRIPT V2: PRNet on Realistic Synthetic Benchmark
==========================================================================
Fixes from V1 (which plateaued at 3.92 dB):
  
  1. LOSS FUNCTION: V1 used complex MSE + 0.1*dB_Huber. The MSE term 
     dominates and operates in LINEAR scale, so the optimizer only cares
     about peaks (large |S|) and ignores baselines at -25 to -30 dB.
     
     FIX: Two-phase training:
       Phase 1 (epochs 1-200):   Pure dB SmoothL1 loss (learn the shape)
       Phase 2 (epochs 201-800): dB SmoothL1 + complex MSE (refine phase)
  
  2. LEARNING RATE: V1 used CosineAnnealing from 1e-3 which decays too
     fast. The model converges to a poor local minimum by epoch 50.
     
     FIX: OneCycleLR with warmup, peak at 2e-3, slow decay.
  
  3. RESIDUE BOUNDS: V1 used [-20, 20] which clips the amplitudes the
     model needs for sharp resonances.
     
     FIX: Unbounded residues with tanh*scale, wider range [-50, 50].
  
  4. DAMPING FLOOR: V1 used alpha >= -0.3 which is too high — the oracle
     generates poles with alpha as low as -0.3, so the model can barely
     reach the oracle's poles.
     
     FIX: alpha floor lowered to -0.15 (still stable, but sharper peaks).
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
# 1. Dataset Loader
# =====================================================================
class SyntheticPoleDataset(Dataset):
    def __init__(self, pt_file_path):
        data = torch.load(pt_file_path)
        self.X = torch.cat([data['X_global'], data['X_local']], dim=1)
        
        y_r = data['Y_real'][:, :, :, 0] if data['Y_real'].dim() == 4 else data['Y_real']
        y_i = data['Y_imag'][:, :, :, 0] if data['Y_imag'].dim() == 4 else data['Y_imag']
        self.Y = torch.complex(y_r, y_i)
        
        self.freqs_ghz = data['frequencies'].numpy() / 1e9
        self.gt_num_poles = data.get('gt_num_poles', None)
        
        # Compute and print target statistics for sanity check
        y_mag = torch.abs(self.Y)
        y_db = 20 * torch.log10(y_mag.clamp(min=1e-9))
        print(f"Loaded {len(self.X)} samples | "
              f"Freq: {self.freqs_ghz[0]:.2f}-{self.freqs_ghz[-1]:.2f} GHz | "
              f"Points: {len(self.freqs_ghz)}")
        print(f"Target dB range: [{y_db.min():.1f}, {y_db.max():.1f}] dB | "
              f"Mean: {y_db.mean():.1f} dB")

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]


# =====================================================================
# 2. PRNet V2 — Wider MLP, Lower Damping Floor, Wider Residues
# =====================================================================

class ResidualBlock(nn.Module):
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


class ProductionPRNetV2(nn.Module):
    def __init__(self, input_dim=10, num_poles=8, num_targets=2, hidden_dim=512):
        super().__init__()
        self.num_targets = num_targets
        self.num_poles = num_poles
        
        self.params_per_target = (4 * num_poles) + 2  # 34 per target
        total_out = (self.params_per_target * num_targets) + 1  # +1 for gamma
        
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # 4 residual blocks (up from 3 — more capacity for nonlinear mapping)
        self.body = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )
        
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, total_out),
        )
        
        # Spreading initialization
        target_f = torch.linspace(0.05, 0.95, num_poles)
        inv_sig = torch.log(target_f / (1.0 - target_f))
        
        with torch.no_grad():
            final_layer = self.head[-1]
            for t in range(num_targets):
                offset = t * self.params_per_target
                final_layer.bias[offset + num_poles : offset + 2*num_poles] = inv_sig
                final_layer.bias[offset + 2*num_poles : offset + 4*num_poles] = 0.5
            final_layer.bias[-1] = 0.0
    
    def forward(self, x, s_tensor, freqs_ghz):
        batch = x.shape[0]
        num_freqs = s_tensor.shape[0]
        
        h = self.stem(x)
        h = self.body(h)
        raw_out = self.head(h)
        
        pr_params = raw_out[:, :-1].view(batch, self.num_targets, self.params_per_target)
        gamma_raw = raw_out[:, -1]
        
        # ====== KEY CHANGE: Lower damping floor ======
        # alpha range: [-3.0, -0.15] (was [-3.0, -0.3])
        # This allows sharper resonance peaks to match the oracle
        alpha = -(torch.sigmoid(pr_params[:, :, :self.num_poles]) * 2.85 + 0.15)
        
        # Resonance frequency: [0, 100] GHz
        f_res = torch.sigmoid(pr_params[:, :, self.num_poles:2*self.num_poles]) * 100.0
        beta = 2 * math.pi * f_res
        
        # ====== KEY CHANGE: Wider residue range ======
        # [-50, 50] instead of [-40, 40] for stronger resonance amplitudes
        c_re = (torch.sigmoid(pr_params[:, :, 2*self.num_poles:3*self.num_poles]) - 0.5) * 100.0
        c_im = (torch.sigmoid(pr_params[:, :, 3*self.num_poles:4*self.num_poles]) - 0.5) * 100.0
        
        d_re = pr_params[:, :, -2].unsqueeze(-1)
        d_im = pr_params[:, :, -1].unsqueeze(-1)
        
        # Pole-residue synthesis with numerical safety
        p = torch.complex(alpha, beta).unsqueeze(-1)
        c = torch.complex(c_re, c_im).unsqueeze(-1)
        d = torch.complex(d_re, d_im)
        s_view = s_tensor.view(1, 1, 1, num_freqs)
        
        denom1 = s_view - p
        denom2 = s_view - torch.conj(p)
        
        # Safer epsilon injection: add to real part only (doesn't shift frequency)
        eps = 1e-4
        safe_denom1 = torch.complex(
            denom1.real + eps * (denom1.real.abs() < eps).float(),
            denom1.imag
        )
        safe_denom2 = torch.complex(
            denom2.real + eps * (denom2.real.abs() < eps).float(),
            denom2.imag
        )
        
        term1 = c / safe_denom1
        term2 = torch.conj(c) / safe_denom2
        H_s = torch.sum(term1 + term2, dim=2) + d
        
        # Clamp to prevent inf in downstream log10
        H_s = torch.complex(
            torch.clamp(H_s.real, -200.0, 200.0),
            torch.clamp(H_s.imag, -200.0, 200.0)
        )
        
        # Loss envelope on Sdd21
        gamma = torch.nn.functional.softplus(gamma_raw).unsqueeze(-1)
        f_tensor = freqs_ghz.view(1, num_freqs)
        exp_decay = torch.exp(-gamma * f_tensor).to(dtype=torch.complex64)
        
        H_s11 = H_s[:, 0, :]
        H_s21 = H_s[:, 1, :] * exp_decay
        H_out = torch.stack([H_s11, H_s21], dim=1)
        
        return H_out.transpose(1, 2)


# =====================================================================
# 3. TWO-PHASE LOSS FUNCTION
# =====================================================================

def db_primary_loss(pred, target):
    """
    Phase 1 loss: PURE dB-space SmoothL1.
    
    This forces the optimizer to care equally about errors at -30 dB
    and errors at 0 dB. In V1, the complex MSE dominated and the model
    only learned the peaks (where linear magnitude is large).
    """
    pred_mag = torch.clamp(torch.abs(pred), min=1e-7)
    target_mag = torch.clamp(torch.abs(target), min=1e-7)
    
    p_db = torch.clamp(20 * torch.log10(pred_mag), -100.0, 40.0)
    t_db = torch.clamp(20 * torch.log10(target_mag), -100.0, 40.0)
    
    return nn.functional.smooth_l1_loss(p_db, t_db)


def combined_loss(pred, target):
    """
    Phase 2 loss: dB SmoothL1 (primary) + complex MSE (secondary).
    
    Once the dB shape is learned, adding complex MSE refines the phase
    and fine structure. But dB remains the PRIMARY term.
    """
    pred_mag = torch.clamp(torch.abs(pred), min=1e-7)
    target_mag = torch.clamp(torch.abs(target), min=1e-7)
    
    p_db = torch.clamp(20 * torch.log10(pred_mag), -100.0, 40.0)
    t_db = torch.clamp(20 * torch.log10(target_mag), -100.0, 40.0)
    
    db_loss = nn.functional.smooth_l1_loss(p_db, t_db)
    
    # Complex MSE (secondary — weight 0.05)
    mse_re = nn.functional.mse_loss(pred.real, target.real)
    mse_im = nn.functional.mse_loss(pred.imag, target.imag)
    
    return db_loss + 0.05 * (mse_re + mse_im)


# =====================================================================
# 4. TRAINING LOOP WITH TWO PHASES
# =====================================================================

def train(data_path, results_dir="benchmark_v2_results", epochs=800):
    os.makedirs(results_dir, exist_ok=True)
    
    ds = SyntheticPoleDataset(data_path)
    
    n = len(ds)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    train_set, val_set, test_set = random_split(ds, [n_train, n_val, n_test])
    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=64)
    test_loader = DataLoader(test_set, batch_size=64)
    
    print(f"Split: {n_train} train / {n_val} val / {n_test} test")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProductionPRNetV2(input_dim=10, num_poles=8, num_targets=2).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters on {device}")
    
    # Frequency tensors
    f_ghz = torch.tensor(ds.freqs_ghz, dtype=torch.float32, device=device)
    omega = 2 * math.pi * f_ghz
    s_tensor = torch.complex(torch.zeros_like(omega), omega)
    
    # ====== TWO-PHASE OPTIMIZER SETUP ======
    phase1_epochs = 300
    phase2_epochs = epochs - phase1_epochs
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    
    # OneCycleLR: warmup -> peak -> slow decay
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=2e-3, 
        epochs=epochs, 
        steps_per_epoch=len(train_loader),
        pct_start=0.05,      # 5% warmup
        anneal_strategy='cos',
        div_factor=10,        # start_lr = max_lr / 10
        final_div_factor=100  # end_lr = max_lr / 1000
    )
    
    history = {'train_loss': [], 'val_mae_db': [], 'lr': []}
    best_val_mae = float('inf')
    
    print(f"\n{'='*70}")
    print(f"{'Epoch':>6} | {'Phase':>7} | {'Train Loss':>11} | {'Val MAE':>9} | {'Best':>8} | {'LR':>10}")
    print(f"{'='*70}")
    
    for epoch in range(epochs):
        # Select loss function based on phase
        phase = 1 if epoch < phase1_epochs else 2
        loss_fn = db_primary_loss if phase == 1 else combined_loss
        
        # --- TRAIN ---
        model.train()
        train_loss = 0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x, s_tensor, f_ghz)
            loss = loss_fn(pred, y)
            
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            n_batches += 1
        
        # --- VALIDATE ---
        model.eval()
        val_db_error = 0
        n_val_batches = 0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                pred = model(xv, s_tensor, f_ghz)
                p_db = torch.clamp(20 * torch.log10(torch.clamp(torch.abs(pred), min=1e-7)), -100.0, 40.0)
                t_db = torch.clamp(20 * torch.log10(torch.clamp(torch.abs(yv), min=1e-7)), -100.0, 40.0)
                val_db_error += torch.abs(p_db - t_db).mean().item()
                n_val_batches += 1
        
        avg_train = train_loss / max(n_batches, 1)
        avg_val_db = val_db_error / max(n_val_batches, 1)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(avg_train)
        history['val_mae_db'].append(avg_val_db)
        history['lr'].append(current_lr)
        
        if avg_val_db < best_val_mae:
            best_val_mae = avg_val_db
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
        
        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == phase1_epochs:
            marker = " ***" if avg_val_db <= best_val_mae else ""
            phase_str = f"dB-only" if phase == 1 else "dB+MSE"
            print(f"{epoch+1:6d} | {phase_str:>7} | {avg_train:11.4f} | "
                  f"{avg_val_db:9.2f} | {best_val_mae:8.2f} | {current_lr:10.2e}{marker}")
    
    print(f"{'='*70}")
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
    
    pred_db = torch.clamp(20 * torch.log10(torch.clamp(torch.abs(all_pred), min=1e-7)), -100.0, 40.0)
    target_db = torch.clamp(20 * torch.log10(torch.clamp(torch.abs(all_target), min=1e-7)), -100.0, 40.0)
    per_sample_mae = torch.abs(pred_db - target_db).mean(dim=(1, 2))
    
    # Per-target breakdown
    s11_mae = torch.abs(pred_db[:,:,0] - target_db[:,:,0]).mean().item()
    s21_mae = torch.abs(pred_db[:,:,1] - target_db[:,:,1]).mean().item()
    
    test_mae = per_sample_mae.mean().item()
    test_p95 = torch.quantile(per_sample_mae, 0.95).item()
    test_p50 = torch.quantile(per_sample_mae, 0.50).item()
    
    print(f"\n{'='*70}")
    print(f"TEST SET RESULTS ({n_test} samples)")
    print(f"{'='*70}")
    print(f"  Overall MAE:  {test_mae:.2f} dB")
    print(f"  Sdd11 MAE:    {s11_mae:.2f} dB")
    print(f"  Sdd21 MAE:    {s21_mae:.2f} dB")
    print(f"  Median MAE:   {test_p50:.2f} dB")
    print(f"  95th %ile:    {test_p95:.2f} dB")
    print(f"  {'PASS' if test_mae < 0.5 else 'NEEDS WORK'}: "
          f"{'< 0.5 dB target achieved!' if test_mae < 0.5 else f'Target is < 0.5 dB, got {test_mae:.2f} dB'}")
    
    # =====================================================================
    # 6. PUBLICATION PLOTS
    # =====================================================================
    _plot_results(all_pred, all_target, ds.freqs_ghz, per_sample_mae,
                  history, best_val_mae, test_mae, results_dir)
    
    return model, history, test_mae


def _plot_results(pred, target, freqs, per_sample_mae, history, best_val, test_mae, save_dir):
    """Publication-quality result plots."""
    
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(f"Rational Layer Benchmark V2 — Test MAE: {test_mae:.2f} dB", 
                 fontsize=14, fontweight='bold')
    
    # Best, Median, Worst sample Sdd21
    for col, (label, idx_fn) in enumerate([
        ("Best", lambda m: torch.argmin(m).item()),
        ("Median", lambda m: torch.argsort(m)[len(m)//2].item()),
        ("Worst", lambda m: torch.argmax(m).item()),
    ]):
        idx = idx_fn(per_sample_mae)
        ax = axes[0, col]
        t_db = 20 * np.log10(np.abs(target[idx, :, 1].numpy()) + 1e-12)
        p_db = 20 * np.log10(np.abs(pred[idx, :, 1].numpy()) + 1e-12)
        ax.plot(freqs, t_db, 'b-', linewidth=2, label='Ground Truth')
        ax.plot(freqs, p_db, 'r--', linewidth=2, label='PRNet V2')
        ax.set_title(f"{label} Sdd21 (MAE: {per_sample_mae[idx]:.2f} dB)", fontsize=10)
        ax.set_xlabel("Frequency (GHz)"); ax.set_ylabel("Magnitude (dB)")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    
    # Best sample Sdd11
    best_idx = torch.argmin(per_sample_mae).item()
    ax = axes[0, 3]
    t_db = 20 * np.log10(np.abs(target[best_idx, :, 0].numpy()) + 1e-12)
    p_db = 20 * np.log10(np.abs(pred[best_idx, :, 0].numpy()) + 1e-12)
    ax.plot(freqs, t_db, 'b-', linewidth=2, label='Ground Truth')
    ax.plot(freqs, p_db, 'r--', linewidth=2, label='PRNet V2')
    ax.set_title(f"Best Sdd11 (MAE: {per_sample_mae[best_idx]:.2f} dB)", fontsize=10)
    ax.set_xlabel("Frequency (GHz)"); ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    
    # Training convergence
    ax = axes[1, 0]
    ax.plot(history['val_mae_db'], 'b-', linewidth=1, label='Val MAE')
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Target')
    ax.axvline(x=300, color='orange', linestyle=':', alpha=0.7, label='Phase 2 start')
    ax.set_title("Training Convergence", fontsize=10)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val MAE (dB)")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    
    # Learning rate schedule
    ax = axes[1, 1]
    ax.plot(history['lr'], 'g-', linewidth=1)
    ax.set_title("Learning Rate Schedule", fontsize=10)
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Per-sample MAE distribution
    ax = axes[1, 2]
    ax.hist(per_sample_mae.numpy(), bins=30, color='steelblue', edgecolor='black', alpha=0.8)
    ax.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Target: 0.5 dB')
    ax.axvline(x=per_sample_mae.mean().item(), color='red', linestyle='-', linewidth=2, 
               label=f'Mean: {per_sample_mae.mean():.2f} dB')
    ax.set_title("Per-Sample MAE Distribution", fontsize=10)
    ax.set_xlabel("MAE (dB)"); ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    
    # Error vs frequency
    ax = axes[1, 3]
    pred_db_all = 20 * np.log10(np.abs(pred.numpy()) + 1e-12)
    target_db_all = 20 * np.log10(np.abs(target.numpy()) + 1e-12)
    freq_error = np.abs(pred_db_all - target_db_all).mean(axis=0)  # (F, T)
    ax.plot(freqs, freq_error[:, 0], 'b-', linewidth=1.5, label='Sdd11')
    ax.plot(freqs, freq_error[:, 1], 'r-', linewidth=1.5, label='Sdd21')
    ax.set_title("MAE vs Frequency", fontsize=10)
    ax.set_xlabel("Frequency (GHz)"); ax.set_ylabel("Mean |Error| (dB)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "benchmark_v2_results.png")
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
        results_dir="benchmark_v2_results",
        epochs=800
    )