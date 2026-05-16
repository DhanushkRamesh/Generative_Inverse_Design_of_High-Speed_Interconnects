"""
==========================================================================
V3: MATCHED SYNTHETIC GENERATOR + TRAINING
==========================================================================
THE REAL PROBLEM (why V1 and V2 both plateau at ~3.9 dB):

There are THREE representation mismatches between the oracle (data generator)
and the PRNet (learner). The PRNet literally CANNOT represent the data:

MISMATCH 1 — LOSS ENVELOPE MODEL
  Oracle uses:     exp(-(alpha*sqrt(f) + beta*f^2))
  PRNet V2 uses:   exp(-gamma * f)
  These are DIFFERENT functions. The oracle's sqrt(f) model creates a
  characteristic concave rolloff that a single linear-exponential cannot match.
  The PRNet would need two gamma parameters (one for sqrt, one for f^2).

MISMATCH 2 — VARIABLE POLE COUNT
  Oracle generates: 4-8 poles per sample (with zero residues for unused poles)
  PRNet always uses: 8 poles
  But the PRNet has no incentive to zero-out unused poles. The extra poles
  add noise/artifacts at frequencies where the ground truth is smooth.

MISMATCH 3 — PASSIVITY SCALING
  Oracle applies:   per-sample global scaling to enforce |S| < 0.9
  PRNet has:        no such global scaling
  This means the PRNet must learn a different effective residue magnitude
  for every sample, making the mapping harder than necessary.

THE FIX:
  Make the oracle generate data using EXACTLY the same functional form
  the PRNet can produce. Same number of poles (always 8), same loss envelope
  (exp(-gamma*f)), no post-hoc passivity scaling.

  This is the fair benchmark: if the PRNet can't learn THIS, it can't
  learn anything. If it CAN learn this, we know the architecture works
  and the remaining gap to TUHH is about capacity/complexity.
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
# 1. MATCHED ORACLE — Same functional form as PRNet
# =====================================================================

class MatchedOracle(nn.Module):
    """
    Generates ground truth data using EXACTLY the same pole-residue formula
    as the PRNet forward pass. No functional mismatch.
    
    S(s) = sum_{n=1}^{N} [ c_n/(s-p_n) + conj(c_n)/(s-conj(p_n)) ] + d
    H_s21 = H_s21_raw * exp(-gamma * f)
    
    All 8 poles are always active (but some may have near-zero residues,
    which the PRNet must learn to produce).
    """
    def __init__(self, input_dim=10, num_poles=8, seed=42):
        super().__init__()
        self.num_poles = num_poles
        
        # Output per target: alpha(8) + f_res(8) + c_re(8) + c_im(8) + d_re(1) + d_im(1) = 34
        # Total: 34*2 targets + 1 gamma = 69
        self.out_dim = (4 * num_poles + 2) * 2 + 1
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.Tanh(),
            nn.Linear(128, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, self.out_dim),
        )
        
        torch.manual_seed(seed)
        for p in self.net.parameters():
            nn.init.xavier_normal_(p) if p.dim() > 1 else nn.init.zeros_(p)
        for p in self.net.parameters():
            p.requires_grad = False
    
    def synthesize(self, X, f_ghz):
        """
        Generate S-parameters using the EXACT same math as the PRNet.
        Returns: (N, num_freqs, 2) complex tensor
        """
        with torch.no_grad():
            raw = self.net(X)
        
        batch = X.shape[0]
        num_freqs = len(f_ghz)
        params_per_target = 4 * self.num_poles + 2
        
        # Parse outputs identically to PRNet
        pr_raw = raw[:, :-1].view(batch, 2, params_per_target)
        gamma_raw = raw[:, -1]
        
        # Same bounded activations as PRNet
        alpha = -(torch.sigmoid(pr_raw[:, :, :self.num_poles]) * 2.85 + 0.15)
        f_res = torch.sigmoid(pr_raw[:, :, self.num_poles:2*self.num_poles]) * 100.0
        beta = 2 * math.pi * f_res
        c_re = (torch.sigmoid(pr_raw[:, :, 2*self.num_poles:3*self.num_poles]) - 0.5) * 100.0
        c_im = (torch.sigmoid(pr_raw[:, :, 3*self.num_poles:4*self.num_poles]) - 0.5) * 100.0
        d_re = pr_raw[:, :, -2].unsqueeze(-1)
        d_im = pr_raw[:, :, -1].unsqueeze(-1)
        
        # Build complex pole-residue
        s = torch.complex(
            torch.zeros(num_freqs), 
            torch.tensor(2 * math.pi * f_ghz, dtype=torch.float32)
        )  # (F,)
        
        p = torch.complex(alpha, beta)          # (B, 2, P)
        c = torch.complex(c_re, c_im)            # (B, 2, P)
        d = torch.complex(d_re, d_im)            # (B, 2, 1)
        
        # Rational function evaluation
        s_view = s.view(1, 1, 1, num_freqs)       # (1, 1, 1, F)
        p_view = p.unsqueeze(-1)                   # (B, 2, P, 1)
        c_view = c.unsqueeze(-1)                   # (B, 2, P, 1)
        
        term1 = c_view / (s_view - p_view)
        term2 = torch.conj(c_view) / (s_view - torch.conj(p_view))
        H_s = torch.sum(term1 + term2, dim=2) + d  # (B, 2, F)
        
        # Loss envelope on S21 — SAME as PRNet: exp(-gamma * f)
        gamma = torch.nn.functional.softplus(gamma_raw).unsqueeze(-1)  # (B, 1)
        f_tensor = torch.tensor(f_ghz, dtype=torch.float32).view(1, num_freqs)
        exp_decay = torch.exp(-gamma * f_tensor).to(torch.complex64)
        
        H_s11 = H_s[:, 0, :]              # (B, F)
        H_s21 = H_s[:, 1, :] * exp_decay  # (B, F)
        
        # Stack: (B, F, 2)
        result = torch.stack([H_s11, H_s21], dim=-1)
        
        return result


def generate_matched_dataset(num_samples=2000, num_freqs=401, save_dir=None):
    """Generate dataset where oracle and PRNet have identical functional form."""
    
    print("=" * 70)
    print("V3: GENERATING MATCHED SYNTHETIC DATASET")
    print("  Oracle uses EXACT same math as PRNet — no representational gap")
    print("=" * 70)
    
    torch.manual_seed(0)
    X = torch.rand(num_samples, 10)
    f_ghz = np.linspace(0.25, 100.0, num_freqs).astype(np.float32)
    
    oracle = MatchedOracle(input_dim=10, num_poles=8, seed=42)
    S_complex = oracle.synthesize(X, f_ghz)
    
    # Statistics
    s_mag = torch.abs(S_complex)
    s_db = 20 * torch.log10(s_mag.clamp(min=1e-9))
    
    print(f"\nGenerated {num_samples} samples, {num_freqs} freq points")
    print(f"Sdd11 dB range: [{s_db[:,:,0].min():.1f}, {s_db[:,:,0].max():.1f}]")
    print(f"Sdd21 dB range: [{s_db[:,:,1].min():.1f}, {s_db[:,:,1].max():.1f}]")
    print(f"Sdd11 mean: {s_db[:,:,0].mean():.1f} dB")
    print(f"Sdd21 mean: {s_db[:,:,1].mean():.1f} dB")
    
    # Check for inf/nan
    n_inf = torch.isinf(S_complex).sum().item()
    n_nan = torch.isnan(S_complex).sum().item()
    print(f"Inf values: {n_inf}, NaN values: {n_nan}")
    
    if n_inf > 0 or n_nan > 0:
        print("WARNING: Inf/NaN in generated data. Cleaning...")
        S_complex = torch.nan_to_num(S_complex, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Package for pipeline
    Y_real = S_complex.real.unsqueeze(2)
    Y_real = torch.cat([Y_real, Y_real], dim=2)
    Y_imag = S_complex.imag.unsqueeze(2)
    Y_imag = torch.cat([Y_imag, Y_imag], dim=2)
    
    dataset = {
        'X_global': X[:, :5],
        'X_local': X[:, 5:],
        'Y_real': Y_real,
        'Y_imag': Y_imag,
        'frequencies': torch.tensor(f_ghz * 1e9, dtype=torch.float32),
    }
    
    if save_dir is None:
        save_dir = os.path.expanduser(
            "~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects"
            "/data/processed/Synthetic-Link"
        )
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "synthetic_poles_dataset.pt")
    torch.save(dataset, save_path)
    print(f"\nSaved to: {save_path}")
    
    # Diagnostic plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for col in range(3):
        idx = col * (num_samples // 3)
        ax = axes[col]
        for t, (label, color) in enumerate([('Sdd11', 'blue'), ('Sdd21', 'red')]):
            y_db = 20 * np.log10(np.abs(S_complex[idx, :, t].numpy()) + 1e-12)
            ax.plot(f_ghz, y_db, color=color, label=label, linewidth=1.5)
        ax.set_title(f"Sample {idx}", fontsize=10)
        ax.set_xlabel("Freq (GHz)"); ax.set_ylabel("Mag (dB)")
        ax.set_ylim(-60, 10); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    plt.suptitle("V3 Matched Dataset Samples", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "v3_diagnostic.png"), dpi=150)
    plt.close()
    
    return save_path


# =====================================================================
# 2. PRNet V3 — Identical to oracle functional form
# =====================================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim),
            nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim),
        )
    def forward(self, x):
        return x + self.block(x)


class PRNetV3(nn.Module):
    def __init__(self, input_dim=10, num_poles=8, num_targets=2, hidden_dim=512):
        super().__init__()
        self.num_targets = num_targets
        self.num_poles = num_poles
        self.params_per_target = 4 * num_poles + 2
        total_out = self.params_per_target * num_targets + 1
        
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.body = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, total_out),
        )
        
        # Spreading init
        target_f = torch.linspace(0.05, 0.95, num_poles)
        inv_sig = torch.log(target_f / (1.0 - target_f))
        with torch.no_grad():
            fl = self.head[-1]
            for t in range(num_targets):
                off = t * self.params_per_target
                fl.bias[off + num_poles : off + 2*num_poles] = inv_sig
                fl.bias[off + 2*num_poles : off + 4*num_poles] = 0.5
            fl.bias[-1] = 0.0
    
    def forward(self, x, s_tensor, freqs_ghz):
        batch = x.shape[0]
        num_freqs = s_tensor.shape[0]
        
        h = self.stem(x)
        h = self.body(h)
        raw_out = self.head(h)
        
        pr_params = raw_out[:, :-1].view(batch, self.num_targets, self.params_per_target)
        gamma_raw = raw_out[:, -1]
        
        # EXACTLY same activations as oracle
        alpha = -(torch.sigmoid(pr_params[:, :, :self.num_poles]) * 2.85 + 0.15)
        f_res = torch.sigmoid(pr_params[:, :, self.num_poles:2*self.num_poles]) * 100.0
        beta = 2 * math.pi * f_res
        c_re = (torch.sigmoid(pr_params[:, :, 2*self.num_poles:3*self.num_poles]) - 0.5) * 100.0
        c_im = (torch.sigmoid(pr_params[:, :, 3*self.num_poles:4*self.num_poles]) - 0.5) * 100.0
        d_re = pr_params[:, :, -2].unsqueeze(-1)
        d_im = pr_params[:, :, -1].unsqueeze(-1)
        
        p = torch.complex(alpha, beta).unsqueeze(-1)
        c = torch.complex(c_re, c_im).unsqueeze(-1)
        d = torch.complex(d_re, d_im)
        s_view = s_tensor.view(1, 1, 1, num_freqs)
        
        # Safe division
        denom1 = s_view - p
        denom2 = s_view - torch.conj(p)
        eps = 1e-4
        safe_d1 = torch.complex(denom1.real + eps * (denom1.real.abs() < eps).float(), denom1.imag)
        safe_d2 = torch.complex(denom2.real + eps * (denom2.real.abs() < eps).float(), denom2.imag)
        
        term1 = c / safe_d1
        term2 = torch.conj(c) / safe_d2
        H_s = torch.sum(term1 + term2, dim=2) + d
        
        H_s = torch.complex(H_s.real.clamp(-200, 200), H_s.imag.clamp(-200, 200))
        
        # EXACTLY same loss envelope as oracle
        gamma = torch.nn.functional.softplus(gamma_raw).unsqueeze(-1)
        f_tensor = freqs_ghz.view(1, num_freqs)
        exp_decay = torch.exp(-gamma * f_tensor).to(torch.complex64)
        
        H_s11 = H_s[:, 0, :]
        H_s21 = H_s[:, 1, :] * exp_decay
        H_out = torch.stack([H_s11, H_s21], dim=1)
        
        return H_out.transpose(1, 2)


# =====================================================================
# 3. DATASET + TRAINING
# =====================================================================

class SyntheticPoleDataset(Dataset):
    def __init__(self, pt_file_path):
        data = torch.load(pt_file_path)
        self.X = torch.cat([data['X_global'], data['X_local']], dim=1)
        y_r = data['Y_real'][:, :, :, 0] if data['Y_real'].dim() == 4 else data['Y_real']
        y_i = data['Y_imag'][:, :, :, 0] if data['Y_imag'].dim() == 4 else data['Y_imag']
        self.Y = torch.complex(y_r, y_i)
        self.freqs_ghz = data['frequencies'].numpy() / 1e9
        y_db = 20 * torch.log10(torch.abs(self.Y).clamp(min=1e-9))
        print(f"Loaded {len(self.X)} samples | {len(self.freqs_ghz)} freqs | "
              f"dB range: [{y_db.min():.1f}, {y_db.max():.1f}]")
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]


def train(data_path, results_dir="benchmark_v3_results", epochs=800):
    os.makedirs(results_dir, exist_ok=True)
    ds = SyntheticPoleDataset(data_path)
    
    n = len(ds)
    n_train, n_val = int(0.8*n), int(0.1*n)
    n_test = n - n_train - n_val
    train_set, val_set, test_set = random_split(ds, [n_train, n_val, n_test])
    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=64)
    test_loader = DataLoader(test_set, batch_size=64)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PRNetV3(input_dim=10, num_poles=8, num_targets=2).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"PRNet V3: {total_params:,} params on {device}")
    print(f"Split: {n_train}/{n_val}/{n_test}")
    
    f_ghz = torch.tensor(ds.freqs_ghz, dtype=torch.float32, device=device)
    omega = 2 * math.pi * f_ghz
    s_tensor = torch.complex(torch.zeros_like(omega), omega)
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3, epochs=epochs, steps_per_epoch=len(train_loader),
        pct_start=0.05, anneal_strategy='cos', div_factor=10, final_div_factor=100
    )
    
    # Pure dB loss for first 300 epochs, then add MSE
    phase1_epochs = 300
    
    history = {'train': [], 'val_db': []}
    best_val = float('inf')
    
    print(f"\n{'Ep':>4} | {'Phase':>6} | {'Loss':>9} | {'Val':>7} | {'Best':>7}")
    print("-" * 50)
    
    for epoch in range(epochs):
        model.train()
        t_loss, n_b = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x, s_tensor, f_ghz)
            
            # dB loss (always)
            pm = torch.clamp(torch.abs(pred), min=1e-7)
            tm = torch.clamp(torch.abs(y), min=1e-7)
            pdb = torch.clamp(20*torch.log10(pm), -100, 40)
            tdb = torch.clamp(20*torch.log10(tm), -100, 40)
            loss = nn.functional.smooth_l1_loss(pdb, tdb)
            
            # Phase 2: add small complex MSE
            if epoch >= phase1_epochs:
                loss = loss + 0.05 * (nn.functional.mse_loss(pred.real, y.real) + 
                                      nn.functional.mse_loss(pred.imag, y.imag))
            
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(); continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            t_loss += loss.item(); n_b += 1
        
        # Validate
        model.eval()
        v_err, n_v = 0, 0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                pred = model(xv, s_tensor, f_ghz)
                pm = torch.clamp(torch.abs(pred), min=1e-7)
                tm = torch.clamp(torch.abs(yv), min=1e-7)
                pdb = torch.clamp(20*torch.log10(pm), -100, 40)
                tdb = torch.clamp(20*torch.log10(tm), -100, 40)
                v_err += torch.abs(pdb - tdb).mean().item()
                n_v += 1
        
        avg_t = t_loss / max(n_b, 1)
        avg_v = v_err / max(n_v, 1)
        history['train'].append(avg_t)
        history['val_db'].append(avg_v)
        
        if avg_v < best_val:
            best_val = avg_v
            torch.save(model.state_dict(), os.path.join(results_dir, "best.pth"))
        
        if (epoch+1) % 50 == 0 or epoch == 0 or epoch == phase1_epochs:
            ph = "dB" if epoch < phase1_epochs else "dB+MSE"
            mk = " *" if avg_v <= best_val else ""
            print(f"{epoch+1:4d} | {ph:>6} | {avg_t:9.4f} | {avg_v:7.2f} | {best_val:7.2f}{mk}")
    
    print(f"\nBest Val MAE: {best_val:.2f} dB")
    
    # =====================================================================
    # TEST EVALUATION
    # =====================================================================
    model.load_state_dict(torch.load(os.path.join(results_dir, "best.pth")))
    model.eval()
    
    preds, targets = [], []
    with torch.no_grad():
        for xv, yv in test_loader:
            xv, yv = xv.to(device), yv.to(device)
            preds.append(model(xv, s_tensor, f_ghz).cpu())
            targets.append(yv.cpu())
    
    all_p = torch.cat(preds)
    all_t = torch.cat(targets)
    
    pdb = torch.clamp(20*torch.log10(torch.clamp(torch.abs(all_p), min=1e-7)), -100, 40)
    tdb = torch.clamp(20*torch.log10(torch.clamp(torch.abs(all_t), min=1e-7)), -100, 40)
    per_sample = torch.abs(pdb - tdb).mean(dim=(1,2))
    
    s11_mae = torch.abs(pdb[:,:,0] - tdb[:,:,0]).mean().item()
    s21_mae = torch.abs(pdb[:,:,1] - tdb[:,:,1]).mean().item()
    test_mae = per_sample.mean().item()
    
    print(f"\n{'='*50}")
    print(f"TEST ({n_test} samples): MAE={test_mae:.2f} dB")
    print(f"  Sdd11: {s11_mae:.2f} dB | Sdd21: {s21_mae:.2f} dB")
    print(f"  Median: {torch.quantile(per_sample, 0.5):.2f} dB")
    print(f"  95th:   {torch.quantile(per_sample, 0.95):.2f} dB")
    result = "PASS" if test_mae < 1.0 else "NEEDS WORK"
    print(f"  {result}: {'<1 dB target' if test_mae < 1.0 else f'got {test_mae:.2f} dB'}")
    print(f"{'='*50}")
    
    # =====================================================================
    # PLOTS
    # =====================================================================
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(f"V3 Matched Benchmark — Test MAE: {test_mae:.2f} dB", fontsize=14, fontweight='bold')
    
    freqs = ds.freqs_ghz
    for col, (label, fn) in enumerate([
        ("Best", lambda m: torch.argmin(m).item()),
        ("Median", lambda m: torch.argsort(m)[len(m)//2].item()),
        ("Worst", lambda m: torch.argmax(m).item()),
    ]):
        idx = fn(per_sample)
        ax = axes[0, col]
        td = 20*np.log10(np.abs(all_t[idx,:,1].numpy())+1e-12)
        pd = 20*np.log10(np.abs(all_p[idx,:,1].numpy())+1e-12)
        ax.plot(freqs, td, 'b-', lw=2, label='Truth')
        ax.plot(freqs, pd, 'r--', lw=2, label='PRNet V3')
        ax.set_title(f"{label} Sdd21 ({per_sample[idx]:.2f} dB)")
        ax.set_xlabel("GHz"); ax.set_ylabel("dB"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    
    # Best Sdd11
    bi = torch.argmin(per_sample).item()
    ax = axes[0, 3]
    td = 20*np.log10(np.abs(all_t[bi,:,0].numpy())+1e-12)
    pd = 20*np.log10(np.abs(all_p[bi,:,0].numpy())+1e-12)
    ax.plot(freqs, td, 'b-', lw=2, label='Truth')
    ax.plot(freqs, pd, 'r--', lw=2, label='PRNet V3')
    ax.set_title(f"Best Sdd11 ({per_sample[bi]:.2f} dB)")
    ax.set_xlabel("GHz"); ax.set_ylabel("dB"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    
    # Convergence
    ax = axes[1, 0]
    ax.plot(history['val_db'], 'b-', lw=1)
    ax.axhline(y=1.0, color='green', ls='--', alpha=0.7, label='1 dB target')
    ax.axvline(x=300, color='orange', ls=':', alpha=0.7, label='Phase 2')
    ax.set_title("Convergence"); ax.set_xlabel("Epoch"); ax.set_ylabel("Val MAE (dB)")
    ax.set_yscale('log'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    
    # Histogram
    ax = axes[1, 1]
    ax.hist(per_sample.numpy(), bins=30, color='steelblue', edgecolor='black', alpha=0.8)
    ax.axvline(x=1.0, color='green', ls='--', lw=2, label='1 dB')
    ax.axvline(x=test_mae, color='red', ls='-', lw=2, label=f'Mean: {test_mae:.2f}')
    ax.set_title("MAE Distribution"); ax.set_xlabel("MAE (dB)"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Error vs freq
    ax = axes[1, 2]
    freq_err = np.abs(pdb.numpy() - tdb.numpy()).mean(axis=0)
    ax.plot(freqs, freq_err[:,0], 'b-', lw=1.5, label='Sdd11')
    ax.plot(freqs, freq_err[:,1], 'r-', lw=1.5, label='Sdd21')
    ax.set_title("MAE vs Freq"); ax.set_xlabel("GHz"); ax.set_ylabel("|Error| dB")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    
    # LR schedule
    ax = axes[1, 3]
    ax.text(0.5, 0.5, f"Test MAE: {test_mae:.2f} dB\n"
            f"Sdd11: {s11_mae:.2f} dB\n"
            f"Sdd21: {s21_mae:.2f} dB\n"
            f"Median: {torch.quantile(per_sample,0.5):.2f} dB\n"
            f"95th: {torch.quantile(per_sample,0.95):.2f} dB\n"
            f"Best Val: {best_val:.2f} dB\n\n"
            f"{'PASS' if test_mae < 1.0 else 'NEEDS WORK'}",
            transform=ax.transAxes, fontsize=14, va='center', ha='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen' if test_mae < 1.0 else 'lightyellow'))
    ax.set_title("Summary"); ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "v3_results.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Plots saved to {results_dir}/v3_results.png")
    
    return model, history, test_mae


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    # Step 1: Generate matched data
    data_path = generate_matched_dataset(num_samples=2000, num_freqs=401)
    
    # Step 2: Train
    model, history, test_mae = train(data_path=data_path, epochs=800)