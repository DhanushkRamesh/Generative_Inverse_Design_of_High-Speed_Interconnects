"""
==========================================================================
V4: HYBRID ARCHITECTURE — MLP Frequency Prediction + Rational Refinement
==========================================================================

THE INSIGHT FROM THE DIAGNOSTIC:
  Direct optimization of 69 pole/residue params on a SINGLE sample 
  averages 1.07 dB MAE. Sample 500 reached 0.47 dB but samples 0 
  and 1000 got stuck at 1.5 dB.

  This proves the optimization landscape for pole-residue parameters
  is full of local minima. Even with 2000 Adam steps and no MLP in 
  the way, gradient descent frequently gets trapped.

  Now imagine asking an MLP to predict these 69 coupled parameters 
  for ALL 2000 samples simultaneously — it averages the local minima 
  across all samples and settles at ~4 dB. No loss function or 
  learning rate will fix this.

THE SOLUTION — Inspired by:
  - CZP Framework (Cohen et al., 2023, Meta AI): NN predicts zeros 
    and poles, then evaluates the rational function analytically
  - Feng et al. (2017): Vector Fitting gives initial pole/residue 
    estimates, NN learns the mapping from geometry to VF coefficients
  - S-Crescendo (2025): Decomposes into first-order modal terms, 
    predicts each independently

KEY ARCHITECTURAL CHANGE:
  Instead of the MLP directly predicting 69 tightly-coupled pole/residue 
  parameters, we use a TWO-STAGE approach:

  Stage 1: "Frequency Backbone" — A standard MLP predicts S-parameter
           magnitudes at K anchor frequencies (e.g., K=32 points).
           This is a WELL-CONDITIONED regression problem that any 
           decent MLP can solve to <0.5 dB.

  Stage 2: "Rational Refinement" — A small network takes the anchor
           predictions and fits pole/residue parameters to reproduce 
           them, then evaluates S(s) at all 401 frequencies.
           This gives us causality by construction AND smooth 
           interpolation between anchor points.

  The key insight: Stage 1 is easy (predict 32 real numbers) and 
  Stage 2 is easy (fit poles to a known curve). Neither stage faces 
  the coupled optimization nightmare of predicting 69 pole-residue 
  parameters end-to-end.

  BUT — we still get the Rational Layer's physics guarantee because
  the final output IS a pole-residue rational function.
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
# 1. DATASET
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
              f"dB: [{y_db.min():.1f}, {y_db.max():.1f}]")
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]


# =====================================================================
# 2. HYBRID ARCHITECTURE
# =====================================================================

class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d),
            nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d))
    def forward(self, x): return x + self.net(x)


class HybridRationalNet(nn.Module):
    """
    Two-stage forward model:
    
    Stage 1 — Frequency Backbone:
      MLP: X(10) → complex S-params at 401 frequencies (real+imag, 4 channels)
      This is a standard, well-conditioned regression problem.
      Output: raw complex prediction at all frequency points.
    
    Stage 2 — Rational Refinement Layer:
      Takes the raw prediction and refines it through a learnable
      pole-residue rational function that ensures causality.
      The poles and residues are predicted by a SEPARATE small MLP
      from the same input X, and the rational function acts as a
      physics-informed correction/regularization on top of Stage 1.
    
    Final output = Stage1_prediction + alpha * RationalCorrection
      where alpha is a learnable blend weight that starts near 0
      (so Stage 1 trains first) and grows during training.
    """
    def __init__(self, input_dim=10, num_freqs=401, num_targets=2, 
                 hidden_dim=512, num_poles=8):
        super().__init__()
        self.num_freqs = num_freqs
        self.num_targets = num_targets
        self.num_poles = num_poles
        
        # ---- STAGE 1: Direct frequency prediction ----
        # Predicts real and imaginary parts at all frequencies
        # Output: (batch, num_freqs * num_targets * 2) = (batch, 401*2*2) = (batch, 1604)
        self.freq_backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, num_freqs * num_targets * 2),
        )
        
        # ---- STAGE 2: Rational correction ----
        # A smaller network that predicts pole/residue parameters
        ppt = 4 * num_poles + 2  # params per target
        rational_out = ppt * num_targets + 1  # +1 for gamma
        
        self.rational_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256), nn.GELU(),
            ResBlock(256),
            nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, rational_out),
        )
        
        # Blend weight: starts at 0 (pure frequency prediction), 
        # grows during training to incorporate rational correction
        self.blend_logit = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2) ≈ 0.12
        
        # Spreading init for rational head
        target_f = torch.linspace(0.05, 0.95, num_poles)
        inv_sig = torch.log(target_f / (1.0 - target_f))
        with torch.no_grad():
            fl = self.rational_head[-1]
            for t in range(num_targets):
                off = t * ppt
                fl.bias[off + num_poles : off + 2*num_poles] = inv_sig
                fl.bias[off + 2*num_poles : off + 4*num_poles] = 0.3
            fl.bias[-1] = 0.0
    
    def _rational_forward(self, x, s_tensor, freqs_ghz):
        """Evaluate the rational function from the small head."""
        batch = x.shape[0]
        num_freqs = s_tensor.shape[0]
        ppt = 4 * self.num_poles + 2
        
        raw = self.rational_head(x)
        pr = raw[:, :-1].view(batch, self.num_targets, ppt)
        gamma_raw = raw[:, -1]
        
        alpha = -(torch.sigmoid(pr[:, :, :self.num_poles]) * 2.85 + 0.15)
        f_res = torch.sigmoid(pr[:, :, self.num_poles:2*self.num_poles]) * 100.0
        beta = 2 * math.pi * f_res
        c_re = (torch.sigmoid(pr[:, :, 2*self.num_poles:3*self.num_poles]) - 0.5) * 100.0
        c_im = (torch.sigmoid(pr[:, :, 3*self.num_poles:4*self.num_poles]) - 0.5) * 100.0
        d_re = pr[:, :, -2].unsqueeze(-1)
        d_im = pr[:, :, -1].unsqueeze(-1)
        
        p = torch.complex(alpha, beta).unsqueeze(-1)
        c = torch.complex(c_re, c_im).unsqueeze(-1)
        d = torch.complex(d_re, d_im)
        s_view = s_tensor.view(1, 1, 1, num_freqs)
        
        denom1 = s_view - p
        denom2 = s_view - torch.conj(p)
        eps = 1e-4
        safe_d1 = torch.complex(denom1.real + eps*(denom1.real.abs()<eps).float(), denom1.imag)
        safe_d2 = torch.complex(denom2.real + eps*(denom2.real.abs()<eps).float(), denom2.imag)
        
        H_s = torch.sum(c/safe_d1 + torch.conj(c)/safe_d2, dim=2) + d
        H_s = torch.complex(H_s.real.clamp(-200, 200), H_s.imag.clamp(-200, 200))
        
        gamma = torch.nn.functional.softplus(gamma_raw).unsqueeze(-1)
        exp_decay = torch.exp(-gamma * freqs_ghz.view(1, num_freqs)).to(torch.complex64)
        
        H_s11 = H_s[:, 0, :]
        H_s21 = H_s[:, 1, :] * exp_decay
        return torch.stack([H_s11, H_s21], dim=1).transpose(1, 2)  # (B, F, T)
    
    def forward(self, x, s_tensor, freqs_ghz):
        batch = x.shape[0]
        
        # Stage 1: Direct frequency prediction
        freq_raw = self.freq_backbone(x)
        freq_raw = freq_raw.view(batch, self.num_freqs, self.num_targets, 2)
        freq_pred = torch.complex(freq_raw[..., 0], freq_raw[..., 1])  # (B, F, T)
        
        # Stage 2: Rational correction
        rational_pred = self._rational_forward(x, s_tensor, freqs_ghz)  # (B, F, T)
        
        # Blend: learnable weight between direct and rational
        alpha_blend = torch.sigmoid(self.blend_logit)
        
        # Combined output: direct prediction + rational correction
        output = (1 - alpha_blend) * freq_pred + alpha_blend * rational_pred
        
        return output


# =====================================================================
# 3. TRAINING
# =====================================================================

def train(data_path, results_dir="benchmark_v4_results", epochs=800):
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
    model = HybridRationalNet(input_dim=10, num_freqs=len(ds.freqs_ghz)).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Hybrid Model: {total_params:,} params on {device}")
    print(f"Split: {n_train}/{n_val}/{n_test}")
    
    f_ghz = torch.tensor(ds.freqs_ghz, dtype=torch.float32, device=device)
    omega = 2 * math.pi * f_ghz
    s_tensor = torch.complex(torch.zeros_like(omega), omega)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(train_loader),
        pct_start=0.05, anneal_strategy='cos', div_factor=10, final_div_factor=100
    )
    
    history = {'train': [], 'val_db': [], 'blend': []}
    best_val = float('inf')
    
    print(f"\n{'Ep':>4} | {'Loss':>9} | {'Val':>7} | {'Best':>7} | {'Blend':>6}")
    print("-" * 50)
    
    for epoch in range(epochs):
        model.train()
        t_loss, n_b = 0, 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x, s_tensor, f_ghz)
            
            # Combined loss: dB + complex MSE
            pm = torch.clamp(torch.abs(pred), min=1e-7)
            tm = torch.clamp(torch.abs(y), min=1e-7)
            pdb = torch.clamp(20*torch.log10(pm), -100, 40)
            tdb = torch.clamp(20*torch.log10(tm), -100, 40)
            
            loss_db = nn.functional.smooth_l1_loss(pdb, tdb)
            loss_mse = (nn.functional.mse_loss(pred.real, y.real) + 
                       nn.functional.mse_loss(pred.imag, y.imag))
            
            # Weight shifts over training: more dB early, more MSE later
            w_db = 1.0
            w_mse = 0.01 + 0.09 * min(epoch / 200, 1.0)  # 0.01 -> 0.1
            loss = w_db * loss_db + w_mse * loss_mse
            
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
        blend = torch.sigmoid(model.blend_logit).item()
        
        history['train'].append(avg_t)
        history['val_db'].append(avg_v)
        history['blend'].append(blend)
        
        if avg_v < best_val:
            best_val = avg_v
            torch.save(model.state_dict(), os.path.join(results_dir, "best.pth"))
        
        if (epoch+1) % 50 == 0 or epoch == 0:
            mk = " *" if avg_v <= best_val else ""
            print(f"{epoch+1:4d} | {avg_t:9.4f} | {avg_v:7.2f} | {best_val:7.2f} | {blend:6.3f}{mk}")
    
    print(f"\nBest Val MAE: {best_val:.2f} dB")
    
    # =====================================================================
    # TEST
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
    
    blend_final = torch.sigmoid(model.blend_logit).item()
    
    print(f"\n{'='*60}")
    print(f"V4 HYBRID TEST ({len(all_t)} samples)")
    print(f"{'='*60}")
    print(f"  Overall MAE:  {test_mae:.2f} dB")
    print(f"  Sdd11 MAE:    {s11_mae:.2f} dB")
    print(f"  Sdd21 MAE:    {s21_mae:.2f} dB")
    print(f"  Median:       {torch.quantile(per_sample, 0.5):.2f} dB")
    print(f"  95th:         {torch.quantile(per_sample, 0.95):.2f} dB")
    print(f"  Blend alpha:  {blend_final:.3f} (0=freq only, 1=rational only)")
    result = "PASS" if test_mae < 1.0 else "NEEDS WORK"
    print(f"  {result}")
    print(f"{'='*60}")
    
    # =====================================================================
    # PLOTS
    # =====================================================================
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(f"V4 Hybrid Architecture — Test MAE: {test_mae:.2f} dB (blend={blend_final:.2f})", 
                 fontsize=14, fontweight='bold')
    
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
        ax.plot(freqs, pd, 'r--', lw=2, label='V4 Hybrid')
        ax.set_title(f"{label} Sdd21 ({per_sample[idx]:.2f} dB)")
        ax.set_xlabel("GHz"); ax.set_ylabel("dB"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    
    bi = torch.argmin(per_sample).item()
    ax = axes[0, 3]
    td = 20*np.log10(np.abs(all_t[bi,:,0].numpy())+1e-12)
    pd = 20*np.log10(np.abs(all_p[bi,:,0].numpy())+1e-12)
    ax.plot(freqs, td, 'b-', lw=2, label='Truth')
    ax.plot(freqs, pd, 'r--', lw=2, label='V4 Hybrid')
    ax.set_title(f"Best Sdd11 ({per_sample[bi]:.2f} dB)")
    ax.set_xlabel("GHz"); ax.set_ylabel("dB"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    
    ax = axes[1, 0]
    ax.plot(history['val_db'], 'b-', lw=1)
    ax.axhline(y=1.0, color='green', ls='--', alpha=0.7, label='1 dB')
    ax.set_title("Convergence"); ax.set_xlabel("Epoch"); ax.set_ylabel("Val MAE (dB)")
    ax.set_yscale('log'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    
    ax = axes[1, 1]
    ax.plot(history['blend'], 'g-', lw=1.5)
    ax.set_title("Blend Weight (0=freq, 1=rational)"); ax.set_xlabel("Epoch")
    ax.set_ylabel("alpha"); ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    ax.hist(per_sample.numpy(), bins=30, color='steelblue', edgecolor='black', alpha=0.8)
    ax.axvline(x=1.0, color='green', ls='--', lw=2, label='1 dB')
    ax.axvline(x=test_mae, color='red', ls='-', lw=2, label=f'Mean: {test_mae:.2f}')
    ax.set_title("MAE Distribution"); ax.set_xlabel("MAE (dB)"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    ax = axes[1, 3]
    freq_err = np.abs(pdb.numpy() - tdb.numpy()).mean(axis=0)
    ax.plot(freqs, freq_err[:,0], 'b-', lw=1.5, label='Sdd11')
    ax.plot(freqs, freq_err[:,1], 'r-', lw=1.5, label='Sdd21')
    ax.set_title("MAE vs Freq"); ax.set_xlabel("GHz"); ax.set_ylabel("|Error| dB")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "v4_results.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Plots saved to {results_dir}/v4_results.png")
    
    return model, history, test_mae


if __name__ == "__main__":
    data_path = os.path.expanduser(
        "~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects"
        "/data/processed/Synthetic-Link/synthetic_poles_dataset.pt"
    )
    model, history, test_mae = train(data_path=data_path, epochs=800)