"""
==========================================================================
REALISTIC SYNTHETIC POLE-RESIDUE BENCHMARK FOR RATIONAL LAYER VALIDATION
==========================================================================
Author: Dhanush Kumar Ramesh
Purpose: Generate synthetic S-parameter data that faithfully reproduces
         the electromagnetic complexity of the TUHH SI/PI database,
         while retaining exact pole-residue ground truth for validation.

Design Philosophy (from 30 years of RF/SI engineering):
-------------------------------------------------------
The previous synthetic generator had THREE fatal flaws:
  1. LINEAR mapping from X -> poles (trivial for any MLP to learn)
  2. Only 3 resonances (TUHH vias show 5-12+ resonances)
  3. No frequency-dependent loss envelope (real Sdd21 drops 30-50 dB)

This generator fixes all three by introducing:
  - A FROZEN random MLP as the "physics oracle" (nonlinear X->pole map)
  - Variable pole count per sample (3-8 resonances, like real vias)
  - Proper dielectric/conductor loss envelope (sqrt(f) + f^2 model)
  - Realistic Sdd11/Sdd21 magnitude ranges matching TUHH statistics
  - Passivity-aware residue scaling (|S| <= 1 at low frequency)

The model under test must learn the NONLINEAR mapping. If it achieves
<0.5 dB MAE here, it is ready for TUHH data.
==========================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import math


# =====================================================================
# 1. THE FROZEN PHYSICS ORACLE
# =====================================================================
# This replaces the trivial linear mapping X -> poles.
# A randomly initialized (but FROZEN) MLP acts as the unknown physics.
# The trainable PRNet must learn to approximate this nonlinear mapping.
# This is what makes the benchmark meaningful.
# =====================================================================

class FrozenPhysicsOracle(nn.Module):
    """
    A frozen (non-trainable) MLP that maps 10 geometry features
    to pole/residue parameters. This simulates the unknown nonlinear
    EM physics that connects PCB geometry to S-parameter behavior.
    
    The oracle outputs are bounded to physically meaningful ranges:
      - alpha (damping):       [-3.0, -0.05]  (stable, passive)
      - f_res (resonance GHz): [2.0, 98.0]    (within measurement band)
      - c_mag (residue mag):   [0.1, 8.0]     (covers TUHH dynamic range)
      - c_phase (residue arg): [-pi, pi]       (full phase rotation)
      - loss_alpha:            [0.001, 0.02]   (conductor loss coefficient)
      - loss_beta:             [1e-5, 5e-4]    (dielectric loss coefficient)
    """
    def __init__(self, input_dim=10, max_poles=8, seed=42):
        super().__init__()
        self.max_poles = max_poles
        
        # Total outputs: per-pole (4 params x max_poles x 2 targets) + 2 loss params
        self.out_dim = (4 * max_poles * 2) + 2
        
        # Deeper than the trainable model — the oracle is "smarter"
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.Tanh(),
            nn.Linear(128, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, self.out_dim)
        )
        
        # FREEZE all weights — this is ground truth, never trained
        torch.manual_seed(seed)
        for p in self.net.parameters():
            nn.init.xavier_normal_(p) if p.dim() > 1 else nn.init.zeros_(p)
        for p in self.net.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        """Returns dict of bounded physical parameters."""
        raw = self.net(x)
        
        idx = 0
        params = {}
        for target_name in ['s11', 's21']:
            # Alpha (damping) — negative, bounded
            # MINIMUM damping of -0.3 prevents near-singular poles
            # (previous -0.05 minimum allowed poles too close to jw axis -> inf magnitudes)
            alpha_raw = raw[:, idx:idx+self.max_poles]
            params[f'{target_name}_alpha'] = -(torch.sigmoid(alpha_raw) * 2.7 + 0.3)
            idx += self.max_poles
            
            # Resonance frequency in GHz — spread across band
            f_raw = raw[:, idx:idx+self.max_poles]
            params[f'{target_name}_f_res'] = torch.sigmoid(f_raw) * 96.0 + 2.0
            idx += self.max_poles
            
            # Residue magnitude — kept moderate to prevent |S| >> 1
            # (previous 0.1-8.0 range with low damping caused massive spikes)
            c_mag_raw = raw[:, idx:idx+self.max_poles]
            params[f'{target_name}_c_mag'] = torch.sigmoid(c_mag_raw) * 3.9 + 0.1
            idx += self.max_poles
            
            # Residue phase
            c_phase_raw = raw[:, idx:idx+self.max_poles]
            params[f'{target_name}_c_phase'] = torch.tanh(c_phase_raw) * math.pi
            idx += self.max_poles
        
        # Global loss envelope parameters (gentler to keep Sdd21 in realistic range)
        params['loss_alpha'] = torch.sigmoid(raw[:, idx]) * 0.015 + 0.001
        params['loss_beta'] = torch.sigmoid(raw[:, idx+1]) * 3e-4 + 1e-5
        
        return params


# =====================================================================
# 2. POLE-RESIDUE SYNTHESIS ENGINE
# =====================================================================
# This is the EXACT same math as your Rational Layer forward pass.
# S(s) = sum_n [ c_n/(s-p_n) + conj(c_n)/(s-conj(p_n)) ] + d
# =====================================================================

def synthesize_s_parameters(params, f_ghz, num_active_poles, max_poles=8):
    """
    Given oracle parameters, synthesize complex S-parameters using
    the pole-residue rational function. This is ground truth.
    
    Args:
        params: dict from FrozenPhysicsOracle
        f_ghz: frequency array in GHz
        num_active_poles: tensor of ints, how many poles per sample
        max_poles: maximum pole count
    
    Returns:
        S_complex: (num_samples, num_freqs, 2) complex tensor [S11, S21]
    """
    batch = num_active_poles.shape[0]
    num_freqs = len(f_ghz)
    s = 1j * 2 * np.pi * f_ghz  # Complex frequency variable
    s_tensor = torch.tensor(s, dtype=torch.complex64)
    
    S_out = torch.zeros(batch, num_freqs, 2, dtype=torch.complex64)
    
    for i in range(batch):
        n_poles = num_active_poles[i].item()
        
        for t_idx, target in enumerate(['s11', 's21']):
            alpha = params[f'{target}_alpha'][i, :n_poles].numpy()
            f_res = params[f'{target}_f_res'][i, :n_poles].numpy()
            c_mag = params[f'{target}_c_mag'][i, :n_poles].numpy()
            c_phase = params[f'{target}_c_phase'][i, :n_poles].numpy()
            
            beta = 2 * np.pi * f_res
            poles = alpha + 1j * beta
            residues = c_mag * np.exp(1j * c_phase)
            
            H = np.zeros(num_freqs, dtype=np.complex64)
            for k in range(n_poles):
                p = poles[k]
                c = residues[k]
                H += c / (s - p) + np.conj(c) / (s - np.conj(p))
            
            # Add direct term (small, sample-dependent)
            d_re = 0.1 * np.sin(float(i) * 0.1)
            d_im = 0.05 * np.cos(float(i) * 0.1)
            H += complex(d_re, d_im)
            
            S_out[i, :, t_idx] = torch.tensor(H, dtype=torch.complex64)
        
        # ============================================================
        # CRITICAL: Apply frequency-dependent loss envelope to Sdd21
        # Real PCB traces have: conductor loss ~ sqrt(f) 
        #                       dielectric loss ~ f
        # This is what makes TUHH Sdd21 drop from ~-15 dB to -50 dB
        # ============================================================
        loss_a = params['loss_alpha'][i].item()
        loss_b = params['loss_beta'][i].item()
        
        # Combined loss model (in linear domain)
        loss_envelope = np.exp(-(loss_a * np.sqrt(f_ghz) + loss_b * f_ghz**2))
        loss_tensor = torch.tensor(loss_envelope, dtype=torch.complex64)
        
        S_out[i, :, 1] = S_out[i, :, 1] * loss_tensor
        
        # ============================================================
        # PASSIVITY ENFORCEMENT: Scale so |S| < 1 at DC
        # Real passive structures cannot exceed unity gain
        # ============================================================
        for t_idx in range(2):
            max_mag = torch.max(torch.abs(S_out[i, :, t_idx]))
            if max_mag > 0.95:
                S_out[i, :, t_idx] = S_out[i, :, t_idx] * (0.9 / max_mag)
    
    return S_out


# =====================================================================
# 3. DATASET GENERATION — THE MAIN FUNCTION
# =====================================================================

def generate_realistic_dataset(
    num_samples=2000,
    num_freqs=401,
    input_dim=10,
    max_poles=8,
    min_poles=4,
    oracle_seed=42,
    save_dir=None
):
    """
    Generate a realistic synthetic pole-residue dataset.
    
    Key differences from previous version:
    1. Nonlinear X->pole mapping via frozen oracle (not linear)
    2. Variable pole count per sample (min_poles to max_poles)
    3. Frequency-dependent loss envelope on Sdd21
    4. Passivity enforcement
    5. 2000 samples (closer to TUHH's ~1900 Array / ~1073 Link)
    """
    print("=" * 70)
    print("GENERATING REALISTIC SYNTHETIC POLE-RESIDUE DATASET")
    print("=" * 70)
    
    # 1. Generate input features (normalized 0-1, like Z-scored TUHH data)
    torch.manual_seed(0)  # Reproducible inputs
    X = torch.rand(num_samples, input_dim)
    
    # 2. Create the frozen physics oracle
    oracle = FrozenPhysicsOracle(input_dim=input_dim, max_poles=max_poles, seed=oracle_seed)
    
    # 3. Get oracle predictions (ground truth pole parameters)
    with torch.no_grad():
        params = oracle(X)
    
    # 4. Variable pole count per sample
    # Real TUHH structures: simple 2-layer vias ~3-4 resonances
    #                        dense 7x7 arrays ~8-12 resonances
    # We model this as uniform random between min_poles and max_poles
    torch.manual_seed(1)
    num_active_poles = torch.randint(min_poles, max_poles + 1, (num_samples,))
    
    # 5. Frequency axis (matches TUHH: 0-100 GHz, 401 points)
    f_ghz = np.linspace(0.25, 100.0, num_freqs)
    
    # 6. Synthesize S-parameters
    print(f"Synthesizing {num_samples} samples with {min_poles}-{max_poles} poles...")
    print(f"Frequency: {f_ghz[0]:.2f} to {f_ghz[-1]:.2f} GHz ({num_freqs} points)")
    
    S_complex = synthesize_s_parameters(params, f_ghz, num_active_poles, max_poles)
    
    # 7. Compute statistics to verify realism
    s11_db = 20 * np.log10(np.abs(S_complex[:, :, 0].numpy()) + 1e-12)
    s21_db = 20 * np.log10(np.abs(S_complex[:, :, 1].numpy()) + 1e-12)
    
    print(f"\n--- Dataset Statistics ---")
    print(f"Sdd11 range: [{s11_db.min():.1f}, {s11_db.max():.1f}] dB")
    print(f"Sdd21 range: [{s21_db.min():.1f}, {s21_db.max():.1f}] dB")
    print(f"Sdd11 mean:  {s11_db.mean():.1f} dB")
    print(f"Sdd21 mean:  {s21_db.mean():.1f} dB")
    print(f"Pole count distribution: min={num_active_poles.min().item()}, "
          f"max={num_active_poles.max().item()}, "
          f"mean={num_active_poles.float().mean().item():.1f}")
    
    # 8. Package into your existing pipeline format
    Y_real = S_complex.real.unsqueeze(2)
    Y_real = torch.cat([Y_real, Y_real], dim=2)  # Mimic (N, 401, 2, 1) -> cat for loader
    Y_imag = S_complex.imag.unsqueeze(2)
    Y_imag = torch.cat([Y_imag, Y_imag], dim=2)
    
    dataset = {
        'X_global': X[:, :5],
        'X_local': X[:, 5:],
        'Y_real': Y_real,
        'Y_imag': Y_imag,
        'frequencies': torch.tensor(f_ghz * 1e9, dtype=torch.float32),
        # BONUS: Store ground truth for post-training analysis
        'gt_num_poles': num_active_poles,
        'gt_oracle_seed': oracle_seed,
    }
    
    # 9. Save
    if save_dir is None:
        save_dir = os.path.expanduser(
            "~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects"
            "/data/processed/Synthetic-Link"
        )
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "synthetic_poles_dataset.pt")
    torch.save(dataset, save_path)
    print(f"\nSaved dataset to: {save_path}")
    
    # 10. Generate diagnostic plots
    _plot_diagnostics(S_complex, f_ghz, num_active_poles, save_dir)
    
    return dataset, save_path


def _plot_diagnostics(S_complex, f_ghz, num_active_poles, save_dir):
    """Generate publication-quality diagnostic plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Realistic Synthetic Dataset — Diagnostic Report", fontsize=14, fontweight='bold')
    
    # --- Row 1: Individual sample plots ---
    # Pick 3 samples with different pole counts
    pole_counts = num_active_poles.numpy()
    samples = []
    for target_n in [4, 6, 8]:
        candidates = np.where(pole_counts == target_n)[0]
        if len(candidates) > 0:
            samples.append(candidates[0])
    # Fallback if exact counts don't exist
    while len(samples) < 3:
        samples.append(np.random.randint(len(S_complex)))
    
    for col, idx in enumerate(samples):
        ax = axes[0, col]
        s11_db = 20 * np.log10(np.abs(S_complex[idx, :, 0].numpy()) + 1e-12)
        s21_db = 20 * np.log10(np.abs(S_complex[idx, :, 1].numpy()) + 1e-12)
        
        ax.plot(f_ghz, s11_db, 'b-', linewidth=1.5, label='Sdd11')
        ax.plot(f_ghz, s21_db, 'r-', linewidth=1.5, label='Sdd21')
        ax.set_title(f"Sample {idx} ({pole_counts[idx]} poles)", fontsize=10)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_ylim(-60, 5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # --- Row 2: Statistical overview ---
    # Plot 1: Overlay of 50 random Sdd21 curves (like TUHH analysis)
    ax = axes[1, 0]
    rand_idx = np.random.choice(len(S_complex), min(50, len(S_complex)), replace=False)
    for idx in rand_idx:
        s21_db = 20 * np.log10(np.abs(S_complex[idx, :, 1].numpy()) + 1e-12)
        ax.plot(f_ghz, s21_db, alpha=0.15, color='blue', linewidth=0.5)
    # Mean envelope
    s21_all = 20 * np.log10(np.abs(S_complex[:, :, 1].numpy()) + 1e-12)
    ax.plot(f_ghz, np.mean(s21_all, axis=0), 'r-', linewidth=2, label='Mean Sdd21')
    ax.set_title("Sdd21 Ensemble (50 samples)", fontsize=10)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_ylim(-60, 5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Plot 2: Pole count histogram
    ax = axes[1, 1]
    ax.hist(pole_counts, bins=range(pole_counts.min(), pole_counts.max()+2),
            align='left', color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_title("Pole Count Distribution", fontsize=10)
    ax.set_xlabel("Number of Active Poles")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    
    # Plot 3: dB statistics across frequency
    ax = axes[1, 2]
    s11_all = 20 * np.log10(np.abs(S_complex[:, :, 0].numpy()) + 1e-12)
    ax.fill_between(f_ghz, np.percentile(s21_all, 10, axis=0),
                    np.percentile(s21_all, 90, axis=0), alpha=0.3, color='red', label='Sdd21 (10-90%)')
    ax.fill_between(f_ghz, np.percentile(s11_all, 10, axis=0),
                    np.percentile(s11_all, 90, axis=0), alpha=0.3, color='blue', label='Sdd11 (10-90%)')
    ax.plot(f_ghz, np.median(s21_all, axis=0), 'r-', linewidth=1.5)
    ax.plot(f_ghz, np.median(s11_all, axis=0), 'b-', linewidth=1.5)
    ax.set_title("Statistical Spread (10th-90th percentile)", fontsize=10)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_ylim(-60, 5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "synthetic_diagnostic_report.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Diagnostic plots saved to: {plot_path}")


# =====================================================================
# 4. MAIN ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    dataset, path = generate_realistic_dataset(
        num_samples=2000,
        num_freqs=401,
        input_dim=10,
        max_poles=8,
        min_poles=4,
        oracle_seed=42
    )
    
    print("\n" + "=" * 70)
    print("DONE. Now train your PRNet on this dataset.")
    print("TARGET: < 0.5 dB MAE before moving to TUHH data.")
    print("=" * 70)