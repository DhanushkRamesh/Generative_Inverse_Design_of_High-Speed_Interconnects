"""
evaluate_tto.py
--------------------------------------------------
Test-Time Optimization (TTO) Inference Pipeline.

This script demonstrates the full Inverse Design capability. 
1. It uses the trained Tandem cVAE to generate an initial geometric guess.
2. It uses the Frozen Forward Surrogate and the Element-Aware Weight Tensor 
   to run gradient descent directly on the predicted physical geometry.
3. It plots the Target vs. cVAE Guess vs. TTO Polished geometry S-parameters.

Author: Lead ML/EM Researcher
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path.home() / "mece_project_inverse_model" / "Generative_Inverse_Design_of_High-Speed_Interconnects"
DATA_PT = PROJECT_ROOT / "data" / "processed" / "Universal-Diff-SI-Array" / "diff_pair_dataset.pt"
WEIGHT_TENSOR_PATH = PROJECT_ROOT / "sandbox_v1" / "data" / "frequency_eda" / "weights_element_aware_per_freq.npy"

# Update this path if your run directory name differs
FORWARD_MODEL_PATH = PROJECT_ROOT / "sandbox_v1" / "models" / "forward_runs" / "run_2026-06-01_213554_direct_resnet_array" / "checkpoint_best.pt"

# Automatically find the latest element-aware cVAE run
INVERSE_RUNS_DIR = PROJECT_ROOT / "sandbox_v1" / "models" / "inverse_runs"
latest_cvae_dir = sorted(list(INVERSE_RUNS_DIR.glob("run_*_inverse_element_aware_*")))[-1]
CVAE_MODEL_PATH = latest_cvae_dir / "cvae_element_aware_best.pt"

OUT_DIR = PROJECT_ROOT / "sandbox_v1" / "models" / "evaluation_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# ARCHITECTURE DEFINITIONS
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
        self.geom_mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.learnable_velocity = nn.Parameter(torch.tensor(10.0))
        self.proj_in = nn.Conv1d(hidden_dim + 3, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([Conv1DResBlock(hidden_dim, dropout=0.10) for _ in range(n_blocks)])
        self.proj_out = nn.Conv1d(hidden_dim, 20, kernel_size=1)
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
        x = torch.cat([x_local, x_global, x_context], dim=-1)
        h_geom = self.geom_mlp(x)
        h_seq = h_geom.unsqueeze(-1).expand(-1, -1, self.F_len)
        f_norm = (freqs_hz / freqs_hz.max()).view(1, 1, self.F_len).expand(B, 1, -1).to(h_seq.dtype)
        length_feature = x_global[:, -1].view(B, 1, 1) if self.is_link_dataset else torch.zeros((B, 1, 1), device=x.device, dtype=h_seq.dtype)
        phase = self.learnable_velocity * f_norm * length_feature
        h = torch.cat([h_seq, f_norm, torch.sin(phase), torch.cos(phase)], dim=1)
        h = self.proj_in(h)
        for block in self.blocks: h = block(h)
        out = self.proj_out(h)
        return self._scatter_symmetric(out[:, :10, :], out[:, 10:, :], B)

class SConditionEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(20, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(), nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(128, out_dim)
    def forward(self, S_complex: torch.Tensor) -> torch.Tensor:
        B = S_complex.shape[0]
        s_real = S_complex[:, :, DirectSequenceResNet.UPPER_R, DirectSequenceResNet.UPPER_C].real.permute(0, 2, 1).float() 
        s_imag = S_complex[:, :, DirectSequenceResNet.UPPER_R, DirectSequenceResNet.UPPER_C].imag.permute(0, 2, 1).float() 
        return self.fc(self.conv(torch.cat([s_real, s_imag], dim=1)).view(B, 128))

class Tandem_cVAE(nn.Module):
    def __init__(self, d_local=8, d_global=6, d_context=7, latent_dim=16, cond_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.s_encoder = SConditionEncoder(out_dim=cond_dim)
        total_cond_dim = cond_dim + d_global + d_context
        self.enc_mlp = nn.Sequential(nn.Linear(d_local + total_cond_dim, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU())
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.dec_mlp = nn.Sequential(nn.Linear(latent_dim + total_cond_dim, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, d_local))
    def forward(self, x_local, x_global, x_context, S_target):
        cond = torch.cat([self.s_encoder(S_target), x_global, x_context], dim=1)
        h = self.enc_mlp(torch.cat([x_local, cond], dim=1))
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return self.dec_mlp(torch.cat([z, cond], dim=1)), mu, logvar

# =============================================================================
# TTO EXECUTION
# =============================================================================
def run_tto_inference(sample_idx=42, tto_steps=75, lr=0.05):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading data from {DATA_PT.name}...")
    payload = torch.load(DATA_PT, map_location=device, weights_only=False)
    
    freqs_hz = payload["frequencies"].to(device)
    freqs_ghz = freqs_hz.cpu().numpy() / 1e9
    
    # Load Weights
    print("Loading Element-Aware Weight Tensor...")
    w_numpy = np.load(WEIGHT_TENSOR_PATH)
    w_tensor = torch.from_numpy(w_numpy).float().to(device)
    w_bcast = w_tensor.permute(2, 0, 1).unsqueeze(0)

    # Extract one test sample
    xg = payload["X_global"][sample_idx:sample_idx+1].to(device)
    xc = payload["X_context"][sample_idx:sample_idx+1].to(device)
    xl_true = payload["X_local"][sample_idx:sample_idx+1].to(device)
    S_tgt = torch.complex(payload["Y_real"][sample_idx:sample_idx+1].to(torch.float64), 
                          payload["Y_imag"][sample_idx:sample_idx+1].to(torch.float64)).to(device)

    # Load Models
    print("Loading Models...")
    forward_model = DirectSequenceResNet(d_local=xl_true.shape[1], d_global=xg.shape[1], d_context=xc.shape[1], F_len=len(freqs_hz)).to(device)
    forward_model.load_state_dict(torch.load(FORWARD_MODEL_PATH, map_location=device, weights_only=True))
    forward_model.eval()
    for param in forward_model.parameters(): param.requires_grad = False

    cvae = Tandem_cVAE(d_local=xl_true.shape[1], d_global=xg.shape[1], d_context=xc.shape[1]).to(device)
    cvae.load_state_dict(torch.load(CVAE_MODEL_PATH, map_location=device, weights_only=True))
    cvae.eval()
    for param in cvae.parameters(): param.requires_grad = False

    # 1. cVAE Base Guess
    print("Generating Initial Guess via cVAE...")
    xl_guess, _, _ = cvae(torch.zeros_like(xl_true), xg, xc, S_tgt) # Dummy input for VAE prior
    S_guess = forward_model(xl_guess, xg, xc, freqs_hz)

    # 2. Setup TTO
    print(f"Running Test-Time Optimization (TTO) for {tto_steps} steps...")
    xl_opt = xl_guess.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([xl_opt], lr=lr)

    for step in range(tto_steps):
        optimizer.zero_grad()
        S_pred = forward_model(xl_opt, xg, xc, freqs_hz)
        
        diff_sq_real = (S_pred.real.float() - S_tgt.real.float()).pow(2)
        diff_sq_imag = (S_pred.imag.float() - S_tgt.imag.float()).pow(2)
        
        # Apply Element-Aware Weights during TTO!
        loss = (diff_sq_real * w_bcast).mean() + (diff_sq_imag * w_bcast).mean()
        
        loss.backward()
        optimizer.step()
        
        if step % 15 == 0 or step == tto_steps - 1:
            print(f"  TTO Step {step:02d} | Physics Loss: {loss.item():.5f}")

    S_tto = forward_model(xl_opt, xg, xc, freqs_hz)
    
    # 3. Visualization
    print("Generating Thesis Plots...")
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Convert to dB
    def to_db(S_tensor, r, c):
        return 20 * torch.log10(S_tensor[0, :, r, c].abs() + 1e-12).detach().cpu().numpy()

    # Sdd11 (Row 0, Col 0)
    ax1.plot(freqs_ghz, to_db(S_tgt, 0, 0), 'k-', lw=2.5, label='Target (Ground Truth)')
    ax1.plot(freqs_ghz, to_db(S_guess, 0, 0), 'tab:red', ls='--', lw=2.0, label='cVAE Initial Guess')
    ax1.plot(freqs_ghz, to_db(S_tto, 0, 0), 'tab:green', ls='-', lw=2.0, label='TTO Polished Geometry')
    ax1.axvspan(0, 28, alpha=0.1, color='orange', label='112G Nyquist (Heavily Weighted)')
    ax1.set_title("Return Loss (|Sdd11|)")
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_ylim(-60, 5)
    ax1.legend()

    # Sdd21 (Row 1, Col 0)
    ax2.plot(freqs_ghz, to_db(S_tgt, 1, 0), 'k-', lw=2.5, label='Target (Ground Truth)')
    ax2.plot(freqs_ghz, to_db(S_guess, 1, 0), 'tab:red', ls='--', lw=2.0, label='cVAE Initial Guess')
    ax2.plot(freqs_ghz, to_db(S_tto, 1, 0), 'tab:green', ls='-', lw=2.0, label='TTO Polished Geometry')
    ax2.axvspan(28, 56, alpha=0.1, color='blue', label='112G Harmonic Band')
    ax2.set_title("Insertion Loss (|Sdd21|)")
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_ylim(-60, 5)
    ax2.legend()

    plt.suptitle("Inverse Design Pipeline: cVAE Guess vs. Test-Time Optimization (TTO)", fontsize=14)
    plt.tight_layout()
    
    save_path = OUT_DIR / f"tto_validation_sample_{sample_idx}.png"
    plt.savefig(save_path)
    print(f"Validation plot saved to: {save_path.name}")
    print("TTO completely successful.")

if __name__ == "__main__":
    run_tto_inference(sample_idx=42) # Pick any index from the dataset to test