"""
tandem_cvae_inverse_masked.py
--------------------------------------------------
Generative Inverse Design using a cVAE with Piecewise Masked Physics Loss.

This script implements Novel Upgrade 2 (Supervisor Point 1 & 3): 
The S-parameter physics loss is dynamically weighted based on discrete frequency 
regions for 112G PAM4 signaling. 
- 0 to 30 GHz (Baseband): Weight 1.0x
- 30 to 65 GHz (Harmonics): Weight 3.0x
- 65+ GHz (Numerical Noise): Weight 0.0x (Ignored)

Reference: Zhu et al. (2024), "Conjugate Adjoint Gradient-Based Inverse Design..."
Author: Lead ML/EM Researcher
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

PROJECT_ROOT = Path.home() / "mece_project_inverse_model" / "Generative_Inverse_Design_of_High-Speed_Interconnects"
RUNS_DIR = PROJECT_ROOT / "sandbox_v1" / "models" / "inverse_runs"

# REQUIRES MANUAL UPDATE: Point to your final 1.94 dB Forward Model checkpoint
FORWARD_MODEL_PATH = PROJECT_ROOT / "sandbox_v1" / "models" / "forward_runs" / "run_2026-06-01_213554_direct_resnet_array" / "checkpoint_best.pt"

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

class DiffPairDataset(Dataset):
    def __init__(self, payload: dict, indices: np.ndarray):
        self.payload = payload
        self.indices = indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        return self.payload["X_local"][self.indices[i]], self.payload["X_global"][self.indices[i]], self.payload["X_context"][self.indices[i]], torch.complex(self.payload["Y_real"][self.indices[i]].to(torch.float64), self.payload["Y_imag"][self.indices[i]].to(torch.float64))

# =============================================================================
# MASKED TRAINING LOOP
# =============================================================================
def train_inverse_epoch_masked(cvae, forward_model, loader, freqs_hz, device, optimizer):
    cvae.train()
    acc = {"loss": 0.0, "kld": 0.0, "recon_x": 0.0, "physics_s": 0.0, "n": 0}
    pbar = tqdm(loader, desc="Training Batches", leave=False)
    
    # -------------------------------------------------------------------------
    # NOVEL METHOD: Piecewise Frequency Region Splitting (Ref: Zhu et al. 2024)
    # -------------------------------------------------------------------------
    f_ghz = freqs_hz / 1e9  # Convert frequencies to GHz
    
    # Define physical regions
    mask_region1 = (f_ghz <= 30.0).float()                     # Baseband
    mask_region2 = ((f_ghz > 30.0) & (f_ghz <= 65.0)).float()  # Critical Harmonics
    mask_region3 = (f_ghz > 65.0).float()                      # Deep Noise
    
    # Apply weights (1.0x, 3.0x, 0.0x) and broadcast to correct tensor shape
    f_weights_1d = (mask_region1 * 1.0) + (mask_region2 * 3.0) + (mask_region3 * 0.0)
    f_weights = f_weights_1d.view(1, -1, 1, 1).to(device)
    # -------------------------------------------------------------------------

    for xl, xg, xc, S_tgt in pbar:
        xl, xg, xc, S_tgt = xl.to(device), xg.to(device), xc.to(device), S_tgt.to(device)
        B = xl.shape[0]

        xl_gen, mu, logvar = cvae(xl, xg, xc, S_tgt)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
        recon_x_loss = F.mse_loss(xl_gen, xl) 
        
        # Unreduced MSE Loss
        S_gen = forward_model(xl_gen, xg, xc, freqs_hz)
        err_real = F.mse_loss(S_gen.real.float(), S_tgt.real.float(), reduction='none')
        err_imag = F.mse_loss(S_gen.imag.float(), S_tgt.imag.float(), reduction='none')
        
        # Apply the Masked Frequency Penalty
        physics_loss = (err_real * f_weights).mean() + (err_imag * f_weights).mean()
        
        total_loss = (1.0 * recon_x_loss) + (0.01 * kld_loss) + (10.0 * physics_loss)

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(cvae.parameters(), 1.0)
        optimizer.step()

        acc["loss"] += total_loss.item() * B
        acc["recon_x"] += recon_x_loss.item() * B
        acc["physics_s"] += physics_loss.item() * B
        acc["kld"] += kld_loss.item() * B
        acc["n"] += B
        pbar.set_postfix({"PhysL": f"{physics_loss.item():.4f}"})

    return {k: v/acc["n"] for k, v in acc.items() if k != "n"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Array")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(PROJECT_ROOT / "data" / "processed" / f"Universal-Diff-SI-{args.dataset}" / "diff_pair_dataset.pt", weights_only=False)
    
    sim_ids = np.array(payload["sim_ids"])
    unique_sims = np.unique(sim_ids)
    np.random.default_rng(42).shuffle(unique_sims)
    train_idx = np.array([i for i, sid in enumerate(sim_ids) if sid in set(unique_sims[:int(0.85 * len(unique_sims))])])
    train_loader = DataLoader(DiffPairDataset(payload, train_idx), batch_size=args.batch_size, shuffle=True)
    freqs_hz = payload["frequencies"].to(device)

    forward_model = DirectSequenceResNet(d_local=payload["X_local"].shape[1], d_global=payload["X_global"].shape[1], d_context=payload["X_context"].shape[1], F_len=len(freqs_hz)).to(device)
    forward_model.load_state_dict(torch.load(FORWARD_MODEL_PATH, map_location=device, weights_only=True))
    forward_model.eval()
    for param in forward_model.parameters(): param.requires_grad = False

    cvae = Tandem_cVAE(d_local=payload["X_local"].shape[1], d_global=payload["X_global"].shape[1], d_context=payload["X_context"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(cvae.parameters(), lr=5e-4, weight_decay=1e-4)

    run_dir = RUNS_DIR / datetime.now().strftime(f"run_%Y-%m-%d_%H%M%S_inverse_masked_{args.dataset.lower()}")
    run_dir.mkdir(parents=True, exist_ok=True)
    best_physics = float('inf')

    print("\nInitiating Piecewise-Masked Tandem Generative Training...")
    for epoch in range(1, args.epochs + 1):
        stats = train_inverse_epoch_masked(cvae, forward_model, train_loader, freqs_hz, device, optimizer)
        marker = " <-- [NEW BEST]" if stats['physics_s'] < best_physics else ""
        if marker: 
            best_physics = stats['physics_s']
            torch.save(cvae.state_dict(), run_dir / "cvae_masked_best.pt")
            
        print(f"Ep {epoch:03d} | PhysL: {stats['physics_s']:.4f} | Recon: {stats['recon_x']:.4f} | KLD: {stats['kld']:.4f}{marker}")

if __name__ == "__main__":
    main()