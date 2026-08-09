"""
train_tandem_cvae_inverse_masked_freq.py
--------------------------------------------------
Generative Inverse Design using a cVAE with Element-Aware Physics Loss.

This script implements the ultimate Physics-Guided Loss strategy derived from 
the dataset variance analysis (04b_element_aware_weights.py). 

Instead of a generic MSE or naive 1D frequency penalty, the S-parameter physics 
loss uses a (4, 4, F) weight tensor. This ensures the optimizer treats Return Loss 
(Sdd11), Insertion Loss (Sdd21), and Mode Conversion (Sdc) independently, anchoring
the geometry generation strictly to 112G PAM4 signaling constraints while masking
irrelevant high-frequency solver noise.

Reference: LaBash et al. 2025 (arXiv:2505.18188)
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
RUNS_DIR = PROJECT_ROOT / "src" / "models" / "inverse_runs"

# Pre-trained components
FORWARD_MODEL_PATH = PROJECT_ROOT / "src" / "models" / "forward_runs" / "run_2026-06-01_213554_direct_resnet_array" / "checkpoint_best.pt"
WEIGHT_TENSOR_PATH = PROJECT_ROOT / "results" / "data" / "frequency_eda" / "array" / "weights_element_aware_per_freq.npy"

# =============================================================================
# ARCHITECTURE DEFINITIONS (Frozen Evaluator & Generator)
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
# frozen_forward_model: The frozen DirectSequenceResNet surrogate evaluator.
class DirectSequenceResNet(nn.Module):
    """The Frozen Forward Surrogate Evaluator."""
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
        return S.permute(0, 3, 1, 2) # Output shape: (B, F_len, 4, 4)

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
# encoder for S-parameter conditions: Converts (B, F, 4, 4) to a compact (B, cond_dim) representation.
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
# tandem_cVAE: The generative inverse model that predicts geometry from S-parameter conditions.
class Tandem_cVAE(nn.Module):
    """The Generative Inverse Model."""
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
# differential dataset wrapper: Provides a PyTorch Dataset interface for the loaded payload.
class DiffPairDataset(Dataset):
    def __init__(self, payload: dict, indices: np.ndarray):
        self.payload = payload
        self.indices = indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        return self.payload["X_local"][self.indices[i]], self.payload["X_global"][self.indices[i]], self.payload["X_context"][self.indices[i]], torch.complex(self.payload["Y_real"][self.indices[i]].to(torch.float64), self.payload["Y_imag"][self.indices[i]].to(torch.float64))

# =============================================================================
# ELEMENT-AWARE TRAINING LOOP
# =============================================================================
def train_inverse_epoch_element_aware(cvae, forward_model, loader, freqs_hz, w_bcast, device, optimizer):
    """
    Executes a training epoch using the (1, F, 4, 4) Element-Aware weight tensor.
    """
    cvae.train()
    acc = {"loss": 0.0, "kld": 0.0, "recon_x": 0.0, "physics_s": 0.0, "n": 0}
    pbar = tqdm(loader, desc="Training Batches", leave=False)
    
    for xl, xg, xc, S_tgt in pbar:
        xl, xg, xc, S_tgt = xl.to(device), xg.to(device), xc.to(device), S_tgt.to(device)
        B = xl.shape[0]

        # 1. Generate Hypothesis Geometry
        xl_gen, mu, logvar = cvae(xl, xg, xc, S_tgt)
        
        # 2. VAE Optimization Losses
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
        recon_x_loss = F.mse_loss(xl_gen, xl) 
        
        # 3. Physics Evaluation (Forward Pass of generated geometry)
        S_gen = forward_model(xl_gen, xg, xc, freqs_hz)  # Output Shape: (Batch, Freq, 4, 4)
        
        # 4. The Element-Aware Constraint Loss
        # We calculate unreduced squared error, multiply by the 4x4xF weight tensor, then reduce.
        diff_sq_real = (S_gen.real.float() - S_tgt.real.float()).pow(2)
        diff_sq_imag = (S_gen.imag.float() - S_tgt.imag.float()).pow(2)
        
        physics_loss = (diff_sq_real * w_bcast).mean() + (diff_sq_imag * w_bcast).mean()
        
        # Composite Loss Aggregation
        total_loss = (1.0 * recon_x_loss) + (0.01 * kld_loss) + (10.0 * physics_loss)

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(cvae.parameters(), 1.0)
        optimizer.step()

        # Metrics Tracking
        acc["loss"] += total_loss.item() * B
        acc["recon_x"] += recon_x_loss.item() * B
        acc["physics_s"] += physics_loss.item() * B
        acc["kld"] += kld_loss.item() * B
        acc["n"] += B
        pbar.set_postfix({"PhysL": f"{physics_loss.item():.4f}"})

    return {k: v/acc["n"] for k, v in acc.items() if k != "n"}

# =============================================================================
# MAIN EXECUTOR
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Array")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device Initialization: {device}")

    # 1. Dataset Loading
    payload = torch.load(PROJECT_ROOT / "data" / "processed" / f"Universal-Diff-SI-{args.dataset}" / "diff_pair_dataset.pt", weights_only=False)
    sim_ids = np.array(payload["sim_ids"])
    unique_sims = np.unique(sim_ids)
    np.random.default_rng(42).shuffle(unique_sims)
    train_idx = np.array([i for i, sid in enumerate(sim_ids) if sid in set(unique_sims[:int(0.85 * len(unique_sims))])])
    train_loader = DataLoader(DiffPairDataset(payload, train_idx), batch_size=args.batch_size, shuffle=True)
    freqs_hz = payload["frequencies"].to(device)

    # 2. Load Element-Aware Weight Tensor (From EDA)
    print(f"Loading Physics-Guided Weights: {WEIGHT_TENSOR_PATH.name}")
    w_numpy = np.load(WEIGHT_TENSOR_PATH)  # Shape expected: (4, 4, F_LEN)
    w_tensor = torch.from_numpy(w_numpy).float().to(device)
    
    # Broadcast configuration: (4, 4, F) -> (1, F, 4, 4) to match (Batch, Freq, Port, Port)
    w_bcast = w_tensor.permute(2, 0, 1).unsqueeze(0)

    # 3. Initialize Forward Surrogate (Frozen)
    forward_model = DirectSequenceResNet(d_local=payload["X_local"].shape[1], d_global=payload["X_global"].shape[1], d_context=payload["X_context"].shape[1], F_len=len(freqs_hz)).to(device)
    forward_model.load_state_dict(torch.load(FORWARD_MODEL_PATH, map_location=device, weights_only=True))
    forward_model.eval()
    for param in forward_model.parameters(): param.requires_grad = False

    # 4. Initialize Generative Inverse Model
    cvae = Tandem_cVAE(d_local=payload["X_local"].shape[1], d_global=payload["X_global"].shape[1], d_context=payload["X_context"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(cvae.parameters(), lr=5e-4, weight_decay=1e-4)

    run_dir = RUNS_DIR / datetime.now().strftime(f"run_%d-%m-%y_%H%M%S_inverse_element_aware_{args.dataset.lower()}")
    run_dir.mkdir(parents=True, exist_ok=True)
    best_physics = float('inf')

    # 5. Execution Loop
    print("\nInitiating Element-Aware Tandem Generative Training...")
    # Wrap range with tqdm progress bar
    pbar = tqdm(range(1, args.epochs + 1), desc="Training cVAE", unit="epoch")
    for epoch in pbar:
        stats = train_inverse_epoch_element_aware(
            cvae, forward_model, train_loader, freqs_hz, w_bcast, device, optimizer
        )

        # Check for new best model
        is_best = stats['physics_s'] < best_physics
        if is_best: 
            best_physics = stats['physics_s']
            torch.save(cvae.state_dict(), run_dir / "cvae_element_aware_best.pt")
            # Explicitly print a permanent line when a NEW BEST is saved!
            tqdm.write(f" Ep {epoch:03d} | PhysL: {stats['physics_s']:.4f} | Recon: {stats['recon_x']:.4f} | KLD: {stats['kld']:.4f} <-- [NEW BEST SAVED]")
        # Live update progress bar with ETA, time, and metrics
        pbar.set_postfix({
            "PhysL": f"{stats['physics_s']:.4f}",
            "Recon": f"{stats['recon_x']:.4f}",
            "KLD": f"{stats['kld']:.4f}",
            "Best": f"{best_physics:.4f}"
        })

if __name__ == "__main__":
    main()