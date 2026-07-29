"""
forward_model_v7.py
Implicit Neural Representation (INR) Surrogate for S-Parameters.
Maps (Geometry, Frequency) -> Complex Symmetric S-Matrix directly,
bypassing the gradient traps of pole-residue rational formulations.
"""

import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Multi-scale Fourier Features
# ----------------------------------------------------------------------
class MultiScaleFourierFeatures(nn.Module):
    """Maps continuous inputs to a high-dimensional periodic space."""
    def __init__(self, in_dim: int, sigmas: List[float], n_features_per_scale: int = 32, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        Bs = [torch.randn(in_dim, n_features_per_scale, generator=g) * s for s in sigmas]
        B = torch.cat(Bs, dim=1)
        self.register_buffer("B", B)
        self.out_dim = 2 * n_features_per_scale * len(sigmas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

# ----------------------------------------------------------------------
# Regularized Residual Block
# ----------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Pre-norm residual block with Dropout for generalization."""
    def __init__(self, dim: int, dropout: float = 0.10):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.drop1(F.silu(self.fc1(h)))
        h = self.drop2(self.fc2(h))
        return x + h

# ----------------------------------------------------------------------
# Frequency-Conditioned INR Model
# ----------------------------------------------------------------------
class FrequencyConditionedResNet(nn.Module):
    """
    Directly predicts S-parameters given geometry and frequency.
    Inputs:
        X_local (8), X_global (6), X_context (7)
        freqs_hz (F,)
    Outputs:
        S-parameters (Batch, F, 4, 4) Complex128
    """
    UPPER_R = (0, 0, 0, 0, 1, 1, 1, 2, 2, 3)
    UPPER_C = (0, 1, 2, 3, 1, 2, 3, 2, 3, 3)

    def __init__(self,
                 d_local: int = 8,
                 d_global: int = 6,
                 d_context: int = 7,
                 geom_sigmas: List[float] = [0.5, 1.0, 2.0],
                 freq_sigmas: List[float] = [0.1, 1.0, 10.0, 50.0],
                 fourier_features: int = 32,
                 hidden_dim: int = 512,
                 n_blocks: int = 6,
                 f_scale_hz: float = 100e9):
        super().__init__()
        self.f_scale_hz = f_scale_hz

        # Geometry Encoder
        self.geom_fourier = MultiScaleFourierFeatures(
            in_dim=d_local + d_global + d_context,
            sigmas=geom_sigmas,
            n_features_per_scale=fourier_features
        )
        
        # Frequency Encoder (1D Input)
        self.freq_fourier = MultiScaleFourierFeatures(
            in_dim=1,
            sigmas=freq_sigmas,
            n_features_per_scale=fourier_features
        )

        in_dim = self.geom_fourier.out_dim + self.freq_fourier.out_dim
        
        # Core Processor
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(n_blocks)])
        
        # Output: 10 Real + 10 Imag parts for the symmetric 4x4 matrix
        self.proj_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 20) 
        )

        # Initialize output layer to zero for stable start
        nn.init.zeros_(self.proj_out[-1].weight)
        nn.init.zeros_(self.proj_out[-1].bias)

        self.register_buffer("upper_r", torch.tensor(self.UPPER_R, dtype=torch.long))
        self.register_buffer("upper_c", torch.tensor(self.UPPER_C, dtype=torch.long))

    def _scatter_symmetric(self, vec_real: torch.Tensor, vec_imag: torch.Tensor, 
                           batch_size: int, num_freqs: int) -> torch.Tensor:
        """Scatters (B*F, 10) back into (B, F, 4, 4) complex symmetric matrices."""
        mat = torch.zeros((batch_size * num_freqs, 4, 4), dtype=torch.complex128, device=vec_real.device)
        cmplx_vec = torch.complex(vec_real.to(torch.float64), vec_imag.to(torch.float64))
        
        mat[:, self.upper_r, self.upper_c] = cmplx_vec
        mat[:, self.upper_c, self.upper_r] = cmplx_vec
        
        return mat.view(batch_size, num_freqs, 4, 4)

    def forward(self, x_local: torch.Tensor, x_global: torch.Tensor, 
                x_context: torch.Tensor, freqs_hz: torch.Tensor) -> torch.Tensor:
        
        B = x_local.shape[0]
        F_len = freqs_hz.shape[0]

        # 1. Combine Geometry
        geom_raw = torch.cat([x_local, x_global, x_context], dim=-1) # (B, 21)
        geom_encoded = self.geom_fourier(geom_raw) # (B, D_geom)
        
        # 2. Normalize and Encode Frequency
        f_norm = (freqs_hz / self.f_scale_hz).view(-1, 1).to(geom_raw.dtype) # (F, 1)
        freq_encoded = self.freq_fourier(f_norm) # (F, D_freq)
        
        # 3. Cartesian Expansion (B x F)
        # Expand geometry to (B, F, D_geom)
        geom_exp = geom_encoded.unsqueeze(1).expand(-1, F_len, -1).reshape(B * F_len, -1)
        # Expand frequency to (B, F, D_freq)
        freq_exp = freq_encoded.unsqueeze(0).expand(B, -1, -1).reshape(B * F_len, -1)
        
        # 4. Fuse and process
        x = torch.cat([geom_exp, freq_exp], dim=-1)
        h = self.proj_in(x)
        for block in self.blocks:
            h = block(h)
        out = self.proj_out(h) # (B*F, 20)

        # 5. Split Real/Imag and Scatter
        vec_real = out[:, :10]
        vec_imag = out[:, 10:]
        
        S_matrix = self._scatter_symmetric(vec_real, vec_imag, B, F_len)
        return S_matrix

def _self_test():
    """Validates structural integrity and Cartesian batching."""
    torch.manual_seed(42)
    model = FrequencyConditionedResNet()
    B = 4
    F_len = 401
    xl = torch.randn(B, 8)
    xg = torch.randn(B, 6)
    xc = torch.randn(B, 7)
    freqs = torch.linspace(0.25e9, 100e9, F_len)
    
    S = model(xl, xg, xc, freqs)
    assert S.shape == (B, F_len, 4, 4), f"Shape mismatch: {S.shape}"
    assert S.dtype == torch.complex128
    
    # Reciprocity
    asym = (S - S.transpose(-1, -2)).abs().max().item()
    assert asym < 1e-8, "Reciprocity failed."
    
    print("Architecture Self-Test Passed.")

if __name__ == "__main__":
    _self_test()