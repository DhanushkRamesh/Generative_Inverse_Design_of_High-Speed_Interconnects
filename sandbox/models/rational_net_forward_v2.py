"""
rational_forward_v2.py — improved forward model.

Changes from v1:
  * Multi-scale Fourier features: σ ∈ {1, 4, 16} concatenated, 32 features each.
    Gives the model bandwidth at multiple input scales (NeRF-style).
  * Encoder: 4 layers, LayerNorm + residual connections, hidden=512, latent=384.
    Plain MLPs plateau on the wrong solution. Residual MLP gets there.
  * Pole head: initial weight std=1e-2 (10x v1). Adds a learnable scalar
    `delta_scale` (init=0.5) multiplied into the head output, letting Adam
    smoothly increase pole-head sensitivity as training progresses.
  * Residue head: hidden=384 (was 256). More capacity for the 704-dim output.

Same external interface as v1. Drop-in replacement.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Multi-scale Fourier features
# ----------------------------------------------------------------------
class MultiScaleFourierFeatures(nn.Module):
    """
    Concatenates Fourier features at multiple sigma scales.
    Output dim = 2 * n_features * len(sigmas).
    """

    def __init__(self, in_dim: int, sigmas: List[float] = [1.0, 4.0, 16.0],
                 n_features: int = 32, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        Bs = []
        for s in sigmas:
            Bs.append(torch.randn(in_dim, n_features, generator=g) * s)
        # Stack as (in_dim, n_features * n_scales)
        B = torch.cat(Bs, dim=1)
        self.register_buffer("B", B)
        self.out_dim = 2 * n_features * len(sigmas)
        self.n_scales = len(sigmas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# ----------------------------------------------------------------------
# Residual MLP block
# ----------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """LayerNorm → Linear → SiLU → Linear, with skip connection."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        h = self.norm(x)
        h = F.silu(self.fc1(h))
        h = self.fc2(h)
        return x + h


# ----------------------------------------------------------------------
# Pole head with learnable delta scale
# ----------------------------------------------------------------------
def softplus_inverse(y: torch.Tensor) -> torch.Tensor:
    return y + torch.log1p(-torch.exp(-y))


class PoleHead(nn.Module):
    """
    Predicts pole parameters with stability enforced by softplus.
    A learnable scalar `delta_scale` multiplies the head output, allowing
    the model to control how strongly inputs influence poles. Starts at 0.5
    so the bias-driven warm start dominates initially, and Adam can grow it.
    """

    def __init__(self, latent_dim: int, n_real: int, n_cmplx: int,
                 init_poles_real: torch.Tensor, init_poles_cmplx: torch.Tensor,
                 f_scale_rad: float, hidden: int = 256, init_weight_scale: float = 1e-2,
                 init_delta_scale: float = 0.5):
        super().__init__()
        self.n_real = n_real
        self.n_cmplx = n_cmplx
        self.f_scale_rad = f_scale_rad

        self.body = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.head = nn.Linear(hidden, n_real + 2 * n_cmplx)
        nn.init.normal_(self.head.weight, std=init_weight_scale)

        # Bias to produce target poles at h=0, scaled output
        target_a_real = softplus_inverse(
            torch.as_tensor(-init_poles_real.real / f_scale_rad, dtype=torch.float32).clamp_min(1e-6)
        )
        target_a_cre = softplus_inverse(
            torch.as_tensor(-init_poles_cmplx.real / f_scale_rad, dtype=torch.float32).clamp_min(1e-6)
        )
        target_a_cim = softplus_inverse(
            torch.as_tensor(init_poles_cmplx.imag / f_scale_rad, dtype=torch.float32).clamp_min(1e-6)
        )
        bias = torch.cat([target_a_real, target_a_cre, target_a_cim])
        with torch.no_grad():
            self.head.bias.copy_(bias)

        # Learnable gate on the (raw - bias) deviation
        self.delta_scale = nn.Parameter(torch.tensor(init_delta_scale))

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.head(self.body(h))
        # Decompose: bias is the warm-start target; deviation is what the MLP added
        bias = self.head.bias  # shape (n_real + 2*n_cmplx,)
        deviation = raw - bias  # broadcast
        scaled = bias + self.delta_scale * deviation

        a_real = scaled[:, :self.n_real]
        a_cre = scaled[:, self.n_real:self.n_real + self.n_cmplx]
        a_cim = scaled[:, self.n_real + self.n_cmplx:]

        re_real = -F.softplus(a_real).clamp_min(1e-4)
        re_cmplx = -F.softplus(a_cre).clamp_min(1e-4)
        im_cmplx = F.softplus(a_cim).clamp_min(1e-4)

        poles_real = re_real.to(torch.float64)
        poles_cmplx = torch.complex(re_cmplx.to(torch.float64), im_cmplx.to(torch.float64))
        return poles_real, poles_cmplx


# ----------------------------------------------------------------------
# Residue and D heads
# ----------------------------------------------------------------------
class ResidueHead(nn.Module):
    def __init__(self, latent_dim: int, n_real: int, n_cmplx: int, hidden: int = 384):
        super().__init__()
        self.n_real = n_real
        self.n_cmplx = n_cmplx
        out_dim = n_real * 16 + n_cmplx * 32
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, h):
        B = h.shape[0]
        out = self.net(h)
        r_real_flat = out[:, :self.n_real * 16]
        r_cmplx_flat = out[:, self.n_real * 16:]
        R_real = r_real_flat.view(B, self.n_real, 4, 4).to(torch.float64)
        r_cmplx_split = r_cmplx_flat.view(B, self.n_cmplx, 2, 4, 4)
        R_cmplx = torch.complex(r_cmplx_split[:, :, 0].to(torch.float64),
                                r_cmplx_split[:, :, 1].to(torch.float64))
        return R_real, R_cmplx


class DHead(nn.Module):
    def __init__(self, latent_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 32),
        )

    def forward(self, h):
        B = h.shape[0]
        out = self.net(h).view(B, 2, 4, 4)
        return torch.complex(out[:, 0].to(torch.float64), out[:, 1].to(torch.float64))


# ----------------------------------------------------------------------
# Rational layer evaluation (same as v1)
# ----------------------------------------------------------------------
def evaluate_rational(poles_real, poles_cmplx, R_real, R_cmplx, D, s_norm):
    if not torch.is_complex(poles_real):
        poles_real_c = torch.complex(poles_real, torch.zeros_like(poles_real))
    else:
        poles_real_c = poles_real

    s_expand = s_norm.view(1, 1, -1)
    denom_r = s_expand - poles_real_c.unsqueeze(-1)
    term_r = R_real.unsqueeze(-1) / denom_r.unsqueeze(-2).unsqueeze(-2)
    sum_r = term_r.sum(dim=1)

    denom_c = s_expand - poles_cmplx.unsqueeze(-1)
    denom_cc = s_expand - poles_cmplx.conj().unsqueeze(-1)
    term_c = R_cmplx.unsqueeze(-1) / denom_c.unsqueeze(-2).unsqueeze(-2)
    term_cc = R_cmplx.conj().unsqueeze(-1) / denom_cc.unsqueeze(-2).unsqueeze(-2)
    sum_c = term_c.sum(dim=1) + term_cc.sum(dim=1)

    S = sum_r + sum_c + D.unsqueeze(-1)
    return S.permute(0, 3, 1, 2)


# ----------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------
class RationalForwardModel(nn.Module):
    def __init__(self,
                 freqs_hz: torch.Tensor,
                 init_poles_real: torch.Tensor,
                 init_poles_cmplx: torch.Tensor,
                 d_local: int = 8,
                 d_global: int = 6,
                 d_context: int = 7,
                 fourier_sigmas: List[float] = [1.0, 4.0, 16.0],
                 fourier_features: int = 32,
                 latent_dim: int = 384,
                 encoder_hidden: int = 512,
                 n_encoder_blocks: int = 4,
                 f_scale_hz: float = 100e9):
        super().__init__()
        self.n_real = len(init_poles_real)
        self.n_cmplx = len(init_poles_cmplx)
        self.f_scale_rad = 2 * math.pi * f_scale_hz

        omega_norm = 2 * math.pi * freqs_hz / self.f_scale_rad
        s_norm = torch.complex(torch.zeros_like(omega_norm), omega_norm).to(torch.complex128)
        self.register_buffer("s_norm", s_norm)

        self.fourier = MultiScaleFourierFeatures(
            d_local, sigmas=fourier_sigmas, n_features=fourier_features
        )
        encoder_in = self.fourier.out_dim + d_global + d_context

        # Encoder: project to hidden, then n_encoder_blocks residual blocks, then project to latent
        self.proj_in = nn.Linear(encoder_in, encoder_hidden)
        self.blocks = nn.ModuleList([ResidualBlock(encoder_hidden) for _ in range(n_encoder_blocks)])
        self.proj_out = nn.Sequential(
            nn.LayerNorm(encoder_hidden),
            nn.SiLU(),
            nn.Linear(encoder_hidden, latent_dim),
            nn.SiLU(),
        )

        self.pole_head = PoleHead(
            latent_dim, self.n_real, self.n_cmplx,
            init_poles_real, init_poles_cmplx, self.f_scale_rad,
        )
        self.residue_head = ResidueHead(latent_dim, self.n_real, self.n_cmplx)
        self.d_head = DHead(latent_dim)

    def encode(self, x_local, x_global, x_context):
        ff = self.fourier(x_local)
        x = torch.cat([ff, x_global, x_context], dim=-1)
        h = self.proj_in(x)
        for block in self.blocks:
            h = block(h)
        return self.proj_out(h)

    def forward(self, x_local, x_global, x_context):
        h = self.encode(x_local, x_global, x_context)
        poles_real_norm, poles_cmplx_norm = self.pole_head(h)
        R_real, R_cmplx = self.residue_head(h)
        D = self.d_head(h)
        return evaluate_rational(poles_real_norm, poles_cmplx_norm, R_real, R_cmplx, D, self.s_norm)

    def get_poles_si_units(self, x_local, x_global, x_context):
        h = self.encode(x_local, x_global, x_context)
        pr, pc = self.pole_head(h)
        return pr * self.f_scale_rad, pc * self.f_scale_rad


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------
def _self_test():
    torch.manual_seed(0)
    init_poles_real = torch.tensor([-1e10, -1e11, -1e12, -1e13], dtype=torch.float64)
    init_poles_cmplx = torch.tensor(
        [-1e10 + 1j*5e10, -2e10 + 1j*1.5e11, -3e10 + 1j*3e11,
         -2e10 + 1j*5e11, -1e10 + 1j*7e11, -5e10 + 1j*9e11],
        dtype=torch.complex128,
    )
    freqs = torch.linspace(0.25e9, 100e9, 401, dtype=torch.float64)
    model = RationalForwardModel(freqs, init_poles_real, init_poles_cmplx)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model parameters: {n_params:,}")

    B = 4
    xl = torch.randn(B, 8); xg = torch.randn(B, 6); xc = torch.randn(B, 7)
    S = model(xl, xg, xc)
    print(f"  forward pass OK, S shape: {tuple(S.shape)}  dtype: {S.dtype}")
    assert S.shape == (B, 401, 4, 4) and S.dtype == torch.complex128

    pr, pc = model.get_poles_si_units(xl, xg, xc)
    assert (pr.real < 0).all() and (pc.real < 0).all() and (pc.imag > 0).all()
    print("  stability OK")
    print(f"  delta_scale init: {model.pole_head.delta_scale.item():.3f}")

    target = torch.randn_like(S)
    loss = ((S.real - target.real)**2 + (S.imag - target.imag)**2).mean()
    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_tot = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  grad flow: {n_grad}/{n_tot}")
    assert n_grad == n_tot
    print("Self-test passed.")


if __name__ == "__main__":
    _self_test()