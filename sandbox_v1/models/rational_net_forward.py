"""
rational_forward.py — physics-constrained rational forward model.

Architecture:
  Inputs:   X_local (8), X_global (6), X_context (7)         [z-scored]
  Encoder:  Fourier features on continuous geometry → MLP → latent h
  Heads:    pole head → (n_real + 2*n_cmplx) real numbers parameterizing
                          stable poles via softplus
            residue head → complex residues for each pole, per S-element
            D head     → constant 4x4 complex matrix
  Layer:    S(s) = Σ R_n / (s - p_n) + D   evaluated at s = j·2π·f
  Output:   S_pred ∈ C^(F, 4, 4)

Design notes:
  * Poles are predicted PER SAMPLE. Stability (Re < 0) is enforced
    structurally by Re(p) = -softplus(a). Upper-half-plane (Im > 0) for
    complex poles by Im(p) = softplus(b). Conjugates added at evaluation.
  * Internal frequency normalization: s_hat = s / (2π·f_scale), f_scale = 100 GHz.
    Without this, parameter scales hit 10¹¹+ and Adam stalls. See Day-1 notes.
  * complex128 in the rational layer. complex64 loses precision in the s-p
    subtraction near resonances. MLP stays float32; cast at the boundary.
  * Pole head bias is initialized so initial poles match the VFIT centroids
    from pole_basis.pt. Warm start, not a constraint — the MLP can drift away.
  * Fourier features applied to X_local only (continuous geometry). X_global
    is mostly integer counts; we let the MLP use them directly.

References:
  Tancik et al. NeurIPS 2020 (Fourier features for low-dim inputs)
  Feng et al. IEEE TMTT 2017 (NN pole-residue formulation)
  Chen-Zhang-Feng et al. IEEE TMTT 2023 (per-sample poles under geom variation)
  Gustavsen-Semlyen IEEE TPD 1999 (rational form, stability constraint)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Fourier feature encoder
# ----------------------------------------------------------------------
class FourierFeatures(nn.Module):
    """
    Random Fourier features (Tancik 2020). Each input dimension is mapped to
    2*n_features outputs: [sin(2π·B·x), cos(2π·B·x)] where B is a fixed random
    matrix sampled from N(0, sigma^2). Sigma controls the bandwidth — higher
    sigma captures higher-frequency dependencies but can hurt generalization.
    """

    def __init__(self, in_dim: int, n_features: int = 32, sigma: float = 2.0, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        B = torch.randn(in_dim, n_features, generator=g) * sigma
        self.register_buffer("B", B)
        self.out_dim = 2 * n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_dim) → (batch, 2*n_features)
        proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# ----------------------------------------------------------------------
# Stable-pole parameterization
# ----------------------------------------------------------------------
def softplus_inverse(y: torch.Tensor) -> torch.Tensor:
    """
    Inverse of softplus(x) = log(1 + exp(x)).  Solves x such that softplus(x) = y.
    Numerically stable: x = y + log(1 - exp(-y)).
    Used to set the pole-head bias so initial outputs match a target value.
    """
    return y + torch.log1p(-torch.exp(-y))


class PoleHead(nn.Module):
    """
    Predicts (a_real, a_cmplx_re, a_cmplx_im) parameters from latent h.
    Output poles:
        p_real_n  = -softplus(a_real_n)            (negative real)
        p_cmplx_n = -softplus(a_cmplx_re_n) +
                     1j·softplus(a_cmplx_im_n)      (Re<0, Im>0)

    Bias of the final linear layer is initialized so that for h=0 the predicted
    poles match the VFIT centroids. Weights are kept tiny so the warm start
    dominates at the beginning of training.
    """

    def __init__(self, latent_dim: int, n_real: int, n_cmplx: int,
                 init_poles_real: torch.Tensor, init_poles_cmplx: torch.Tensor,
                 f_scale_rad: float, hidden: int = 128, init_weight_scale: float = 1e-3):
        super().__init__()
        self.n_real = n_real
        self.n_cmplx = n_cmplx
        self.f_scale_rad = f_scale_rad  # 2π · f_scale_hz

        self.body = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.head = nn.Linear(hidden, n_real + 2 * n_cmplx)

        # Initialize head weights tiny so bias dominates at init
        nn.init.normal_(self.head.weight, std=init_weight_scale)

        # Compute biases that produce the target initial poles
        # init_poles_real: (n_real,) real, negative
        # init_poles_cmplx: (n_cmplx,) complex, Re<0, Im>0
        # In NORMALIZED units (divide by f_scale_rad)
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

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (poles_real_norm, poles_cmplx_norm) — both in NORMALIZED frequency units.
            poles_real_norm:  (batch, n_real)        real, negative
            poles_cmplx_norm: (batch, n_cmplx)       complex, Re<0, Im>0
        Caller multiplies by f_scale_rad to recover SI units (rad/s).
        """
        raw = self.head(self.body(h))                  # (B, n_real + 2*n_cmplx)
        a_real = raw[:, :self.n_real]
        a_cre = raw[:, self.n_real:self.n_real + self.n_cmplx]
        a_cim = raw[:, self.n_real + self.n_cmplx:]

        # softplus floor to avoid pole exactly at origin (would NaN in s-p)
        # 1e-4 in normalized units = ~6e7 rad/s = 10 MHz. Below freq range, harmless.
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
    """
    Predicts complex residue tensors:
        R_real:  (B, n_real, 4, 4)   real-pole residues (real-valued)
        R_cmplx: (B, n_cmplx, 4, 4)  complex-pole residues (complex)

    Real-pole residues are real because the pole is real and the time-domain
    response must be real-valued.  Complex-pole residues are complex; their
    conjugates pair with conjugate poles to keep the output real.
    """

    def __init__(self, latent_dim: int, n_real: int, n_cmplx: int, hidden: int = 256):
        super().__init__()
        self.n_real = n_real
        self.n_cmplx = n_cmplx
        # 16 real numbers per real pole  +  32 real numbers (16 re + 16 im) per complex pole
        out_dim = n_real * 16 + n_cmplx * 32
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = h.shape[0]
        out = self.net(h)
        r_real_flat = out[:, :self.n_real * 16]
        r_cmplx_flat = out[:, self.n_real * 16:]
        R_real = r_real_flat.view(B, self.n_real, 4, 4).to(torch.float64)
        r_cmplx_split = r_cmplx_flat.view(B, self.n_cmplx, 2, 4, 4)  # last dim 2 = [re, im]
        R_cmplx = torch.complex(r_cmplx_split[:, :, 0].to(torch.float64),
                                r_cmplx_split[:, :, 1].to(torch.float64))
        return R_real, R_cmplx


class DHead(nn.Module):
    """Predicts the constant 4x4 complex D matrix."""

    def __init__(self, latent_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 32),  # 16 re + 16 im
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        out = self.net(h).view(B, 2, 4, 4)
        return torch.complex(out[:, 0].to(torch.float64), out[:, 1].to(torch.float64))


# ----------------------------------------------------------------------
# Rational layer (evaluation only — poles/residues come from heads)
# ----------------------------------------------------------------------
def evaluate_rational(poles_real: torch.Tensor,    # (B, n_real)   complex (or real cast)
                       poles_cmplx: torch.Tensor,   # (B, n_cmplx)  complex (Im > 0)
                       R_real: torch.Tensor,         # (B, n_real, 4, 4) complex
                       R_cmplx: torch.Tensor,        # (B, n_cmplx, 4, 4) complex
                       D: torch.Tensor,              # (B, 4, 4) complex
                       s_norm: torch.Tensor,         # (F,) complex (j·omega_norm)
                       ) -> torch.Tensor:
    """
    Evaluate S(s) = sum_real R_real_n / (s - p_real_n)
                  + sum_cmplx R_cmplx_n / (s - p_cmplx_n)
                  + sum_cmplx conj(R_cmplx_n) / (s - conj(p_cmplx_n))
                  + D

    All inputs are in NORMALIZED frequency. Poles must have Re<0 (caller's job).
    Returns S: (B, F, 4, 4) complex.
    """
    # Cast poles_real to complex for uniform handling
    if not torch.is_complex(poles_real):
        poles_real_c = torch.complex(poles_real, torch.zeros_like(poles_real))
    else:
        poles_real_c = poles_real

    # s - p:   s_norm has shape (F,), poles have shape (B, n_p)
    # We want denom of shape (B, n_p, F) so we can broadcast against residues (B, n_p, 4, 4)
    s_expand = s_norm.view(1, 1, -1)                                    # (1, 1, F)

    # Real poles
    denom_r = s_expand - poles_real_c.unsqueeze(-1)                     # (B, n_real, F)
    term_r = R_real.unsqueeze(-1) / denom_r.unsqueeze(-2).unsqueeze(-2) # (B, n_real, 4, 4, F)
    sum_r = term_r.sum(dim=1)                                            # (B, 4, 4, F)

    # Complex poles + conjugates
    denom_c = s_expand - poles_cmplx.unsqueeze(-1)                      # (B, n_cmplx, F)
    denom_cc = s_expand - poles_cmplx.conj().unsqueeze(-1)              # (B, n_cmplx, F)
    term_c = R_cmplx.unsqueeze(-1) / denom_c.unsqueeze(-2).unsqueeze(-2)
    term_cc = R_cmplx.conj().unsqueeze(-1) / denom_cc.unsqueeze(-2).unsqueeze(-2)
    sum_c = term_c.sum(dim=1) + term_cc.sum(dim=1)                      # (B, 4, 4, F)

    S = sum_r + sum_c + D.unsqueeze(-1)                                 # (B, 4, 4, F)
    return S.permute(0, 3, 1, 2)                                         # (B, F, 4, 4)


# ----------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------
class RationalForwardModel(nn.Module):
    """
    Full forward model: (X_local, X_global, X_context) → S(f) ∈ C^(F, 4, 4).
    """

    def __init__(self,
                 freqs_hz: torch.Tensor,
                 init_poles_real: torch.Tensor,
                 init_poles_cmplx: torch.Tensor,
                 d_local: int = 8,
                 d_global: int = 6,
                 d_context: int = 7,
                 fourier_features: int = 32,
                 fourier_sigma: float = 2.0,
                 latent_dim: int = 256,
                 encoder_hidden: int = 256,
                 f_scale_hz: float = 100e9):
        super().__init__()
        self.n_real = len(init_poles_real)
        self.n_cmplx = len(init_poles_cmplx)
        self.f_scale_rad = 2 * math.pi * f_scale_hz

        # Normalized angular frequency for evaluation: s = j · 2π·f / f_scale_rad
        omega_norm = 2 * math.pi * freqs_hz / self.f_scale_rad
        s_norm = torch.complex(torch.zeros_like(omega_norm), omega_norm).to(torch.complex128)
        self.register_buffer("s_norm", s_norm)

        # Fourier features on continuous geometry (X_local)
        self.fourier = FourierFeatures(d_local, n_features=fourier_features, sigma=fourier_sigma)
        encoder_in = self.fourier.out_dim + d_global + d_context

        # Encoder MLP → shared latent h
        self.encoder = nn.Sequential(
            nn.Linear(encoder_in, encoder_hidden),
            nn.SiLU(),
            nn.Linear(encoder_hidden, encoder_hidden),
            nn.SiLU(),
            nn.Linear(encoder_hidden, latent_dim),
            nn.SiLU(),
        )

        self.pole_head = PoleHead(
            latent_dim, self.n_real, self.n_cmplx,
            init_poles_real, init_poles_cmplx,
            self.f_scale_rad,
        )
        self.residue_head = ResidueHead(latent_dim, self.n_real, self.n_cmplx)
        self.d_head = DHead(latent_dim)

    def forward(self, x_local: torch.Tensor, x_global: torch.Tensor, x_context: torch.Tensor) -> torch.Tensor:
        """
        x_local, x_global, x_context are z-scored (statistics in the .pt).
        Returns S_pred: (B, F, 4, 4) complex128.
        """
        ff = self.fourier(x_local)
        h = self.encoder(torch.cat([ff, x_global, x_context], dim=-1))

        poles_real_norm, poles_cmplx_norm = self.pole_head(h)
        R_real, R_cmplx = self.residue_head(h)
        D = self.d_head(h)

        S = evaluate_rational(poles_real_norm, poles_cmplx_norm, R_real, R_cmplx, D, self.s_norm)
        return S

    def get_poles_si_units(self, x_local: torch.Tensor, x_global: torch.Tensor, x_context: torch.Tensor):
        """Diagnostic: return predicted poles in SI units (rad/s)."""
        ff = self.fourier(x_local)
        h = self.encoder(torch.cat([ff, x_global, x_context], dim=-1))
        pr, pc = self.pole_head(h)
        return pr * self.f_scale_rad, pc * self.f_scale_rad


# ----------------------------------------------------------------------
# Self-test: build the model, run a forward pass, verify shapes,
# verify stability of initial poles, verify gradients flow.
# ----------------------------------------------------------------------
def _self_test():
    """Run a basic sanity check. Imported by callers; also runs as __main__."""
    torch.manual_seed(0)

    # Fake pole basis (4 real + 6 complex)
    init_poles_real = torch.tensor([-1e10, -1e11, -1e12, -1e13], dtype=torch.float64)
    init_poles_cmplx = torch.tensor(
        [-1e10 + 1j * 5e10, -2e10 + 1j * 1.5e11, -3e10 + 1j * 3e11,
         -2e10 + 1j * 5e11, -1e10 + 1j * 7e11, -5e10 + 1j * 9e11],
        dtype=torch.complex128,
    )

    freqs = torch.linspace(0.25e9, 100e9, 401, dtype=torch.float64)
    model = RationalForwardModel(freqs, init_poles_real, init_poles_cmplx,
                                  d_local=8, d_global=6, d_context=7)

    B = 4
    x_local = torch.randn(B, 8)
    x_global = torch.randn(B, 6)
    x_context = torch.randn(B, 7)

    S = model(x_local, x_global, x_context)
    print(f"  forward pass OK, S shape: {tuple(S.shape)}  dtype: {S.dtype}")
    assert S.shape == (B, 401, 4, 4)
    assert S.dtype == torch.complex128

    pr_si, pc_si = model.get_poles_si_units(x_local, x_global, x_context)
    print(f"  poles (sample 0):")
    print(f"    real:    {pr_si[0].tolist()}")
    print(f"    complex: {pc_si[0].tolist()}")
    assert (pr_si.real < 0).all(), "real poles not stable"
    assert (pc_si.real < 0).all(), "complex pole real parts not stable"
    assert (pc_si.imag > 0).all(), "complex pole imag parts not in UHP"
    print("  stability OK")

    # At init, poles should be close to the basis centroids
    # (only "close" because the head MLP body is non-zero with random init;
    #  but for a non-trained MLP receiving zero-mean inputs the body output is small)
    print(f"\n  init-pole alignment with basis (sample 0 vs init):")
    print(f"    real:    init={init_poles_real.tolist()}")
    print(f"             pred={pr_si[0].tolist()}")
    # Numbers won't match exactly because the body MLP doesn't produce exactly 0.
    # But they should be in the same order of magnitude.

    # Gradient check
    target = torch.randn_like(S)
    loss = ((S.real - target.real) ** 2 + (S.imag - target.imag) ** 2).mean()
    loss.backward()
    n_params_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_params_total = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"\n  grad flow: {n_params_with_grad}/{n_params_total} parameters got non-zero grads")
    assert n_params_with_grad == n_params_total, "some parameters got no gradient"

    print("\nSelf-test passed.")


if __name__ == "__main__":
    _self_test()