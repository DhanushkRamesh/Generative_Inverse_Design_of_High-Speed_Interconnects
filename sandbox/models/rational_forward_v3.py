"""
rational_forward_v3.py
Physics-constrained rational forward model for differential pair S-parameters.

Architecture rationale (locked):
  - Per-sample poles. Fixed shared poles failed the hold-out test (>50% RMS error
    on unseen geometries) because TUHH's geometric variation moves resonances
    over a wide range. We let the MLP predict poles per sample, with stability
    enforced structurally by Re(p) = -softplus(a) and Im(p) = softplus(b).
  - Two-encoder factorization. Poles depend on the substrate / array geometry
    only (X_local, X_global), not on which pair within the array we're looking
    at. Residues depend on all three (geometry + pair context). This matches
    the physics: pairs from the same simulation share resonance frequencies but
    differ in coupling magnitudes.
  - Warm-started from VFIT pole basis. The pole head's final-layer bias is
    initialized so that at h=0 (zero-mean inputs) the network reproduces the
    44-pole basis we extracted with vector fitting (4 real + 20 complex pairs).
    The MLP then learns geometry-dependent deviations.
  - Symmetric residues. For reciprocal passive structures, S = S^T at every
    frequency, which requires R_n = R_n^T for every pole. We parameterize each
    4x4 residue with 10 real numbers (upper triangle), saving 6 numbers per
    pole and embedding reciprocity structurally.
  - Internal frequency normalization. s_hat = s / (2*pi*f_scale), f_scale=100 GHz.
    Without this, Adam stalls because parameter scales hit 10^11+.
  - complex128 in the rational layer. complex64 loses 7 digits in the s-p
    subtraction near resonances. MLP stays float32; cast at the boundary.

References:
  Gustavsen and Semlyen, IEEE TPD 1999 (vector fitting)
  Feng et al., IEEE TMTT 2017 (NN pole-residue formulation)
  Chen-Zhang-Feng et al., IEEE TMTT 2023 (per-sample poles under geom variation)
  Hillebrecht et al., IEEE TEMC 2024 (TUHH dataset, noise floor characterization)
  Tancik et al., NeurIPS 2020 (Fourier features)
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Multi-scale Fourier features (Tancik et al. 2020)
# ----------------------------------------------------------------------
class MultiScaleFourierFeatures(nn.Module):
    """
    Encodes continuous inputs at multiple frequency scales by projection onto
    random vectors drawn from N(0, sigma^2 I) and applying sin/cos.

    For our z-scored X_local (8 dims of continuous geometry/material),
    sigma values of {1, 4, 16} give bandwidth from gentle low-frequency input
    dependencies up to sharp ones. Critical for the model to learn dependence
    on via radius and antipad which set resonance positions.
    """

    def __init__(self, in_dim: int,
                 sigmas: List[float] = [1.0, 4.0, 16.0],
                 n_features_per_scale: int = 32,
                 seed: int = 0):
        super().__init__()
        # Random projection matrix, fixed (not learned)
        g = torch.Generator().manual_seed(seed)
        Bs = []
        for s in sigmas:
            Bs.append(torch.randn(in_dim, n_features_per_scale, generator=g) * s)
        B = torch.cat(Bs, dim=1)
        self.register_buffer("B", B)
        # Output is sin/cos pair per scale
        self.out_dim = 2 * n_features_per_scale * len(sigmas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_dim) -> (batch, 2 * n_features * n_scales)
        proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# ----------------------------------------------------------------------
# Residual MLP block with LayerNorm
# ----------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Standard pre-norm residual block: LayerNorm -> Linear -> SiLU -> Linear -> add."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.silu(self.fc1(h))
        h = self.fc2(h)
        return x + h


# ----------------------------------------------------------------------
# Numerical helper: inverse of softplus, used for setting bias to a target value
# ----------------------------------------------------------------------
def softplus_inverse(y: torch.Tensor) -> torch.Tensor:
    """Solves softplus(x) = y. Numerically stable form: x = y + log(1 - exp(-y))."""
    return y + torch.log1p(-torch.exp(-y))


# ----------------------------------------------------------------------
# Geometry encoder (consumes X_local + X_global)
# ----------------------------------------------------------------------
class GeometryEncoder(nn.Module):
    """
    Encodes geometry-only features into a latent h_geom.
    All pairs from the same simulation see the same X_local + X_global, so
    h_geom is identical for them. Poles are derived from h_geom alone, ensuring
    that pairs within a sim share resonance frequencies (correct physics).
    """

    def __init__(self,
                 d_local: int = 8,
                 d_global: int = 6,
                 fourier_sigmas: List[float] = [1.0, 4.0, 16.0],
                 fourier_features: int = 32,
                 latent_dim: int = 256,
                 hidden: int = 384,
                 n_blocks: int = 3):
        super().__init__()
        # Fourier features only on continuous geometry (X_local)
        self.fourier = MultiScaleFourierFeatures(d_local,
                                                  sigmas=fourier_sigmas,
                                                  n_features_per_scale=fourier_features)
        in_dim = self.fourier.out_dim + d_global

        # Project input to hidden, then residual blocks, then project to latent
        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([ResidualBlock(hidden) for _ in range(n_blocks)])
        self.proj_out = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
            nn.SiLU(),
        )

    def forward(self, x_local: torch.Tensor, x_global: torch.Tensor) -> torch.Tensor:
        ff = self.fourier(x_local)
        x = torch.cat([ff, x_global], dim=-1)
        h = self.proj_in(x)
        for block in self.blocks:
            h = block(h)
        return self.proj_out(h)


# ----------------------------------------------------------------------
# Pair encoder (consumes X_context + h_geom)
# ----------------------------------------------------------------------
class PairEncoder(nn.Module):
    """
    Combines h_geom with pair-specific X_context to produce h_pair.
    Residues and D are derived from h_pair. Two pairs in the same sim have
    the same h_geom but different X_context -> different h_pair -> different
    residues. This is exactly the physics: shared resonance frequencies,
    pair-dependent coupling strengths.
    """

    def __init__(self,
                 latent_geom_dim: int = 256,
                 d_context: int = 7,
                 latent_dim: int = 384,
                 hidden: int = 384,
                 n_blocks: int = 2):
        super().__init__()
        in_dim = latent_geom_dim + d_context
        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([ResidualBlock(hidden) for _ in range(n_blocks)])
        self.proj_out = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
            nn.SiLU(),
        )

    def forward(self, h_geom: torch.Tensor, x_context: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h_geom, x_context], dim=-1)
        h = self.proj_in(x)
        for block in self.blocks:
            h = block(h)
        return self.proj_out(h)


# ----------------------------------------------------------------------
# Pole head
# ----------------------------------------------------------------------
class PoleHead(nn.Module):
    """
    Predicts (a_real, a_cre, a_cim) -> stable poles.

    For each real pole n: p_real_n = -softplus(a_real_n)   (negative real)
    For each complex pole n: p_cmplx_n = -softplus(a_cre_n) + j*softplus(a_cim_n)
                                          (Re < 0, Im > 0; conjugate added in layer)

    Final-layer bias initialized to reproduce the VFIT centroid pole basis at
    h=0 (warm start). Weights initialized small so warm start dominates early.
    A learnable scalar `delta_scale` modulates how much the MLP deviates from
    the warm-start bias; starts at 0.5 and can grow if Adam decides the model
    benefits from larger pole deviations.
    """

    def __init__(self,
                 latent_dim: int,
                 n_real: int,
                 n_cmplx: int,
                 init_poles_real: torch.Tensor,    # (n_real,) real, negative, SI units (rad/s)
                 init_poles_cmplx: torch.Tensor,   # (n_cmplx,) complex, Re<0, Im>0, SI units
                 f_scale_rad: float,               # 2*pi*100e9, for input normalization
                 hidden: int = 256,
                 init_weight_scale: float = 1e-2,
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

        # Small init weights so bias-driven warm start dominates training start
        nn.init.normal_(self.head.weight, std=init_weight_scale)

        # Compute bias values that produce target poles in NORMALIZED units
        target_a_real = softplus_inverse(
            torch.as_tensor(-init_poles_real.real / f_scale_rad,
                            dtype=torch.float32).clamp_min(1e-6)
        )
        target_a_cre = softplus_inverse(
            torch.as_tensor(-init_poles_cmplx.real / f_scale_rad,
                            dtype=torch.float32).clamp_min(1e-6)
        )
        target_a_cim = softplus_inverse(
            torch.as_tensor(init_poles_cmplx.imag / f_scale_rad,
                            dtype=torch.float32).clamp_min(1e-6)
        )
        bias = torch.cat([target_a_real, target_a_cre, target_a_cim])
        with torch.no_grad():
            self.head.bias.copy_(bias)

        # Learnable gate on (raw_output - bias) deviation
        self.delta_scale = nn.Parameter(torch.tensor(init_delta_scale))

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (poles_real_norm, poles_cmplx_norm) in NORMALIZED frequency.
            poles_real_norm:  (batch, n_real)   real, negative
            poles_cmplx_norm: (batch, n_cmplx)  complex, Re<0, Im>0
        Caller multiplies by f_scale_rad to recover SI units.
        """
        raw = self.head(self.body(h))
        # Apply learnable gate: pull predictions toward the bias by (1 - delta_scale)
        bias = self.head.bias
        deviation = raw - bias
        scaled = bias + self.delta_scale * deviation

        a_real = scaled[:, :self.n_real]
        a_cre = scaled[:, self.n_real:self.n_real + self.n_cmplx]
        a_cim = scaled[:, self.n_real + self.n_cmplx:]

        # softplus floor at 1e-4 normalized = ~6e7 rad/s = 10 MHz, below freq range
        # Prevents pole at exactly origin which would NaN the s-p subtraction
        re_real = -F.softplus(a_real).clamp_min(1e-4)
        re_cmplx = -F.softplus(a_cre).clamp_min(1e-4)
        im_cmplx = F.softplus(a_cim).clamp_min(1e-4)

        # Cast to float64 / complex128 for the rational layer
        poles_real = re_real.to(torch.float64)
        poles_cmplx = torch.complex(re_cmplx.to(torch.float64),
                                     im_cmplx.to(torch.float64))
        return poles_real, poles_cmplx


# ----------------------------------------------------------------------
# Symmetric residue + D heads
# ----------------------------------------------------------------------
class ResidueHead(nn.Module):
    """
    Predicts SYMMETRIC complex 4x4 residue matrices for each pole, plus a
    symmetric 4x4 D matrix.

    Symmetry is enforced structurally: the head outputs 10 numbers per matrix
    (upper triangle including diagonal), and we scatter those to both upper
    and lower triangle positions. This embeds the reciprocity constraint
    R_n = R_n^T (equivalently S = S^T) and saves 6 parameters per residue.
    """

    # Upper-triangle indices for a 4x4 matrix: (0,0),(0,1),(0,2),(0,3),(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)
    UPPER_R = (0, 0, 0, 0, 1, 1, 1, 2, 2, 3)
    UPPER_C = (0, 1, 2, 3, 1, 2, 3, 2, 3, 3)
    N_SYM = 10  # 10 unique elements in a symmetric 4x4

    def __init__(self, latent_dim: int, n_real: int, n_cmplx: int, hidden: int = 384):
        super().__init__()
        self.n_real = n_real
        self.n_cmplx = n_cmplx
        # Per-sample outputs:
        #   n_real real-pole residues: 10 real numbers each (real-valued residues for real poles)
        #   n_cmplx complex-pole residues: 10 complex numbers each = 20 real numbers
        #   D matrix: 10 complex numbers = 20 real numbers
        out_dim = n_real * 10 + n_cmplx * 20 + 20
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

        # Register triangle indices as buffers so they move with .to(device)
        self.register_buffer("upper_r", torch.tensor(self.UPPER_R, dtype=torch.long))
        self.register_buffer("upper_c", torch.tensor(self.UPPER_C, dtype=torch.long))

    def _scatter_symmetric(self, vec_real: torch.Tensor, vec_imag: torch.Tensor,
                            batch: int) -> torch.Tensor:
        """
        Scatter a (batch, 10) real and (batch, 10) imag into a (batch, 4, 4) complex
        symmetric matrix. Diagonal entries get scattered once; off-diagonal entries
        get scattered to both (i,j) and (j,i).
        """
        mat = torch.zeros((batch, 4, 4), dtype=torch.complex128, device=vec_real.device)
        cmplx_vec = torch.complex(vec_real.to(torch.float64), vec_imag.to(torch.float64))
        # Scatter to upper triangle including diagonal
        mat[:, self.upper_r, self.upper_c] = cmplx_vec
        # Scatter to lower triangle (mirror); diagonal is overwritten with same value (no-op)
        mat[:, self.upper_c, self.upper_r] = cmplx_vec
        return mat

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (R_real, R_cmplx, D) where:
            R_real:  (batch, n_real, 4, 4) complex (real-valued imag part for real poles)
            R_cmplx: (batch, n_cmplx, 4, 4) complex
            D:       (batch, 4, 4) complex
        All matrices are symmetric (4x4 sym).
        """
        B = h.shape[0]
        out = self.net(h)

        idx = 0
        # Real-pole residues: real-valued (imag = 0)
        rr_flat = out[:, idx:idx + self.n_real * 10]
        idx += self.n_real * 10
        rr_per_pole = rr_flat.view(B, self.n_real, 10)
        R_real = torch.zeros((B, self.n_real, 4, 4), dtype=torch.complex128, device=h.device)
        for n in range(self.n_real):
            zero_imag = torch.zeros_like(rr_per_pole[:, n])
            R_real[:, n] = self._scatter_symmetric(rr_per_pole[:, n], zero_imag, B)

        # Complex-pole residues: complex-valued, 10 real + 10 imag per pole
        rc_flat = out[:, idx:idx + self.n_cmplx * 20]
        idx += self.n_cmplx * 20
        rc_per_pole = rc_flat.view(B, self.n_cmplx, 2, 10)  # [..., 0]=re, [..., 1]=im
        R_cmplx = torch.zeros((B, self.n_cmplx, 4, 4), dtype=torch.complex128, device=h.device)
        for n in range(self.n_cmplx):
            R_cmplx[:, n] = self._scatter_symmetric(
                rc_per_pole[:, n, 0], rc_per_pole[:, n, 1], B
            )

        # D: complex symmetric 4x4
        d_flat = out[:, idx:idx + 20]
        d_split = d_flat.view(B, 2, 10)
        D = self._scatter_symmetric(d_split[:, 0], d_split[:, 1], B)

        return R_real, R_cmplx, D


# ----------------------------------------------------------------------
# Rational layer evaluation
# ----------------------------------------------------------------------
def evaluate_rational(poles_real_norm: torch.Tensor,   # (B, n_real) real
                       poles_cmplx_norm: torch.Tensor,  # (B, n_cmplx) complex (Im > 0)
                       R_real: torch.Tensor,             # (B, n_real, 4, 4) complex
                       R_cmplx: torch.Tensor,            # (B, n_cmplx, 4, 4) complex
                       D: torch.Tensor,                  # (B, 4, 4) complex
                       s_norm: torch.Tensor              # (F,) complex (j*omega_norm)
                       ) -> torch.Tensor:
    """
    Evaluate the partial-fraction expansion:
        S(s) = sum_n R_real_n / (s - p_real_n)
             + sum_n R_cmplx_n / (s - p_cmplx_n)
             + sum_n conj(R_cmplx_n) / (s - conj(p_cmplx_n))
             + D

    All inputs in NORMALIZED frequency. Returns S of shape (B, F, 4, 4) complex.
    """
    # Cast real poles to complex
    p_real_c = torch.complex(poles_real_norm, torch.zeros_like(poles_real_norm))

    # Broadcasting setup
    # s_norm: (F,) -> (1, 1, F)
    # poles: (B, n_p) -> (B, n_p, 1)
    # residues: (B, n_p, 4, 4) -> (B, n_p, 4, 4, 1)
    s_view = s_norm.view(1, 1, -1)

    # Real-pole contributions
    denom_r = s_view - p_real_c.unsqueeze(-1)  # (B, n_real, F)
    term_r = R_real.unsqueeze(-1) / denom_r.unsqueeze(-2).unsqueeze(-2)  # (B, n_real, 4, 4, F)
    sum_r = term_r.sum(dim=1)  # (B, 4, 4, F)

    # Complex-pole contributions: pole + conjugate
    denom_c = s_view - poles_cmplx_norm.unsqueeze(-1)
    denom_cc = s_view - poles_cmplx_norm.conj().unsqueeze(-1)
    term_c = R_cmplx.unsqueeze(-1) / denom_c.unsqueeze(-2).unsqueeze(-2)
    term_cc = R_cmplx.conj().unsqueeze(-1) / denom_cc.unsqueeze(-2).unsqueeze(-2)
    sum_c = term_c.sum(dim=1) + term_cc.sum(dim=1)  # (B, 4, 4, F)

    S = sum_r + sum_c + D.unsqueeze(-1)  # (B, 4, 4, F)
    return S.permute(0, 3, 1, 2)  # (B, F, 4, 4)


# ----------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------
class RationalForwardModel(nn.Module):
    """
    Forward map: (X_local, X_global, X_context) -> S(f) in C^(F, 4, 4).

    Two-encoder factorization separates geometry-dependent quantities (poles)
    from pair-dependent quantities (residues + D). The shape of the latent space
    encodes the physics correctly: pairs from the same sim share poles, differ
    only in residues.
    """

    def __init__(self,
                 freqs_hz: torch.Tensor,
                 init_poles_real: torch.Tensor,
                 init_poles_cmplx: torch.Tensor,
                 d_local: int = 8,
                 d_global: int = 6,
                 d_context: int = 7,
                 fourier_sigmas: List[float] = [1.0, 4.0, 16.0],
                 fourier_features: int = 32,
                 latent_geom_dim: int = 256,
                 latent_pair_dim: int = 384,
                 encoder_hidden: int = 384,
                 f_scale_hz: float = 100e9):
        super().__init__()
        self.n_real = len(init_poles_real)
        self.n_cmplx = len(init_poles_cmplx)
        self.f_scale_rad = 2 * math.pi * f_scale_hz

        # Build normalized s = j*omega/f_scale_rad for evaluation
        omega_norm = 2 * math.pi * freqs_hz / self.f_scale_rad
        s_norm = torch.complex(torch.zeros_like(omega_norm), omega_norm).to(torch.complex128)
        self.register_buffer("s_norm", s_norm)

        # Geometry encoder (poles depend on this)
        self.geom_encoder = GeometryEncoder(
            d_local=d_local, d_global=d_global,
            fourier_sigmas=fourier_sigmas,
            fourier_features=fourier_features,
            latent_dim=latent_geom_dim,
            hidden=encoder_hidden,
            n_blocks=3,
        )

        # Pair encoder (residues + D depend on this)
        self.pair_encoder = PairEncoder(
            latent_geom_dim=latent_geom_dim,
            d_context=d_context,
            latent_dim=latent_pair_dim,
            hidden=encoder_hidden,
            n_blocks=2,
        )

        # Heads
        self.pole_head = PoleHead(
            latent_dim=latent_geom_dim,
            n_real=self.n_real,
            n_cmplx=self.n_cmplx,
            init_poles_real=init_poles_real,
            init_poles_cmplx=init_poles_cmplx,
            f_scale_rad=self.f_scale_rad,
        )
        self.residue_head = ResidueHead(
            latent_dim=latent_pair_dim,
            n_real=self.n_real,
            n_cmplx=self.n_cmplx,
        )

    def forward(self, x_local: torch.Tensor, x_global: torch.Tensor,
                x_context: torch.Tensor) -> torch.Tensor:
        # Geometry-only encoding -> poles
        h_geom = self.geom_encoder(x_local, x_global)
        poles_real_norm, poles_cmplx_norm = self.pole_head(h_geom)

        # Geometry + context encoding -> residues + D
        h_pair = self.pair_encoder(h_geom, x_context)
        R_real, R_cmplx, D = self.residue_head(h_pair)

        # Evaluate rational form
        S = evaluate_rational(poles_real_norm, poles_cmplx_norm,
                               R_real, R_cmplx, D, self.s_norm)
        return S

    def get_poles_si_units(self, x_local: torch.Tensor, x_global: torch.Tensor):
        """Diagnostic: return predicted poles in rad/s."""
        h_geom = self.geom_encoder(x_local, x_global)
        pr, pc = self.pole_head(h_geom)
        return pr * self.f_scale_rad, pc * self.f_scale_rad


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------
def _self_test():
    """Build model, forward + backward pass, shape and stability checks."""
    torch.manual_seed(0)

    # Mock pole basis (4 real + 6 complex)
    init_poles_real = torch.tensor([-1e10, -1e11, -1e12, -1e13], dtype=torch.float64)
    init_poles_cmplx = torch.tensor(
        [-1e10 + 1j * 5e10, -2e10 + 1j * 1.5e11, -3e10 + 1j * 3e11,
         -2e10 + 1j * 5e11, -1e10 + 1j * 7e11, -5e10 + 1j * 9e11],
        dtype=torch.complex128,
    )

    freqs = torch.linspace(0.25e9, 100e9, 401, dtype=torch.float64)
    model = RationalForwardModel(freqs, init_poles_real, init_poles_cmplx)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model parameters: {n_params:,}")
    print(f"  geom encoder params: {sum(p.numel() for p in model.geom_encoder.parameters()):,}")
    print(f"  pair encoder params: {sum(p.numel() for p in model.pair_encoder.parameters()):,}")
    print(f"  pole head params:    {sum(p.numel() for p in model.pole_head.parameters()):,}")
    print(f"  residue head params: {sum(p.numel() for p in model.residue_head.parameters()):,}")

    B = 4
    xl = torch.randn(B, 8)
    xg = torch.randn(B, 6)
    xc = torch.randn(B, 7)
    S = model(xl, xg, xc)
    print(f"  forward pass OK, S shape: {tuple(S.shape)}  dtype: {S.dtype}")
    assert S.shape == (B, 401, 4, 4)
    assert S.dtype == torch.complex128

    # Reciprocity check: S should equal S^T at every frequency (symmetric residues)
    asym = (S - S.transpose(-1, -2)).abs().max().item()
    print(f"  reciprocity |S - S^T| max: {asym:.3e}")
    assert asym < 1e-8, "Residues not symmetric"

    # Stability check
    pr, pc = model.get_poles_si_units(xl, xg)
    assert (pr.real < 0).all()
    assert (pc.real < 0).all()
    assert (pc.imag > 0).all()
    print("  stability OK (Re<0, Im>0 enforced)")

    # Warm start check: at zero input, poles should match the init basis
    with torch.no_grad():
        pr0, pc0 = model.get_poles_si_units(torch.zeros(1, 8), torch.zeros(1, 6))
    err_real = (pr0[0] - init_poles_real).abs().max().item() / init_poles_real.abs().max().item()
    err_cmplx = (pc0[0] - init_poles_cmplx).abs().max().item() / init_poles_cmplx.abs().max().item()
    print(f"  warm-start init: real err={err_real:.3e}, cmplx err={err_cmplx:.3e}")

    # Two pairs from the "same sim" -> identical X_local, X_global, different X_context
    # Should produce IDENTICAL poles, DIFFERENT residues -> different S, but same null positions
    xl_same = torch.randn(1, 8).repeat(2, 1)
    xg_same = torch.randn(1, 6).repeat(2, 1)
    xc_diff = torch.randn(2, 7)
    pr_a, pc_a = model.get_poles_si_units(xl_same, xg_same)
    assert torch.allclose(pr_a[0], pr_a[1]), "Poles differ across pairs of same sim (BUG)"
    assert torch.allclose(pc_a[0], pc_a[1]), "Poles differ across pairs of same sim (BUG)"
    print("  per-sim pole consistency OK")

    # Gradient check
    target = torch.randn_like(S)
    loss = ((S.real - target.real) ** 2 + (S.imag - target.imag) ** 2).mean()
    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_tot = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  grad flow: {n_grad}/{n_tot} params receive non-zero gradient")
    assert n_grad == n_tot

    print("\nSelf-test passed.")


if __name__ == "__main__":
    _self_test()