"""
rational_forward_v4.py
Parallel-decomposition rational forward model for differential pair S-parameters.

Theoretical basis (locked):
  - Single high-order rational models suffer from "high-sensitivity issue":
    small errors in coefficients produce large errors in output. With wide
    geometric variation this stalls training. See:
      Zhao et al., Micromachines 11(7) 696, 2020 (decomposition technique)
      Zhang et al., IEEE TMTT 68(12) 5288, 2020 (parallel decomposition)
  - Fix: decompose into M parallel sub-models, each with low-order transfer
    function. Sum their outputs to recover full response. Sensitivity scales
    with order N within each sub-model, not the total order M*N.

Architecture (per sub-model):
  Each sub-model has its own pole head (n_real_per_sub real + n_cmplx_per_sub
  complex poles) and its own residue head. They SHARE the geometry encoder
  and pair encoder (cheap parameter sharing where the geometry-dependence is
  consistent across sub-models), but diverge at the heads.

  S_total(s, x) = sum_{i=1..M} S_i(s, x)

  where each S_i(s, x) is a low-order rational function with its own
  per-sample poles, warm-started from a different slice of the VFIT basis.

Other choices kept from v3:
  - Two-encoder factorization (poles depend on geometry only, residues on
    geometry + pair context)
  - Per-sample stable poles via softplus
  - Symmetric residue parameterization (S = S^T for reciprocal structures)
  - Internal frequency normalization to keep params O(1)
  - complex128 in the rational layer

Reference list for thesis:
  Gustavsen & Semlyen, IEEE TPD 1999 (vector fitting)
  Feng et al., IEEE TMTT 2017 (NN pole-residue formulation)
  Zhao et al., Micromachines 11(7) 696, 2020 (decomposition idea)
  Zhang et al., IEEE TMTT 68(12) 5288, 2020 (parallel decomposition)
  Hillebrecht et al., IEEE TEMC 2024 (TUHH dataset)
  Tancik et al., NeurIPS 2020 (Fourier features)
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Multi-scale Fourier features (Tancik 2020)
# Used only on continuous geometry inputs (X_local)
# ----------------------------------------------------------------------
class MultiScaleFourierFeatures(nn.Module):
    """Random Fourier projection at multiple bandwidths, sin/cos encoded."""

    def __init__(self, in_dim: int,
                 sigmas: List[float] = [1.0, 4.0, 16.0],
                 n_features_per_scale: int = 32,
                 seed: int = 0):
        super().__init__()
        # Fixed random projection matrix, not learned
        g = torch.Generator().manual_seed(seed)
        Bs = [torch.randn(in_dim, n_features_per_scale, generator=g) * s for s in sigmas]
        B = torch.cat(Bs, dim=1)
        self.register_buffer("B", B)
        self.out_dim = 2 * n_features_per_scale * len(sigmas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# ----------------------------------------------------------------------
# Residual MLP block with LayerNorm (pre-norm style)
# ----------------------------------------------------------------------
class ResidualBlock(nn.Module):
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


def softplus_inverse(y: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse of softplus(x) = log(1 + exp(x))."""
    return y + torch.log1p(-torch.exp(-y))


# ----------------------------------------------------------------------
# Geometry encoder: X_local + X_global -> h_geom
# Shared across all sub-models. Two pairs in the same sim get the same h_geom.
# ----------------------------------------------------------------------
class GeometryEncoder(nn.Module):
    def __init__(self,
                 d_local: int = 8,
                 d_global: int = 6,
                 fourier_sigmas: List[float] = [1.0, 4.0, 16.0],
                 fourier_features: int = 32,
                 latent_dim: int = 256,
                 hidden: int = 384,
                 n_blocks: int = 3):
        super().__init__()
        self.fourier = MultiScaleFourierFeatures(d_local,
                                                  sigmas=fourier_sigmas,
                                                  n_features_per_scale=fourier_features)
        in_dim = self.fourier.out_dim + d_global
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
# Pair encoder: h_geom + X_context -> h_pair
# Shared across all sub-models.
# ----------------------------------------------------------------------
class PairEncoder(nn.Module):
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
# Per-sub-model pole head
# Same softplus stability and warm-start logic as v3, but each sub-model
# gets its own subset of the VFIT basis as the initialization.
# ----------------------------------------------------------------------
class SubPoleHead(nn.Module):
    """Pole head for one sub-model. Predicts n_real + n_cmplx pole parameters."""

    def __init__(self,
                 latent_dim: int,
                 n_real: int,
                 n_cmplx: int,
                 init_poles_real: torch.Tensor,    # (n_real,) real, negative, SI units
                 init_poles_cmplx: torch.Tensor,   # (n_cmplx,) complex, Re<0, Im>0, SI
                 f_scale_rad: float,
                 hidden: int = 128,
                 init_weight_scale: float = 1e-2,
                 init_delta_scale: float = 0.5):
        super().__init__()
        self.n_real = n_real
        self.n_cmplx = n_cmplx
        self.f_scale_rad = f_scale_rad

        # Smaller body MLP per sub-model since they're parallel
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

        # Bias initialized so that at body output = 0, predicted poles match the
        # warm-start subset for this sub-model. Each sub-model gets a different
        # slice of the VFIT basis (typically by frequency band).
        if n_real > 0:
            target_a_real = softplus_inverse(
                torch.as_tensor(-init_poles_real.real / f_scale_rad,
                                dtype=torch.float32).clamp_min(1e-6)
            )
        else:
            target_a_real = torch.zeros(0)
        if n_cmplx > 0:
            target_a_cre = softplus_inverse(
                torch.as_tensor(-init_poles_cmplx.real / f_scale_rad,
                                dtype=torch.float32).clamp_min(1e-6)
            )
            target_a_cim = softplus_inverse(
                torch.as_tensor(init_poles_cmplx.imag / f_scale_rad,
                                dtype=torch.float32).clamp_min(1e-6)
            )
        else:
            target_a_cre = torch.zeros(0)
            target_a_cim = torch.zeros(0)

        bias = torch.cat([target_a_real, target_a_cre, target_a_cim])
        with torch.no_grad():
            self.head.bias.copy_(bias)

        # Learnable gate on (raw - bias) deviation, starts at 0.5
        self.delta_scale = nn.Parameter(torch.tensor(init_delta_scale))

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.head(self.body(h))
        bias = self.head.bias
        deviation = raw - bias
        scaled = bias + self.delta_scale * deviation

        a_real = scaled[:, :self.n_real]
        a_cre = scaled[:, self.n_real:self.n_real + self.n_cmplx]
        a_cim = scaled[:, self.n_real + self.n_cmplx:]

        # softplus floor at 1e-4 normalized = ~6e7 rad/s = 10 MHz, below freq range
        if self.n_real > 0:
            re_real = -F.softplus(a_real).clamp_min(1e-4)
            poles_real = re_real.to(torch.float64)
        else:
            poles_real = torch.zeros(h.shape[0], 0, dtype=torch.float64, device=h.device)

        if self.n_cmplx > 0:
            re_cmplx = -F.softplus(a_cre).clamp_min(1e-4)
            im_cmplx = F.softplus(a_cim).clamp_min(1e-4)
            poles_cmplx = torch.complex(re_cmplx.to(torch.float64),
                                         im_cmplx.to(torch.float64))
        else:
            poles_cmplx = torch.zeros(h.shape[0], 0,
                                       dtype=torch.complex128, device=h.device)

        return poles_real, poles_cmplx


# ----------------------------------------------------------------------
# Per-sub-model residue + D head, with symmetric residues
# ----------------------------------------------------------------------
class SubResidueHead(nn.Module):
    """
    Predicts symmetric 4x4 residues for each pole, plus a symmetric D.
    Symmetry enforced structurally: head outputs 10 numbers per matrix
    (upper triangle), scattered to both upper and lower triangle positions.
    """

    UPPER_R = (0, 0, 0, 0, 1, 1, 1, 2, 2, 3)
    UPPER_C = (0, 1, 2, 3, 1, 2, 3, 2, 3, 3)

    def __init__(self, latent_dim: int, n_real: int, n_cmplx: int,
                 hidden: int = 192, include_d: bool = True):
        super().__init__()
        self.n_real = n_real
        self.n_cmplx = n_cmplx
        self.include_d = include_d
        # 10 real numbers per real-pole residue (real-valued residues for real poles)
        # 20 real numbers (10 re + 10 im) per complex-pole residue
        # 20 real numbers for D (if this sub-model carries D)
        out_dim = n_real * 10 + n_cmplx * 20 + (20 if include_d else 0)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

        # Initialize residue weights very small so we don't blow up the rational
        # output at t=0 (the sum of M sub-models with random residues otherwise
        # blows up; we want a near-zero start that learns to add structure)
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

        # Triangle indices for scatter
        self.register_buffer("upper_r", torch.tensor(self.UPPER_R, dtype=torch.long))
        self.register_buffer("upper_c", torch.tensor(self.UPPER_C, dtype=torch.long))

    def _scatter_symmetric(self, vec_real: torch.Tensor, vec_imag: torch.Tensor,
                            batch: int) -> torch.Tensor:
        """Scatter (batch, 10) real and imag into (batch, 4, 4) complex symmetric."""
        mat = torch.zeros((batch, 4, 4), dtype=torch.complex128, device=vec_real.device)
        cmplx_vec = torch.complex(vec_real.to(torch.float64), vec_imag.to(torch.float64))
        # Upper triangle including diagonal
        mat[:, self.upper_r, self.upper_c] = cmplx_vec
        # Lower triangle by mirror; diagonal overwritten with same value
        mat[:, self.upper_c, self.upper_r] = cmplx_vec
        return mat

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            R_real: (batch, n_real, 4, 4) complex symmetric (real-valued for real poles)
            R_cmplx: (batch, n_cmplx, 4, 4) complex symmetric
            D: (batch, 4, 4) complex symmetric (zero matrix if include_d=False)
        """
        B = h.shape[0]
        out = self.net(h)

        idx = 0
        # Real-pole residues
        if self.n_real > 0:
            rr_flat = out[:, idx:idx + self.n_real * 10]
            idx += self.n_real * 10
            rr_per_pole = rr_flat.view(B, self.n_real, 10)
            R_real = torch.zeros((B, self.n_real, 4, 4),
                                  dtype=torch.complex128, device=h.device)
            for n in range(self.n_real):
                zero_imag = torch.zeros_like(rr_per_pole[:, n])
                R_real[:, n] = self._scatter_symmetric(rr_per_pole[:, n], zero_imag, B)
        else:
            R_real = torch.zeros((B, 0, 4, 4), dtype=torch.complex128, device=h.device)

        # Complex-pole residues
        if self.n_cmplx > 0:
            rc_flat = out[:, idx:idx + self.n_cmplx * 20]
            idx += self.n_cmplx * 20
            rc_per_pole = rc_flat.view(B, self.n_cmplx, 2, 10)
            R_cmplx = torch.zeros((B, self.n_cmplx, 4, 4),
                                   dtype=torch.complex128, device=h.device)
            for n in range(self.n_cmplx):
                R_cmplx[:, n] = self._scatter_symmetric(
                    rc_per_pole[:, n, 0], rc_per_pole[:, n, 1], B
                )
        else:
            R_cmplx = torch.zeros((B, 0, 4, 4), dtype=torch.complex128, device=h.device)

        # D matrix (only one sub-model carries D; others have D=0)
        if self.include_d:
            d_flat = out[:, idx:idx + 20]
            d_split = d_flat.view(B, 2, 10)
            D = self._scatter_symmetric(d_split[:, 0], d_split[:, 1], B)
        else:
            D = torch.zeros((B, 4, 4), dtype=torch.complex128, device=h.device)

        return R_real, R_cmplx, D


# ----------------------------------------------------------------------
# Rational layer evaluation (single sub-model)
# ----------------------------------------------------------------------
def evaluate_rational_sub(poles_real_norm: torch.Tensor,
                           poles_cmplx_norm: torch.Tensor,
                           R_real: torch.Tensor,
                           R_cmplx: torch.Tensor,
                           D: torch.Tensor,
                           s_norm: torch.Tensor) -> torch.Tensor:
    """
    Evaluate one sub-model's rational contribution at frequencies s_norm.

        S_i(s) = sum_n R_real_n / (s - p_real_n)
               + sum_n R_cmplx_n / (s - p_cmplx_n)
               + sum_n conj(R_cmplx_n) / (s - conj(p_cmplx_n))
               + D

    All inputs in NORMALIZED frequency. Returns (B, F, 4, 4) complex.
    Handles n_real=0 or n_cmplx=0 cleanly.
    """
    B = R_real.shape[0] if R_real.numel() > 0 else R_cmplx.shape[0]
    F_len = s_norm.shape[0]
    s_view = s_norm.view(1, 1, -1)

    out = D.unsqueeze(-1).expand(-1, -1, -1, F_len)  # (B, 4, 4, F)

    if R_real.shape[1] > 0:
        p_real_c = torch.complex(poles_real_norm, torch.zeros_like(poles_real_norm))
        denom_r = s_view - p_real_c.unsqueeze(-1)  # (B, n_real, F)
        term_r = R_real.unsqueeze(-1) / denom_r.unsqueeze(-2).unsqueeze(-2)
        out = out + term_r.sum(dim=1)

    if R_cmplx.shape[1] > 0:
        denom_c = s_view - poles_cmplx_norm.unsqueeze(-1)
        denom_cc = s_view - poles_cmplx_norm.conj().unsqueeze(-1)
        term_c = R_cmplx.unsqueeze(-1) / denom_c.unsqueeze(-2).unsqueeze(-2)
        term_cc = R_cmplx.conj().unsqueeze(-1) / denom_cc.unsqueeze(-2).unsqueeze(-2)
        out = out + term_c.sum(dim=1) + term_cc.sum(dim=1)

    return out.permute(0, 3, 1, 2)  # (B, F, 4, 4)


# ----------------------------------------------------------------------
# Helper: split VFIT pole basis into M slices for sub-model initialization
# ----------------------------------------------------------------------
def split_pole_basis(poles_real: torch.Tensor, poles_cmplx: torch.Tensor,
                      M: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Split the VFIT pole basis into M slices for warm-starting M sub-models.
    Real poles are distributed round-robin (typically there are very few).
    Complex poles are sorted by Im(p) and chunked into M frequency bands.

    Returns: list of M tuples (sub_poles_real, sub_poles_cmplx).
    """
    # Sort complex poles by imaginary part (frequency)
    sorted_idx = torch.argsort(poles_cmplx.imag)
    sorted_cmplx = poles_cmplx[sorted_idx]
    n_cmplx_per_sub = len(sorted_cmplx) // M
    remainder = len(sorted_cmplx) % M

    slices = []
    cursor = 0
    for i in range(M):
        # Distribute remainder among first sub-models
        n_cmplx_i = n_cmplx_per_sub + (1 if i < remainder else 0)
        sub_cmplx = sorted_cmplx[cursor:cursor + n_cmplx_i]
        cursor += n_cmplx_i
        # Real poles round-robin
        sub_real = poles_real[i::M] if len(poles_real) > 0 else torch.zeros(0, dtype=torch.float64)
        slices.append((sub_real, sub_cmplx))
    return slices


# ----------------------------------------------------------------------
# Full model: parallel decomposition with M sub-models
# ----------------------------------------------------------------------
class RationalForwardModel(nn.Module):
    """
    Forward map: (X_local, X_global, X_context) -> S(f) in C^(F, 4, 4)
    via parallel decomposition:

        S_total = sum_{i=1..M} S_i

    Each S_i is a low-order rational function with its own per-sample poles.
    Sub-models share the geometry and pair encoders but have separate
    pole and residue heads. Only sub-model 0 carries the D matrix; others
    have D=0 to avoid degenerate parameterization (M copies of D summing).
    """

    def __init__(self,
                 freqs_hz: torch.Tensor,
                 init_poles_real: torch.Tensor,
                 init_poles_cmplx: torch.Tensor,
                 M: int = 5,
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
        self.M = M
        self.f_scale_rad = 2 * math.pi * f_scale_hz

        # Build normalized s = j*omega/f_scale_rad for rational evaluation
        omega_norm = 2 * math.pi * freqs_hz / self.f_scale_rad
        s_norm = torch.complex(torch.zeros_like(omega_norm), omega_norm).to(torch.complex128)
        self.register_buffer("s_norm", s_norm)

        # Shared encoders (most of the parameter budget lives here)
        self.geom_encoder = GeometryEncoder(
            d_local=d_local, d_global=d_global,
            fourier_sigmas=fourier_sigmas,
            fourier_features=fourier_features,
            latent_dim=latent_geom_dim,
            hidden=encoder_hidden,
            n_blocks=3,
        )
        self.pair_encoder = PairEncoder(
            latent_geom_dim=latent_geom_dim,
            d_context=d_context,
            latent_dim=latent_pair_dim,
            hidden=encoder_hidden,
            n_blocks=2,
        )

        # Split VFIT basis across M sub-models
        slices = split_pole_basis(init_poles_real, init_poles_cmplx, M)
        self.sub_n_real = [int(s[0].numel()) for s in slices]
        self.sub_n_cmplx = [int(s[1].numel()) for s in slices]
        self.n_real_total = sum(self.sub_n_real)
        self.n_cmplx_total = sum(self.sub_n_cmplx)

        # Build M parallel sub-heads
        self.pole_heads = nn.ModuleList()
        self.residue_heads = nn.ModuleList()
        for i in range(M):
            sub_real, sub_cmplx = slices[i]
            self.pole_heads.append(SubPoleHead(
                latent_dim=latent_geom_dim,
                n_real=int(sub_real.numel()),
                n_cmplx=int(sub_cmplx.numel()),
                init_poles_real=sub_real,
                init_poles_cmplx=sub_cmplx,
                f_scale_rad=self.f_scale_rad,
            ))
            # Only sub-model 0 includes D to avoid M-way degenerate parameterization
            self.residue_heads.append(SubResidueHead(
                latent_dim=latent_pair_dim,
                n_real=int(sub_real.numel()),
                n_cmplx=int(sub_cmplx.numel()),
                include_d=(i == 0),
            ))

    def forward(self, x_local: torch.Tensor, x_global: torch.Tensor,
                x_context: torch.Tensor) -> torch.Tensor:
        # Encode geometry and pair context once (shared work)
        h_geom = self.geom_encoder(x_local, x_global)
        h_pair = self.pair_encoder(h_geom, x_context)

        # Each sub-model produces its own rational contribution; sum them
        S_total = None
        for i in range(self.M):
            poles_real_n, poles_cmplx_n = self.pole_heads[i](h_geom)
            R_real, R_cmplx, D = self.residue_heads[i](h_pair)
            S_i = evaluate_rational_sub(poles_real_n, poles_cmplx_n,
                                         R_real, R_cmplx, D, self.s_norm)
            if S_total is None:
                S_total = S_i
            else:
                S_total = S_total + S_i
        return S_total

    def get_delta_scales(self) -> List[float]:
        """Diagnostic: current delta_scale per sub-model. Reported during training."""
        return [head.delta_scale.item() for head in self.pole_heads]

    def get_poles_si_units(self, x_local: torch.Tensor,
                            x_global: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Diagnostic: return all sub-model poles in rad/s for inspection."""
        h_geom = self.geom_encoder(x_local, x_global)
        out = []
        for i in range(self.M):
            pr, pc = self.pole_heads[i](h_geom)
            out.append((pr * self.f_scale_rad, pc * self.f_scale_rad))
        return out


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------
def _self_test():
    """Build, forward + backward, verify shapes, stability, reciprocity, pole consistency."""
    torch.manual_seed(0)

    # Mock VFIT basis: 4 real + 20 complex (matches the real pole_basis.pt)
    init_poles_real = torch.tensor(
        [-3.9e12, -7.4e11, -6.8e10, -1.3e10], dtype=torch.float64
    )
    # Complex centroids: roughly spread across 5-100 GHz range
    fghz = [5.8, 11.3, 15.8, 24.4, 32.8, 36.2, 42.0, 42.5, 52.0, 60.0,
            64.6, 69.2, 78.0, 86.0, 88.5, 97.3, 105.9, 111.2, 144.5, 576.0]
    init_poles_cmplx = torch.tensor(
        [(-2e10) + 1j * (2 * math.pi * f * 1e9) for f in fghz],
        dtype=torch.complex128,
    )

    freqs = torch.linspace(0.25e9, 100e9, 401, dtype=torch.float64)
    M = 5
    model = RationalForwardModel(freqs, init_poles_real, init_poles_cmplx, M=M)

    n_total = sum(p.numel() for p in model.parameters())
    n_geom = sum(p.numel() for p in model.geom_encoder.parameters())
    n_pair = sum(p.numel() for p in model.pair_encoder.parameters())
    n_poles = sum(p.numel() for h in model.pole_heads for p in h.parameters())
    n_res = sum(p.numel() for h in model.residue_heads for p in h.parameters())
    print(f"  total params:          {n_total:,}")
    print(f"  geom encoder:          {n_geom:,} ({100*n_geom/n_total:.1f}%)")
    print(f"  pair encoder:          {n_pair:,} ({100*n_pair/n_total:.1f}%)")
    print(f"  pole heads (x{M}):      {n_poles:,} ({100*n_poles/n_total:.1f}%)")
    print(f"  residue heads (x{M}):   {n_res:,} ({100*n_res/n_total:.1f}%)")
    print(f"  sub-model breakdown:")
    for i in range(M):
        print(f"    sub {i}: {model.sub_n_real[i]} real + {model.sub_n_cmplx[i]} complex poles")
    print(f"  total layer poles: {model.n_real_total} real + {2*model.n_cmplx_total} complex"
          f" (with conjugates) = {model.n_real_total + 2*model.n_cmplx_total}")

    B = 4
    xl = torch.randn(B, 8)
    xg = torch.randn(B, 6)
    xc = torch.randn(B, 7)

    S = model(xl, xg, xc)
    print(f"\n  forward pass OK, S shape: {tuple(S.shape)}  dtype: {S.dtype}")
    assert S.shape == (B, 401, 4, 4)
    assert S.dtype == torch.complex128

    # Reciprocity check (symmetric residues structurally enforce S = S^T)
    asym = (S - S.transpose(-1, -2)).abs().max().item()
    print(f"  reciprocity |S - S^T| max: {asym:.3e}")
    assert asym < 1e-8

    # Stability check across all sub-models
    sub_poles = model.get_poles_si_units(xl, xg)
    for i, (pr, pc) in enumerate(sub_poles):
        if pr.numel() > 0:
            assert (pr.real < 0).all(), f"sub {i}: unstable real pole"
        if pc.numel() > 0:
            assert (pc.real < 0).all(), f"sub {i}: unstable complex pole (Re)"
            assert (pc.imag > 0).all(), f"sub {i}: complex pole not in UHP"
    print("  stability OK across all sub-models")

    # Pole consistency within a sim (key physics requirement)
    xl_same = torch.randn(1, 8).repeat(2, 1)
    xg_same = torch.randn(1, 6).repeat(2, 1)
    xc_diff = torch.randn(2, 7)
    poles_per_sub = model.get_poles_si_units(xl_same, xg_same)
    for i, (pr, pc) in enumerate(poles_per_sub):
        if pr.numel() > 0:
            assert torch.allclose(pr[0], pr[1]), f"sub {i}: real poles differ within sim"
        if pc.numel() > 0:
            assert torch.allclose(pc[0], pc[1]), f"sub {i}: cmplx poles differ within sim"
    print("  per-sim pole consistency OK across all sub-models")

    # Delta scales
    deltas = model.get_delta_scales()
    print(f"  delta_scales (init): {[f'{d:.2f}' for d in deltas]}")

    # Gradient flow
    target = torch.randn_like(S)
    loss = ((S.real - target.real) ** 2 + (S.imag - target.imag) ** 2).mean()
    loss.backward()
    n_grad = sum(1 for p in model.parameters()
                  if p.grad is not None and p.grad.abs().sum() > 0)
    n_tot = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  grad flow: {n_grad}/{n_tot} params receive non-zero gradient")
    assert n_grad == n_tot

    print("\nSelf-test passed.")


if __name__ == "__main__":
    _self_test()