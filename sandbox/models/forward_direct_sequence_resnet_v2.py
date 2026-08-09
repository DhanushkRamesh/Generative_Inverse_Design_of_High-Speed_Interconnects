"""
forward_model_tcn.py
Direct-Sequence Temporal Convolutional Network for differential pair S-parameters.

This is the final forward model. It builds on the working DirectSequenceResNet
(~2.11 dB passband MAE) by:
  1. Replacing fixed kernel-5 convs with DILATED convs (Bai-Kolter-Koltun 2018)
     to give a full-band receptive field across all 401 frequency points.
  2. Adding STRUCTURAL physics via:
       - Reciprocity:  symmetric residue scatter (S = S^T, exact)
       - Passivity:    soft loss + SVD projection at inference (Grivet-Talocia 2016)
       - Causality:    IFFT-roundtrip regularizer on diagonal reflections (Torun 2019)
  3. Multi-scale Fourier features on continuous geometry (Tancik et al. 2020)
     for sharper geometry-to-resonance mapping.

The architecture is a single non-decomposed direct-sequence CNN.  We have moved
away from the rational neuro-TF / decomposition approach (v3, v4) because, on
TUHH's wide-geometric-variation dataset, the rational approach hit the
"high-sensitivity issue" documented by Zhao et al. Micromachines 11(7) 696, 2020
and plateaued at ~5-7 dB MAE.  The direct-sequence approach (Torun et al. ICCAD
2019, S-TCNN) gives lower error at the cost of needing soft physics regularizers
instead of structural ones.

References (full thesis citations):
  Bai, Kolter, Koltun.  "An Empirical Evaluation of Generic Convolutional and
      Recurrent Networks for Sequence Modeling." arXiv:1803.01271, 2018.
      Dilated TCN design.
  Torun, Durgun, Aygun, Swaminathan.  "A Spectral Convolutional Net for
      Co-Optimization of Integrated Voltage Regulators and Embedded Inductors."
      IEEE/ACM ICCAD 2019.  Direct-sequence S-parameter NN (S-TCNN).
  Torun, Durgun, Aygun, Swaminathan.  "Enforcing Causality and Passivity of
      Neural Network Models of Broadband S-Parameters." IEEE EPEPS 2019.
      Causality (CEL) and passivity (PEL) enforcement layers.
  Torun, Durgun, Aygun, Swaminathan.  "Causal and Passive Parameterization of
      S-Parameters Using Neural Networks." IEEE TMTT, 2020.  Journal version.
  Triverio, Grivet-Talocia, Nakhla, Canavero, Achar.  "Stability, Causality,
      and Passivity in Electrical Interconnect Models." IEEE TAP 2007.
      Standard reference for SVD-based passivity perturbation.
  Grivet-Talocia, Gustavsen.  "Passive Macromodeling: Theory and Applications."
      Wiley, 2016.  Chapter 11.4 covers the SVD perturbation scheme used here.
  Tancik et al.  "Fourier Features Let Networks Learn High Frequency Functions
      in Low Dimensional Domains." NeurIPS 2020.
  Liu et al.  "Physics-Guided Neural Surrogate Model with Particle Swarm-Based
      Multi-Objective Optimization for Quasi-Coaxial TSV Interconnect Design."
      Micromachines 16(10) 1134, 2025.  Validates the soft physics regularizer
      approach on a similar interconnect problem.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Multi-scale Fourier feature encoding (Tancik et al. NeurIPS 2020)
# =============================================================================
class MultiScaleFourierFeatures(nn.Module):
    """
    Maps a low-dim continuous input x in R^d to [sin(2pi B x), cos(2pi B x)]
    with B drawn from N(0, sigma^2 I) at multiple sigmas.

    Used on X_local (8 continuous geometry/material features) so the geometry
    encoder can resolve sharp parameter-dependent resonance positions instead
    of low-pass smoothing them.
    """

    def __init__(self, in_dim: int,
                 sigmas: List[float] = [1.0, 4.0, 16.0],
                 n_features_per_scale: int = 32,
                 seed: int = 0):
        super().__init__()
        # Fixed random projection (not trained); reproducibility via seed
        g = torch.Generator().manual_seed(seed)
        bands = [torch.randn(in_dim, n_features_per_scale, generator=g) * s
                 for s in sigmas]
        B = torch.cat(bands, dim=1)
        self.register_buffer("B", B)
        self.out_dim = 2 * n_features_per_scale * len(sigmas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# =============================================================================
# Frequency positional encoding (sinusoidal, multi-scale)
# =============================================================================
def build_freq_positional_encoding(freqs_hz: torch.Tensor,
                                    n_dim: int = 8,
                                    f_max: float = 100e9) -> torch.Tensor:
    """
    Sinusoidal positional encoding over the frequency axis, like
    Transformer positional encodings but applied to S-parameter freq grid.

    Returns (n_dim, F) buffer.  Each channel is sin or cos at a different
    angular wavelength spanning low to high frequency variation rates.

    This lets the conv backbone distinguish "I'm at 5 GHz" from "I'm at 90 GHz"
    even though the conv kernels are translation-invariant.
    """
    f_norm = freqs_hz / f_max  # in [~0, 1]
    F_len = f_norm.shape[0]
    pe = torch.zeros(n_dim, F_len, dtype=torch.float32)
    half = n_dim // 2
    # Wavelengths spaced exponentially from 1 down to ~1/100
    div = torch.exp(torch.arange(half) * (-math.log(100.0) / max(half - 1, 1)))
    for i in range(half):
        pe[2 * i, :] = torch.sin(f_norm * div[i] * 2 * math.pi)
        pe[2 * i + 1, :] = torch.cos(f_norm * div[i] * 2 * math.pi)
    return pe  # (n_dim, F)


# =============================================================================
# Dilated 1D residual block (TCN-style, Bai-Kolter-Koltun 2018)
# =============================================================================
class DilatedConvBlock(nn.Module):
    """
    Pre-norm residual block with two dilated 1D convolutions.

    Replicate padding is used so the conv treats the frequency axis as having
    held-boundary behavior at 0 and 100 GHz, not zero-padding (which would
    bias predictions near the band edges).
    """

    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1,
                 dropout: float = 0.1, n_groups: int = 8):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.norm1 = nn.GroupNorm(n_groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                                padding=pad, dilation=dilation,
                                padding_mode="replicate")
        self.norm2 = nn.GroupNorm(n_groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                                padding=pad, dilation=dilation,
                                padding_mode="replicate")
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(self.conv1(h))
        h = self.drop(h)
        h = self.norm2(h)
        h = self.conv2(h)
        return x + h


# =============================================================================
# Direct Sequence TCN model
# =============================================================================
class DirectSequenceTCN(nn.Module):
    """
    Maps (X_local, X_global, X_context) -> S(f) in C^(F, 4, 4) directly via
    a temporal convolutional backbone.

    Structural physics:
      - Reciprocity: only 10 unique elements predicted per (B, F), scattered
        symmetrically into a 4x4 complex matrix.

    Soft physics (via loss):
      - Passivity: relu(sigma_max - 1) summed over all frequencies
      - Causality: IFFT-roundtrip energy in non-causal time region (diagonal only)

    Hard physics (at inference):
      - Passivity: SVD projection clamps singular values to <= 1
        (Grivet-Talocia & Gustavsen 2016, eq. 11.86; Triverio et al. 2007)
    """

    # Upper-triangle indices for a 4x4 symmetric matrix: 10 unique elements
    UPPER_R = (0, 0, 0, 0, 1, 1, 1, 2, 2, 3)
    UPPER_C = (0, 1, 2, 3, 1, 2, 3, 2, 3, 3)

    def __init__(self,
                 freqs_hz: torch.Tensor,
                 d_local: int = 8,
                 d_global: int = 6,
                 d_context: int = 7,
                 fourier_sigmas: List[float] = [1.0, 4.0, 16.0],
                 fourier_features: int = 32,
                 hidden_dim: int = 256,
                 dilations: List[int] = [1, 2, 4, 8, 16, 32, 1],
                 kernel_size: int = 5,
                 dropout: float = 0.10,
                 freq_pe_dim: int = 8,
                 f_max_hz: float = 100e9):
        super().__init__()
        self.F_len = freqs_hz.shape[0]

        # ----- Geometry encoder -----
        # Fourier features expand X_local; X_global is concatenated raw.
        self.fourier = MultiScaleFourierFeatures(
            d_local, sigmas=fourier_sigmas,
            n_features_per_scale=fourier_features,
        )
        geom_in = self.fourier.out_dim + d_global + d_context
        self.geom_mlp = nn.Sequential(
            nn.Linear(geom_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # ----- Frequency positional encoding -----
        # Computed once at construction; held as buffer for device portability
        pe = build_freq_positional_encoding(freqs_hz, n_dim=freq_pe_dim,
                                             f_max=f_max_hz)
        self.register_buffer("freq_pe", pe)  # (freq_pe_dim, F)

        # ----- Sequence input projection -----
        # Input channels: hidden_dim (geometry broadcast) + freq_pe_dim
        self.proj_in = nn.Conv1d(hidden_dim + freq_pe_dim, hidden_dim,
                                  kernel_size=1)

        # ----- TCN backbone with exponential dilations -----
        # Receptive field with kernel=5 and 2 convs per block:
        #   RF = 1 + (k-1) * 2 * sum(dilations)
        # For dilations [1,2,4,8,16,32,1]: RF = 1 + 4 * 2 * 64 = 513  (>> 401, full coverage)
        self.blocks = nn.ModuleList([
            DilatedConvBlock(hidden_dim, kernel_size=kernel_size,
                             dilation=d, dropout=dropout)
            for d in dilations
        ])
        self.dilations = dilations

        # ----- Output head: 1x1 conv to 20 channels (10 real + 10 imag) -----
        self.proj_out = nn.Conv1d(hidden_dim, 20, kernel_size=1)
        # Small init so the model starts near zero and learns to add structure;
        # avoids exploding initial passivity / causality penalties.
        nn.init.normal_(self.proj_out.weight, std=1e-3)
        nn.init.zeros_(self.proj_out.bias)

        # Indices for symmetric scatter
        self.register_buffer("upper_r", torch.tensor(self.UPPER_R, dtype=torch.long))
        self.register_buffer("upper_c", torch.tensor(self.UPPER_C, dtype=torch.long))

    # ---- Helper: 1D sequence (B, 10, F) -> symmetric 4x4 complex (B, F, 4, 4) ----
    def _scatter_symmetric(self, vec_real: torch.Tensor,
                            vec_imag: torch.Tensor) -> torch.Tensor:
        B = vec_real.shape[0]
        F_len = vec_real.shape[-1]
        # Allocate real and imaginary parts separately (cheaper than complex zeros)
        re = torch.zeros((B, 4, 4, F_len), dtype=torch.float64,
                          device=vec_real.device)
        im = torch.zeros((B, 4, 4, F_len), dtype=torch.float64,
                          device=vec_real.device)
        # Scatter upper triangle
        re[:, self.upper_r, self.upper_c, :] = vec_real.to(torch.float64)
        im[:, self.upper_r, self.upper_c, :] = vec_imag.to(torch.float64)
        # Mirror to lower triangle for symmetry; diagonal is overwritten with same value
        re[:, self.upper_c, self.upper_r, :] = vec_real.to(torch.float64)
        im[:, self.upper_c, self.upper_r, :] = vec_imag.to(torch.float64)
        # Combine and reorder to (B, F, 4, 4)
        S = torch.complex(re, im)
        return S.permute(0, 3, 1, 2)

    def forward(self, x_local: torch.Tensor, x_global: torch.Tensor,
                x_context: torch.Tensor) -> torch.Tensor:
        B = x_local.shape[0]

        # 1. Encode geometry (per-sample latent)
        ff = self.fourier(x_local)
        x = torch.cat([ff, x_global, x_context], dim=-1)
        h_geom = self.geom_mlp(x)  # (B, hidden_dim)

        # 2. Broadcast to sequence and concatenate with frequency PE
        h_seq = h_geom.unsqueeze(-1).expand(-1, -1, self.F_len)  # (B, hidden, F)
        pe = self.freq_pe.unsqueeze(0).expand(B, -1, -1)  # (B, freq_pe_dim, F)
        h = torch.cat([h_seq, pe], dim=1)
        h = self.proj_in(h)

        # 3. TCN backbone with dilated convolutions
        for block in self.blocks:
            h = block(h)

        # 4. Output: 20 channels split into 10 real + 10 imag (B, 20, F)
        out = self.proj_out(h)
        vec_real = out[:, :10, :]
        vec_imag = out[:, 10:, :]

        # 5. Symmetric scatter to (B, F, 4, 4) complex
        return self._scatter_symmetric(vec_real, vec_imag)


# =============================================================================
# Passivity SVD projection (post-hoc, applied at inference)
# Reference: Grivet-Talocia & Gustavsen 2016, "Passive Macromodeling", eq. 11.86
#           Triverio et al. IEEE TAP 2007
# =============================================================================
@torch.no_grad()
def passivity_project_svd(S: torch.Tensor) -> torch.Tensor:
    """
    Project S onto the set of passive matrices via SVD clamping.

    Singular values of a passive scattering matrix satisfy sigma_i <= 1 for all i.
    For each frequency f, if sigma_max(S(f)) > 1, decompose S = U diag(sigma) V^H,
    clamp sigma_i to min(sigma_i, 1), and recompose.

    This is rank-preserving and minimally perturbative.  Standard tool in
    commercial SI flows (HSPICE PASSIVITY=ON, ADS).

    S: (B, F, 4, 4) complex.  Returns same shape, passive.
    """
    U, sv, Vh = torch.linalg.svd(S, full_matrices=False)
    sv_clamped = torch.clamp(sv, max=1.0)
    # Recompose: U @ diag(sv) @ Vh, with sv broadcast to complex
    S_proj = U @ torch.diag_embed(sv_clamped.to(S.dtype)) @ Vh
    return S_proj


# =============================================================================
# Passivity loss (soft, applied during training)
# =============================================================================
def passivity_loss(S: torch.Tensor) -> torch.Tensor:
    """
    Squared excess of maximum singular value above 1, averaged over batch
    and frequency.  Applied at EVERY frequency point (not random subsample)
    so the model gets a passivity signal everywhere.

    This is the soft analog of Torun's PEL (Passivity Enforcement Layer,
    EPEPS 2019), implemented as a regularizer rather than a hard layer.

    Squared (not just relu) gives a quadratic penalty that grows fast when
    violation is large, encouraging the model to actually push sigma below 1.
    """
    sv = torch.linalg.svdvals(S)  # (B, F, 4)
    sigma_max = sv.max(dim=-1).values  # (B, F)
    excess = F.relu(sigma_max - 1.0)
    return (excess ** 2).mean()


# =============================================================================
# Causality residual (IFFT round-trip energy in non-causal region)
# Reference: Torun et al. EPEPS 2019, "Enforcing Causality and Passivity..."
#           Triverio et al. IEEE TAP 2007 (Hilbert/Kramers-Kronig formalism)
# =============================================================================
def causality_loss(S: torch.Tensor) -> torch.Tensor:
    """
    Energy of the time-domain impulse response at t < 0, for the diagonal
    reflection coefficients.

    For a causal LTI system, the impulse response h(t) is zero for t < 0.
    For S-parameters sampled at positive frequencies only, we construct the
    full bilateral spectrum via Hermitian symmetry S(-f) = conj(S(f)), then
    IFFT to obtain h(t), then measure the energy of the non-causal half.

    Applied only to the diagonal reflection elements (S11, S22, S33, S44)
    because off-diagonal transmission elements have group delay that makes
    the simple Hilbert/causality relation harder to enforce directly.
    """
    # Diagonal elements only: (B, F, 4)
    diag = torch.diagonal(S, dim1=-2, dim2=-1)

    # Build bilateral spectrum via Hermitian symmetry, skipping DC duplicate
    diag_neg = torch.conj(torch.flip(diag[:, 1:], dims=[1]))
    diag_full = torch.cat([diag_neg, diag], dim=1)  # (B, 2F-1, 4)

    # IFFT to time domain, then fftshift to put t=0 at center
    h_time = torch.fft.ifft(diag_full, dim=1)
    h_time = torch.fft.fftshift(h_time, dim=1)

    # Non-causal region: first half (t < 0)
    N = h_time.shape[1]
    h_noncausal = h_time[:, :N // 2]
    return h_noncausal.abs().pow(2).mean()


# =============================================================================
# Self-test
# =============================================================================
def _self_test():
    """Build, forward + backward, verify shapes, reciprocity, gradient flow."""
    torch.manual_seed(0)

    freqs = torch.linspace(0.25e9, 100e9, 401, dtype=torch.float64)
    model = DirectSequenceTCN(freqs_hz=freqs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  total params:    {n_params:,}")
    print(f"  dilations used:  {model.dilations}")
    rf = 1 + (5 - 1) * 2 * sum(model.dilations)
    print(f"  receptive field: {rf} samples (covers {min(rf, 401)}/401 freq points)")

    B = 4
    xl = torch.randn(B, 8)
    xg = torch.randn(B, 6)
    xc = torch.randn(B, 7)

    S = model(xl, xg, xc)
    print(f"\n  forward shape:  {tuple(S.shape)}  dtype: {S.dtype}")
    assert S.shape == (B, 401, 4, 4)
    assert S.dtype == torch.complex128

    asym = (S - S.transpose(-1, -2)).abs().max().item()
    print(f"  reciprocity |S - S^T| max: {asym:.3e}  (structural)")
    assert asym < 1e-8

    pl = passivity_loss(S).item()
    cl = causality_loss(S).item()
    print(f"  passivity loss (untrained): {pl:.3e}")
    print(f"  causality loss (untrained): {cl:.3e}")

    # SVD projection sanity
    S_proj = passivity_project_svd(S)
    sv_after = torch.linalg.svdvals(S_proj).max().item()
    print(f"  max sigma after SVD projection: {sv_after:.4f}  (should be <= 1)")
    assert sv_after <= 1.0 + 1e-6

    # Gradient flow
    target = torch.randn_like(S)
    loss = ((S.real - target.real) ** 2 + (S.imag - target.imag) ** 2).mean()
    loss.backward()
    n_grad = sum(1 for p in model.parameters()
                  if p.grad is not None and p.grad.abs().sum() > 0)
    n_tot = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  grad flow: {n_grad}/{n_tot} params receive gradient")
    assert n_grad == n_tot

    print("\n  Self-test passed.")


if __name__ == "__main__":
    _self_test()