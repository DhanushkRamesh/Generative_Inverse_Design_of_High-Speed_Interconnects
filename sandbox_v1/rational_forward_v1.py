"""
rational_layer.py  (v2 - with frequency normalisation)
------------------------------------------------------
Differentiable rational layer:   S(s) = sum_n R_n / (s - p_n) + D
where s = j*2*pi*freqs.

What's new vs v1:
  All internal arithmetic is in NORMALISED units s_hat = s / (2*pi*f_scale).
  This keeps |s_hat| and |p_hat| in O(1), which gives sane gradient magnitudes
  for Adam without needing exotic learning rates.

  Why this is mathematically lossless:
      S(s) = sum_n  R_n / (s - p_n) + D
           = sum_n  (R_n / w_scale) / (s/w_scale - p_n/w_scale) + D
           = sum_n  R_hat_n      /   (s_hat   -  p_hat_n)        + D
  where w_scale = 2*pi*f_scale.

  We train R_hat_n directly. The "physical" residues are R_n = w_scale * R_hat_n,
  but for SI surrogate modelling we never actually need them in physical units
  — only S(f) matters.
"""

import math
import torch
import torch.nn as nn


class RationalLayer(nn.Module):
    """Rational layer with internal frequency normalisation."""

    def __init__(
        self,
        poles: torch.Tensor,
        n_ports: int = 4,
        f_scale: float = 100e9,
        init_scale: float = 0.01,
        dtype: torch.dtype = torch.complex128,
    ):
        """
        Args:
            poles:    complex tensor (N,), physical-units poles, all Re < 0.
            n_ports:  number of ports P.
            f_scale:  frequency scale used to normalise s and p (Hz).
                      Pick f_scale = max frequency in the band.
            init_scale: stddev of residue initialisation in normalised units.
            dtype:    complex128 strongly recommended; complex64 OK if speed
                      matters and the band is < ~10 GHz.
        """
        super().__init__()
        assert poles.is_complex(), "poles must be a complex tensor"
        assert torch.all(poles.real < 0), "all pole real parts must be negative"
        self.dtype = dtype
        self.real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32

        self.f_scale = f_scale
        self.w_scale = 2.0 * math.pi * f_scale

        # Store normalised poles as a buffer.
        poles_hat = poles.to(dtype) / self.w_scale
        self.register_buffer('poles_hat', poles_hat)
        self.N = poles.shape[0]
        self.n_ports = n_ports

        # Trainable normalised residues + D.  Two real tensors -> recombine.
        self.residues_re = nn.Parameter(
            torch.randn(self.N, n_ports, n_ports, dtype=self.real_dtype) * init_scale
        )
        self.residues_im = nn.Parameter(
            torch.randn(self.N, n_ports, n_ports, dtype=self.real_dtype) * init_scale
        )
        self.D = nn.Parameter(torch.zeros(n_ports, n_ports, dtype=self.real_dtype))

    def forward(self, freqs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            freqs: real tensor (F,), physical frequencies in Hz.
        Returns:
            S: complex tensor (F, P, P) in physical units.
        """
        # Normalised frequency variable
        s_hat = 1j * (freqs.to(self.real_dtype) / self.f_scale).to(self.dtype)  # (F,)

        residues_hat = (self.residues_re + 1j * self.residues_im).to(self.dtype)
        denom = s_hat.unsqueeze(-1) - self.poles_hat.unsqueeze(0)                # (F, N)
        inv_denom = 1.0 / denom

        S = torch.einsum('fn,npq->fpq', inv_denom, residues_hat)
        S = S + self.D.to(self.dtype)
        return S


def complex_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE on real+imag parts of a complex tensor."""
    diff = pred - target
    return (diff.real ** 2 + diff.imag ** 2).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    dummy_poles = torch.complex(
        -torch.rand(40) * 1e10 - 1e9,
        (torch.rand(40) - 0.5) * 6e11
    )
    layer = RationalLayer(dummy_poles, n_ports=4, f_scale=100e9)
    freqs = torch.linspace(0.25e9, 100e9, 401)
    S = layer(freqs)
    print(f"Output S shape: {S.shape}, dtype: {S.dtype}")
    print(f"|S| range:      {S.abs().min().item():.4f} ... {S.abs().max().item():.4f}")
    print(f"Param count:    {sum(p.numel() for p in layer.parameters() if p.requires_grad):,}")