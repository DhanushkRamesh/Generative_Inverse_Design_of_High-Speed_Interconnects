"""
physics_utils.py
================
SI/PI math helpers used by the diff-pair parser and any downstream analysis.

This module collects three small operations that every parsing pass needs:

  1. Single-ended -> mixed-mode conversion (Bockelman-Eisenstadt).
     We assume the four input ports are ordered  [TX+, TX-, RX+, RX-]
     and the output is  [d_TX, d_RX, c_TX, c_RX].
     Reference: D. E. Bockelman and W. R. Eisenstadt,
       "Combined differential and common-mode scattering parameters: theory
        and simulation," IEEE TMTT, vol. 43, no. 7, July 1995.

  2. Reciprocity enforcement.
     A reciprocal LTI network has S = S^T at every frequency.  Real-world
     simulations carry small numerical asymmetries; projecting onto the
     reciprocal subspace via  (S + S^T) / 2  removes them before passivity
     is checked.  This is a standard step in vector-fitting pipelines.

  3. Passivity check via eigenvalues of the dissipation matrix.
     A passive network satisfies  Q = I - S^H S >= 0  (positive semi-definite)
     at every frequency.  We use np.linalg.eigvalsh, which is the
     symmetric/Hermitian eigensolver and is exact to within float64 epsilon.
     Reference: Triverio et al., "Stability, Causality, and Passivity in
       Electrical Interconnect Models," IEEE TADVP, 2007.

Conventions
-----------
- All S-parameter arrays are numpy complex arrays of shape (F, P, P) where
  F is the number of frequency points and P is the number of ports.
- All routines are pure numpy (no torch); torch conversion happens in the
  main pipeline after parsing is complete.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Bockelman-Eisenstadt 4x4 mixed-mode matrix
# ---------------------------------------------------------------------------
# Acts on a 4-port single-ended S-matrix with ports ordered as
#   [TX+, TX-, RX+, RX-]
# and produces the mixed-mode S-matrix with ports ordered as
#   [diff TX, diff RX, common TX, common RX].
#
# Row 0:  diff TX  = (P1 - P2) / sqrt(2)
# Row 1:  diff RX  = (P3 - P4) / sqrt(2)
# Row 2:  com  TX  = (P1 + P2) / sqrt(2)
# Row 3:  com  RX  = (P3 + P4) / sqrt(2)
#
# The matrix is real and orthogonal (M M^T = I), so it commutes with reciprocity
# and preserves passivity.  S_mm = M @ S_se @ M.T
M_BE = (1.0 / np.sqrt(2.0)) * np.array(
    [
        [1.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)


def convert_to_mixed_mode(s_se: np.ndarray) -> np.ndarray:
    """Apply the Bockelman-Eisenstadt SE -> MM transform.

    Parameters
    ----------
    s_se : (F, 4, 4) complex
        Single-ended S-parameters with ports in [TX+, TX-, RX+, RX-] order.

    Returns
    -------
    s_mm : (F, 4, 4) complex
        Mixed-mode S-parameters with ports in [d_TX, d_RX, c_TX, c_RX] order.
    """
    if s_se.ndim != 3 or s_se.shape[1:] != (4, 4):
        raise ValueError(
            f"Expected s_se of shape (F, 4, 4); got {s_se.shape}"
        )
    # Broadcasting: M_BE has shape (4, 4); s_se has shape (F, 4, 4).
    # Matrix product is well-defined element-wise across the F axis.
    return M_BE @ s_se @ M_BE.T


def enforce_reciprocity(s: np.ndarray) -> np.ndarray:
    """Project an S-matrix onto the reciprocal subspace.

    Computes  S_recip(f) = ( S(f) + S(f)^T ) / 2  for every frequency.
    This kills small numerical asymmetries from the EM solver before
    downstream passivity checks.

    Parameters
    ----------
    s : (F, P, P) complex

    Returns
    -------
    s_recip : (F, P, P) complex
    """
    if s.ndim != 3 or s.shape[1] != s.shape[2]:
        raise ValueError(f"Expected s of shape (F, P, P); got {s.shape}")
    # Transpose only the last two axes (per-frequency transpose).
    return 0.5 * (s + np.transpose(s, (0, 2, 1)))


def reciprocity_residual(s: np.ndarray) -> float:
    """Diagnostic: how non-reciprocal was the input before we enforced it?

    Returns the maximum element-wise |S - S^T| across all frequencies.
    Typical raw simulation values: ~1e-5 (good) to ~1e-3 (acceptable);
    anything above ~1e-2 is a flag.
    """
    return float(np.max(np.abs(s - np.transpose(s, (0, 2, 1)))))


def check_passivity(
    s: np.ndarray, threshold: float = -1e-6
) -> tuple[bool, float]:
    """Eigenvalue-based passivity check.

    For every frequency f, compute  Q(f) = I - S(f)^H S(f).
    The network is passive iff Q(f) is positive semi-definite for all f,
    i.e. the minimum eigenvalue of Q over f is >= 0.

    We allow a small negative tolerance (default: -1e-6) to absorb
    numerical noise from the EM solver; a strictly tighter threshold
    would drop a non-trivial fraction of otherwise-valid samples.

    Parameters
    ----------
    s : (F, P, P) complex
    threshold : float
        Lower bound on min(eig(Q)) that still counts as passive.

    Returns
    -------
    is_passive : bool
    min_eigenvalue : float
        The actual minimum eigenvalue across all frequencies (real-valued
        because Q is Hermitian by construction).
    """
    if s.ndim != 3 or s.shape[1] != s.shape[2]:
        raise ValueError(f"Expected s of shape (F, P, P); got {s.shape}")

    n_ports = s.shape[1]
    identity = np.eye(n_ports, dtype=np.complex128)

    min_eig = np.inf
    for f_idx in range(s.shape[0]):
        S = s[f_idx].astype(np.complex128)
        # Q is Hermitian by construction: (I - S^H S)^H = I - S^H S
        Q = identity - S.conj().T @ S
        # eigvalsh assumes Hermitian and returns sorted real eigenvalues
        eigs = np.linalg.eigvalsh(Q)
        if eigs[0] < min_eig:
            min_eig = float(eigs[0])

    return (min_eig >= threshold), min_eig


def s_to_db(s: np.ndarray) -> np.ndarray:
    """Magnitude of complex S in dB with a small floor to avoid log(0)."""
    return 20.0 * np.log10(np.abs(s) + 1e-12)


# ---------------------------------------------------------------------------
# Smoke test — run `python -m utils.physics_utils` to exercise the module.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Build a tiny passive reciprocal toy S-matrix and verify all three checks.
    F = 11
    rng = np.random.default_rng(0)
    # Generate a passive S as I - A A^H where A is a random small matrix
    s_test = np.zeros((F, 4, 4), dtype=np.complex128)
    for f in range(F):
        A = (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))) * 0.2
        # Symmetrise to make it reciprocal
        A = 0.5 * (A + A.T)
        # Map to S by  S = A  (just keep |S| small enough that I - S^H S > 0)
        s_test[f] = A

    print("Reciprocity residual before enforce_reciprocity:",
          reciprocity_residual(s_test))
    s_test = enforce_reciprocity(s_test)
    print("Reciprocity residual after  enforce_reciprocity:",
          reciprocity_residual(s_test))

    is_passive, min_eig = check_passivity(s_test)
    print(f"Passivity check: passive={is_passive}, min_eig={min_eig:.4e}")

    s_mm = convert_to_mixed_mode(s_test)
    print(f"Mixed-mode shape: {s_mm.shape}")
    print("All physics_utils smoke tests passed.")