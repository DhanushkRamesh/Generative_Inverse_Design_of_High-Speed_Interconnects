"""
synthetic_data.py
-----------------
Generates 4x4 S-parameter samples from known complex pole-residue pairs.

The math:
    S(s) = sum_n  R_n / (s - p_n)  +  D
where s = j*2*pi*f.

Physical constraints we enforce:
  - Poles come in complex-conjugate pairs (so the impulse response is real-valued).
  - Residue matrices for conjugate poles are conjugates of each other.
  - Each residue matrix R_n is symmetric (reciprocity: S = S^T).
  - D is real and symmetric.
  - All pole real parts are strictly negative (stability / causality).

We do NOT enforce passivity for the Day-1 sanity check, because we only care
whether the rational layer can recover the *parameters* of a known target.
Passivity is a constraint on the dataset, not on the layer's fitting ability.
"""

import numpy as np


def generate_synthetic_sample(
    n_pole_pairs: int = 20,
    n_freqs: int = 401,
    f_min: float = 0.25e9,
    f_max: float = 100e9,
    n_ports: int = 4,
    residue_scale: float = 5e9,
    D_scale: float = 0.02,
    seed: int | None = None,
):
    """Generate one synthetic 4x4 S-parameter trace from random poles/residues.

    Returns a dict with the ground-truth poles, residues, D, freqs, and S(f).
    The total number of poles is 2 * n_pole_pairs (each pair + its conjugate).
    """
    rng = np.random.default_rng(seed)

    # ----- 1. Generate complex-conjugate pole pairs --------------------------
    # Real part: negative damping. Use log-uniform so we get a mix of sharp
    # and broad resonances. Units are rad/s.
    sigma = rng.uniform(0.2e9, 5.0e9, size=n_pole_pairs) * 2 * np.pi  # damping
    omega = rng.uniform(1.0e9, 0.95 * f_max, size=n_pole_pairs) * 2 * np.pi  # resonance freq

    poles_upper = -sigma + 1j * omega          # Im > 0 half-plane
    poles_lower = -sigma - 1j * omega          # complex conjugates
    poles = np.concatenate([poles_upper, poles_lower])     # shape (2N,)

    # ----- 2. Generate symmetric residue matrices ----------------------------
    R_upper = np.zeros((n_pole_pairs, n_ports, n_ports), dtype=complex)
    for k in range(n_pole_pairs):
        A = (rng.standard_normal((n_ports, n_ports))
             + 1j * rng.standard_normal((n_ports, n_ports))) * residue_scale
        R_upper[k] = 0.5 * (A + A.T)           # symmetrise -> reciprocity
    R_lower = np.conj(R_upper)                  # conjugate residues
    residues = np.concatenate([R_upper, R_lower], axis=0)   # (2N, P, P)

    # ----- 3. Real symmetric direct-feedthrough term D -----------------------
    D = rng.standard_normal((n_ports, n_ports)) * D_scale
    D = 0.5 * (D + D.T)

    # ----- 4. Evaluate S(f) on the frequency grid ----------------------------
    freqs = np.linspace(f_min, f_max, n_freqs)
    s = 1j * 2 * np.pi * freqs                   # (F,) complex

    denom = s[:, None] - poles[None, :]          # (F, 2N)
    inv_denom = 1.0 / denom                      # (F, 2N)
    # Sum_n  R_n / (s - p_n)  using einsum: (F,N) * (N,P,P) -> (F,P,P)
    S = np.einsum('fn,npq->fpq', inv_denom, residues) + D[None, :, :]

    return {
        'poles': poles,                          # (2N,) complex
        'residues': residues,                    # (2N, P, P) complex
        'D': D,                                  # (P, P) real
        'freqs': freqs,                          # (F,) real, Hz
        'S': S,                                  # (F, P, P) complex
    }


def s_to_db(s_complex):
    """Convert complex S-parameter to magnitude in dB (with floor for log)."""
    return 20.0 * np.log10(np.abs(s_complex) + 1e-12)


if __name__ == "__main__":
    # Quick smoke test: generate one sample, print stats
    sample = generate_synthetic_sample(seed=42)
    print(f"Poles shape:    {sample['poles'].shape}")
    print(f"Residues shape: {sample['residues'].shape}")
    print(f"D shape:        {sample['D'].shape}")
    print(f"S shape:        {sample['S'].shape}")
    print(f"|S| range:      {np.abs(sample['S']).min():.4f} ... {np.abs(sample['S']).max():.4f}")
    print(f"Sdd11(0) [dB]:  {s_to_db(sample['S'][0, 0, 0]):.2f}")
    print(f"Sdd11(end)[dB]: {s_to_db(sample['S'][-1, 0, 0]):.2f}")
    print(f"Reciprocity check |S - S^T|_max: "
          f"{np.max(np.abs(sample['S'] - np.transpose(sample['S'], (0, 2, 1)))):.2e}")
    print(f"All poles Re<0: {np.all(sample['poles'].real < 0)}")