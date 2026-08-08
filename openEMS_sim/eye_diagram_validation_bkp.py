#!/usr/bin/env python3
"""
eye_diagram_validation.py
================================================================================
Stage 08 -- application-level eye-diagram comparison (112G PAM4, PRBS13Q).

WHAT THIS DOES
  Builds a 112 Gb/s PAM4 eye (56 GBaud, Nyquist ~= 28 GHz) from a via's
  differential insertion loss Sdd21, and compares the eye produced by the
  TARGET via against the eye produced by the GENERATED (inverse-designed) via.

IMPROVEMENTS IN THIS VERSION
  - Automated CDR (Clock Data Recovery) phase alignment locks the sampling
    point so the eye openings are perfectly centered at 1.0 UI (17.86 ps).
  - Exact 32 samples/UI integer resampling eliminates fractional sample drift.
  - Robust 5th/95th percentile cloud metrics for clean eye height extraction.
  - Added --out-name argument to easily set custom output image names.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = _THIS_DIR / "results" / "08_eye"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Link parameters (112G PAM4)
# ----------------------------------------------------------------------------
BAUD = 56e9                     # 56 GBaud -> 112 Gb/s PAM4
UI = 1.0 / BAUD                 # 17.857 ps
F_GRID = np.linspace(0.25e9, 100e9, 401)
F_PAD_MAX = 400e9
PAM4_LEVELS = np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0])

M_BE = (1 / np.sqrt(2)) * np.array(
    [[1, -1, 0, 0], [0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 1, 1]], float
)


# ----------------------------------------------------------------------------
# Channel model: lossy transmission line
# ----------------------------------------------------------------------------
def lossy_line_sdd21(freq, length_in, loss_db_per_in_at_ghz,
                     ref_ghz=1.0, eps_r=4.0):
    """Sdd21 of a lossy differential line of `length_in` inches (PER SIDE)."""
    c = 2.998e8
    f = np.asarray(freq, float)
    loss_db_per_in = loss_db_per_in_at_ghz * np.sqrt(f / (ref_ghz * 1e9))
    alpha_db = loss_db_per_in * length_in
    mag = 10.0 ** (-alpha_db / 20.0)
    L_m = length_in * 0.0254
    beta = 2 * np.pi * f * np.sqrt(eps_r) / c
    return mag * np.exp(-1j * beta * L_m)


# ----------------------------------------------------------------------------
# Sdd21 loading
# ----------------------------------------------------------------------------
def load_sdd21_csv(path: str) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    s = data["Sdd21_re"] + 1j * data["Sdd21_im"]
    f = data["freq_Hz"]
    if len(f) != len(F_GRID) or not np.allclose(f, F_GRID, rtol=1e-6):
        s = np.interp(F_GRID, f, s.real) + 1j * np.interp(F_GRID, f, s.imag)
    return s


def load_sdd21_npz(path: str, pair_k: int) -> np.ndarray:
    d = np.load(path)
    S = d["S"]
    base = 4 * (pair_k - 1)
    idx = [base, base + 2, base + 1, base + 3]
    s = S[:, idx][:, :, idx]
    s = 0.5 * (s + np.transpose(s, (0, 2, 1)))
    mm = M_BE @ s @ M_BE.T
    return mm[:, 1, 0]


# ----------------------------------------------------------------------------
# Spectrum -> impulse response
# ----------------------------------------------------------------------------
def impulse_response(sdd21: np.ndarray):
    df = 0.25e9
    n_meas = len(F_GRID)
    n_pos = int(F_PAD_MAX / df) + 1
    spec = np.zeros(n_pos, dtype=complex)
    spec[1:n_meas + 1] = sdd21
    spec[0] = np.abs(sdd21[0])
    
    # Smooth raised-cosine roll-off window
    k0 = int(0.8 * n_meas)
    w = np.ones(n_meas)
    ramp = np.linspace(0, np.pi, n_meas - k0)
    w[k0:] = 0.5 * (1 + np.cos(ramp))
    spec[1:n_meas + 1] *= w
    
    full = np.concatenate([spec, np.conj(spec[-2:0:-1])])
    h = np.fft.ifft(full).real
    n_t = len(full)
    dt = 1.0 / (n_t * df)
    t = np.arange(n_t) * dt
    return t, h, dt


# ----------------------------------------------------------------------------
# PRBS13Q PAM4 pattern generator
# ----------------------------------------------------------------------------
def prbs13_bits() -> np.ndarray:
    state = np.ones(13, dtype=int)
    bits = np.empty(8191, dtype=int)
    for i in range(8191):
        new = state[12] ^ state[11] ^ state[1] ^ state[0]
        bits[i] = state[12]
        state[1:] = state[:-1]
        state[0] = new
    return bits


def pam4_symbols() -> np.ndarray:
    b = prbs13_bits()
    if len(b) % 2:
        b = b[:-1]
    pairs = b.reshape(-1, 2)
    gray_index = pairs[:, 0] * 2 + (pairs[:, 0] ^ pairs[:, 1])
    return PAM4_LEVELS[gray_index]


# ----------------------------------------------------------------------------
# Eye construction with automated CDR phase alignment
# ----------------------------------------------------------------------------
def build_eye(sdd21: np.ndarray, sps_target: int = 32):
    t_raw, h_raw, _ = impulse_response(sdd21)
    
    # Resample impulse response to exact integer samples per UI
    dt = UI / float(sps_target)
    t_max = t_raw[-1]
    t_resampled = np.arange(0, t_max, dt)
    h = np.interp(t_resampled, t_raw, h_raw)
    
    sym = pam4_symbols()
    tx = np.repeat(sym, sps_target)
    n_wave = len(tx)
    
    # Truncate impulse response at 99.99% energy
    e = np.cumsum(h ** 2)
    n_keep = int(np.searchsorted(e, 0.9999 * e[-1])) + 1
    rx = np.convolve(tx, h[:n_keep])[:n_wave]
    
    # --- CDR Phase Lock (Find Optimal Sampling Phase) ---
    sps = sps_target
    seg_len = 2 * sps
    start_search = 100 * sps  # Skip initial transient
    
    best_p = 0
    max_opening = -1.0
    for p in range(sps):
        v_p = rx[start_search + p :: sps]
        if len(v_p) > 0:
            q_hi = np.quantile(v_p, 0.875)
            q_lo = np.quantile(v_p, 0.125)
            spread = q_hi - q_lo
            if spread > max_opening:
                max_opening = spread
                best_p = p
                
    # Center the optimal sampling point at UI = 1.0 (middle of 2-UI plot window)
    start_aligned = start_search + best_p - sps
    if start_aligned < 0:
        start_aligned = best_p
        
    n_seg = (n_wave - start_aligned) // seg_len
    segs = rx[start_aligned : start_aligned + n_seg * seg_len].reshape(n_seg, seg_len)
    
    t_eye = np.arange(seg_len) * dt * 1e12  # in ps
    
    # Extract eye heights at sample center (k_samp = sps, i.e., 1.0 UI)
    k_samp = sps
    v_samp = segs[:, k_samp]
    v_sorted = np.sort(v_samp)
    rails = np.quantile(v_sorted, [0.125, 0.375, 0.625, 0.875])
    
    heights = []
    for lo_rail, hi_rail in zip(rails[:-1], rails[1:]):
        lo_cloud = v_sorted[(v_sorted > lo_rail - 0.18) & (v_sorted < lo_rail + 0.18)]
        hi_cloud = v_sorted[(v_sorted > hi_rail - 0.18) & (v_sorted < hi_rail + 0.18)]
        if len(lo_cloud) > 5 and len(hi_cloud) > 5:
            # 5th percentile of upper cloud minus 95th percentile of lower cloud
            h_val = float(np.percentile(hi_cloud, 5) - np.percentile(lo_cloud, 95))
            heights.append(h_val)
        else:
            heights.append(float("nan"))
            
    metrics = {
        "rail_centres": [round(float(r), 4) for r in rails],
        "eye_heights": [round(h_, 4) for h_ in heights]
    }
    return t_eye, segs, metrics


def plot_eye(ax, t_eye, segs, title):
    step = max(1, len(segs) // 800)
    for s in segs[::step]:
        ax.plot(t_eye, s, color="tab:blue", alpha=0.05, lw=0.7)
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


# ----------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=str)
    src.add_argument("--npz", type=str)
    ap.add_argument("--pair", type=int, default=1)
    ap.add_argument("--label", type=str, default="channel")
    ap.add_argument("--compare-csv", type=str, default=None)
    ap.add_argument("--compare-npz", type=str, default=None)
    ap.add_argument("--compare-pair", type=int, default=1)
    ap.add_argument("--compare-label", type=str, default="generated")
    ap.add_argument("--channel-length", type=float, default=0.0,
                    help="lossy line length PER SIDE, inches (0 = bare via)")
    ap.add_argument("--channel-loss", type=float, default=1.0,
                    help="line loss dB/inch at 1 GHz, scales as sqrt(f)")
    ap.add_argument("--channel-eps", type=float, default=4.0,
                    help="effective dielectric constant of the line")
    ap.add_argument("--out-name", type=str, default=None,
                    help="Custom output filename or prefix (e.g. sim_pkg_5844)")
    args = ap.parse_args()

    def load(csv, npz, pair):
        return load_sdd21_csv(csv) if csv else load_sdd21_npz(npz, pair)

    print("=" * 70)
    print(f"Stage 08: 112G PAM4 eye (56 GBaud, UI = {UI*1e12:.3f} ps, PRBS13Q)")
    if args.channel_length > 0:
        print(f"  channel: {args.channel_length} in/side, "
              f"{args.channel_loss} dB/in@1GHz (sqrt-f), eps_r={args.channel_eps}")
    else:
        print("  channel: NONE (bare via)")
    print("=" * 70)

    # Build shared channel line
    _line = (lossy_line_sdd21(F_GRID, args.channel_length, args.channel_loss,
                              eps_r=args.channel_eps)
             if args.channel_length > 0 else None)

    s1 = load(args.csv, args.npz, args.pair)
    if _line is not None:
        s1 = s1 * _line * _line
    t_eye, segs, m1 = build_eye(s1)
    print(f"\n  [{args.label}] rail centres: {m1['rail_centres']}")
    print(f"  [{args.label}] eye heights (3 eyes): {m1['eye_heights']}")

    has_cmp = args.compare_csv or args.compare_npz
    if has_cmp:
        s2 = load(args.compare_csv, args.compare_npz, args.compare_pair)
        if _line is not None:
            s2 = s2 * _line * _line
        _, segs2, m2 = build_eye(s2)
        print(f"\n  [{args.compare_label}] rail centres: {m2['rail_centres']}")
        print(f"  [{args.compare_label}] eye heights: {m2['eye_heights']}")

        h1 = np.array(m1["eye_heights"], float)
        h2 = np.array(m2["eye_heights"], float)
        tot1, tot2 = np.nansum(h1), np.nansum(h2)
        
        pct = 100.0 * abs(tot1 - tot2) / max(abs(tot1), 1e-9)
        per = np.abs(h1 - h2)
        eye_open = (tot1 > 0) and (tot2 > 0)

        print("\n  --- EYE AGREEMENT (target vs generated) ---")
        print(f"    total eye opening [{args.label}]  : {tot1:.4f}")
        print(f"    total eye opening [{args.compare_label}]: {tot2:.4f}")
        print(f"    eye state: {'OPEN' if eye_open else 'CLOSED (reduce channel loss/length)'}")
        print(f"    total-opening difference          : {pct:.1f} %")
        print(f"    per-sub-eye |height diff|         : "
              f"{', '.join(f'{d:.4f}' for d in per)}")
              
        if not eye_open:
            print("    NOTE: eye is CLOSED under this channel configuration.")
            print("    Reduce --channel-length / --channel-loss if open eyes are required.")
        elif pct < 10.0:
            print(f"    VERDICT: eyes agree within {pct:.1f}% (<10%) -- the")
            print("    generated design reproduces the target's signal integrity.")
        else:
            print(f"    VERDICT: eyes differ by {pct:.1f}%.")

        ymax = 1.15 * max(np.abs(segs).max(), np.abs(segs2).max())
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), tight_layout=True)
        plot_eye(axes[0], t_eye, segs, f"{args.label} (112G PAM4)")
        plot_eye(axes[1], t_eye, segs2, f"{args.compare_label} (112G PAM4)")
        for a in axes:
            a.set_ylim(-ymax, ymax)
        fig.suptitle(f"112G PAM4 eye: {args.label} vs {args.compare_label} "
                     f"(opening diff {pct:.1f}%)")
    else:
        fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
        plot_eye(ax, t_eye, segs, f"{args.label} (112G PAM4)")

    # Output file handling
    if args.out_name:
        fname = args.out_name if args.out_name.endswith(".png") else f"{args.out_name}.png"
        out = OUT_DIR / fname
    elif has_cmp:
        out = OUT_DIR / f"best_eye_{args.label}_vs_{args.compare_label}.png"
    else:
        out = OUT_DIR / f"best_eye_{args.label}.png"

    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  figure saved: {out}")


if __name__ == "__main__":
    main()