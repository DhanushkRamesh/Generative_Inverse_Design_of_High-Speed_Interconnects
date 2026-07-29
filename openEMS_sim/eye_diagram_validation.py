"""
08_pam4_eye.py
================================================================================
Stage 08 of the openEMS validation pipeline -- the application-level finale.

PURPOSE
    Turn an OpenEMS-simulated differential insertion loss Sdd21(f) into a
    112 Gb/s PAM4 eye diagram. S-parameters are abstract; the eye is what a
    link designer actually judges. The thesis finale figure is two eyes side
    by side -- one from the TARGET response, one from the OpenEMS-simulated
    GENERATED design -- processed identically. Open, comparable eyes = the
    designed interconnect carries real 112G traffic.

SIGNAL CHAIN (standard S-parameter -> eye flow)
    1. Load Sdd21(f) on the 0.25-100 GHz / 401-pt grid.
    2. Condition the spectrum: extrapolate a DC point, enforce Hermitian
       symmetry, raised-cosine window the band edge (reduces IFFT ringing).
    3. Zero-pad the spectrum to 400 GHz so the time step is dt = 1.25 ps
       (14.3 samples per UI at 56 GBaud -- enough to draw a clean eye).
       Zero-padding adds no information; Sdd21 is ~-30 dB by 100 GHz, so the
       band-limited assumption is honest.
    4. IFFT -> causal impulse response h(t).
    5. Generate a PRBS13Q PAM4 symbol stream (IEEE-standard test pattern:
       PRBS13 bit sequence, consecutive bit pairs Gray-mapped to 4 levels
       {-1, -1/3, +1/3, +1}), 56 GBaud, UI = 17.857 ps.
    6. Convolve waveform with h(t), discard the transient, fold into 2-UI
       segments, overlay -> eye diagram.
    7. Report the three PAM4 eye heights at the centre sampling instant.

    No equalisation is applied (un-equalised eye). Both target and generated
    responses go through the IDENTICAL chain, so the comparison is fair.

INPUTS (either)
    --csv  <file>   stage 03b/05 ground-truth CSV
                    (freq_Hz, Sdd11_re, Sdd11_im, Sdd21_re, Sdd21_im, ...)
    --npz  <file> --pair K
                    a stage 04/06/07 single-ended result
                    ({sim}_openems_se.npz or case_{i}_openems.npz);
                    pair K is extracted with the pipeline conventions.
    Give --label to name the curve; run twice (target, generated) and use
    --compare-with to overlay both eyes in one figure.

USAGE
    # eye of the CONMLS target for sim_0017 pair 1 (from stage 03b export):
    python 08_pam4_eye.py --csv results/03b_touchstone/sim_pkg_0017_pair1_conmls_mixedmode.csv --label target

    # eye of an OpenEMS-simulated generated design:
    python 08_pam4_eye.py --npz results/07_generated/case_0_openems.npz --pair 1 --label generated

    # side-by-side finale figure:
    python 08_pam4_eye.py --csv <target csv> --label target \
        --compare-npz results/07_generated/case_0_openems.npz --compare-pair 1 \
        --compare-label generated
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
F_PAD_MAX = 400e9               # zero-pad spectrum to here -> dt = 1.25 ps
PAM4_LEVELS = np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0])  # Gray order 00,01,11,10

M_BE = (1 / np.sqrt(2)) * np.array(
    [[1, -1, 0, 0], [0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 1, 1]], float)


# ----------------------------------------------------------------------------
# Sdd21 loading
# ----------------------------------------------------------------------------
def load_sdd21_csv(path: str) -> np.ndarray:
    """From the stage 03b/05 ground-truth CSV format."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    s = data["Sdd21_re"] + 1j * data["Sdd21_im"]
    f = data["freq_Hz"]
    if len(f) != len(F_GRID) or not np.allclose(f, F_GRID, rtol=1e-6):
        s = (np.interp(F_GRID, f, s.real) + 1j * np.interp(F_GRID, f, s.imag))
    return s


def load_sdd21_npz(path: str, pair_k: int) -> np.ndarray:
    """From a stage 04/06/07 single-ended npz; extract pair k mixed-mode."""
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
    """Windowed, zero-padded, Hermitian IFFT -> (t, h) causal impulse response.

    Frequency grid handling: the data lives on 0.25-100 GHz. We build a
    uniform 0..F_PAD_MAX grid at df = 0.25 GHz (mil-stone: 0.25 GHz happens to
    be both the first point and the spacing of the 401-pt grid), fill DC by
    extrapolating |S| flat from the first point (phase -> 0 at DC), window the
    100 GHz edge with a raised cosine over the last 20% of the measured band,
    and zero beyond 100 GHz.
    """
    df = 0.25e9
    n_meas = len(F_GRID)                       # 401 points, 0.25..100 GHz
    n_pos = int(F_PAD_MAX / df) + 1            # 0..400 GHz inclusive
    spec = np.zeros(n_pos, dtype=complex)

    # measured band: grid index k corresponds to f = k*df, k = 1..400
    spec[1:n_meas + 1] = sdd21

    # DC point: magnitude of the lowest measured point, zero phase
    spec[0] = np.abs(sdd21[0])

    # raised-cosine window over the top 20% of the measured band
    k0 = int(0.8 * n_meas)
    w = np.ones(n_meas)
    ramp = np.linspace(0, np.pi, n_meas - k0)
    w[k0:] = 0.5 * (1 + np.cos(ramp))
    spec[1:n_meas + 1] *= w

    # Hermitian symmetric double-sided spectrum -> real impulse response
    full = np.concatenate([spec, np.conj(spec[-2:0:-1])])
    h = np.fft.ifft(full).real
    n_t = len(full)
    dt = 1.0 / (n_t * df)                      # = 1/(2*F_PAD_MAX) approx 1.25 ps
    t = np.arange(n_t) * dt
    return t, h, dt


# ----------------------------------------------------------------------------
# PRBS13Q PAM4 pattern
# ----------------------------------------------------------------------------
def prbs13_bits() -> np.ndarray:
    """PRBS13: x^13 + x^12 + x^2 + x + 1, length 2^13-1 = 8191 bits."""
    state = np.ones(13, dtype=int)
    bits = np.empty(8191, dtype=int)
    for i in range(8191):
        new = state[12] ^ state[11] ^ state[1] ^ state[0]
        bits[i] = state[12]
        state[1:] = state[:-1]
        state[0] = new
    return bits


def pam4_symbols() -> np.ndarray:
    """PRBS13Q: consecutive bit pairs Gray-mapped to PAM4 levels."""
    b = prbs13_bits()
    if len(b) % 2:
        b = b[:-1]
    pairs = b.reshape(-1, 2)
    gray_index = pairs[:, 0] * 2 + (pairs[:, 0] ^ pairs[:, 1])  # 00,01,11,10
    return PAM4_LEVELS[gray_index]


# ----------------------------------------------------------------------------
# Eye construction
# ----------------------------------------------------------------------------
def build_eye(sdd21: np.ndarray):
    """Return (eye_time_ps, segments, dt, metrics)."""
    t, h, dt = impulse_response(sdd21)

    # transmit waveform: symbols upsampled to dt with ideal NRZ (zero-order hold)
    sym = pam4_symbols()
    sps = UI / dt                              # samples per UI (approx 14.29)
    n_wave = int(len(sym) * sps)
    idx = np.floor(np.arange(n_wave) * dt / UI).astype(int)
    idx = np.clip(idx, 0, len(sym) - 1)
    tx = sym[idx]

    # channel: convolve with impulse response (truncate h to its energy span)
    e = np.cumsum(h ** 2)
    n_keep = int(np.searchsorted(e, 0.9999 * e[-1])) + 1
    rx = np.convolve(tx, h[:n_keep])[:n_wave]

    # fold into 2-UI eye segments, discarding the first 50 UI of transient
    seg_len = int(round(2 * UI / dt))
    start = int(50 * sps)
    n_seg = (n_wave - start) // seg_len
    segs = rx[start:start + n_seg * seg_len].reshape(n_seg, seg_len)
    t_eye = np.arange(seg_len) * dt * 1e12     # ps

    # eye metrics at the centre sampling instant (t = UI in the 2-UI window)
    k_samp = int(round(UI / dt))
    v = np.sort(segs[:, k_samp])
    # cluster into 4 rails by histogram valleys (simple quartile split works
    # for an open eye; report NaN if levels collapse)
    q = np.quantile(v, [0.125, 0.375, 0.625, 0.875])
    rails = q                                   # approximate rail centres
    heights = []
    for lo_rail, hi_rail in zip(rails[:-1], rails[1:]):
        lo_cloud = v[(v > lo_rail - 0.15) & (v < lo_rail + 0.15)]
        hi_cloud = v[(v > hi_rail - 0.15) & (v < hi_rail + 0.15)]
        if len(lo_cloud) and len(hi_cloud):
            heights.append(float(hi_cloud.min() - lo_cloud.max()))
        else:
            heights.append(float("nan"))
    metrics = {"rail_centres": [round(float(r), 4) for r in rails],
               "eye_heights": [round(h_, 4) for h_ in heights]}
    return t_eye, segs, metrics


def plot_eye(ax, t_eye, segs, title):
    step = max(1, len(segs) // 800)            # cap drawn traces
    for s in segs[::step]:
        ax.plot(t_eye, s, color="tab:blue", alpha=0.05, lw=0.7)
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("amplitude")
    ax.set_title(title)
    ax.grid(alpha=0.3)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=str)
    src.add_argument("--npz", type=str)
    ap.add_argument("--pair", type=int, default=1)
    ap.add_argument("--label", type=str, default="channel")
    # optional second channel for the side-by-side finale figure
    ap.add_argument("--compare-csv", type=str, default=None)
    ap.add_argument("--compare-npz", type=str, default=None)
    ap.add_argument("--compare-pair", type=int, default=1)
    ap.add_argument("--compare-label", type=str, default="generated")
    args = ap.parse_args()

    def load(csv, npz, pair):
        return load_sdd21_csv(csv) if csv else load_sdd21_npz(npz, pair)

    print("=" * 70)
    print(f"Stage 08: 112G PAM4 eye  (56 GBaud, UI = {UI*1e12:.3f} ps, "
          f"PRBS13Q, un-equalised)")
    print("=" * 70)

    s1 = load(args.csv, args.npz, args.pair)
    t_eye, segs, m1 = build_eye(s1)
    print(f"\n  [{args.label}] rail centres: {m1['rail_centres']}")
    print(f"  [{args.label}] eye heights (3 eyes): {m1['eye_heights']}")

    has_cmp = args.compare_csv or args.compare_npz
    if has_cmp:
        s2 = load(args.compare_csv, args.compare_npz, args.compare_pair)
        _, segs2, m2 = build_eye(s2)
        print(f"\n  [{args.compare_label}] rail centres: {m2['rail_centres']}")
        print(f"  [{args.compare_label}] eye heights: {m2['eye_heights']}")

        # ---- quantitative eye-agreement metrics (the thesis proof) ----------
        h1 = np.array([h for h in m1["eye_heights"]], float)
        h2 = np.array([h for h in m2["eye_heights"]], float)
        # total eye opening = sum of the three sub-eye heights
        tot1, tot2 = np.nansum(h1), np.nansum(h2)
        pct = 100.0 * abs(tot1 - tot2) / max(tot1, 1e-9)
        # per-eye differences
        per = np.abs(h1 - h2)
        print("\n  --- EYE AGREEMENT (target vs OpenEMS) ---")
        print(f"    total eye opening [{args.label}]  : {tot1:.4f}")
        print(f"    total eye opening [{args.compare_label}]: {tot2:.4f}")
        print(f"    total-opening difference          : {pct:.1f} %")
        print(f"    per-sub-eye |height diff|         : "
              f"{', '.join(f'{d:.4f}' for d in per)}")
        if pct < 10.0:
            print(f"    VERDICT: eyes agree within {pct:.1f}% (<10%) -- the")
            print(f"    solver difference does NOT materially change the eye.")
            print(f"    This validates the OpenEMS channel at the application")
            print(f"    level, independent of the per-frequency delta_solver.")
        else:
            print(f"    VERDICT: eyes differ by {pct:.1f}% -- the solver")
            print(f"    difference is visible in the eye; report honestly and")
            print(f"    trace it to the band where Sdd21 disagrees.")

        # shared y-limits so the two eyes are directly comparable by eye
        ymax = 1.15 * max(np.abs(segs).max(), np.abs(segs2).max())
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), tight_layout=True)
        plot_eye(axes[0], t_eye, segs, f"{args.label} (112G PAM4)")
        plot_eye(axes[1], t_eye, segs2, f"{args.compare_label} (112G PAM4)")
        for a in axes:
            a.set_ylim(-ymax, ymax)
        fig.suptitle(f"112G PAM4 eye: {args.label} vs {args.compare_label}  "
                     f"(total-opening diff {pct:.1f}%)")
        out = OUT_DIR / f"best_eye_{args.label}_vs_{args.compare_label}.png"
    else:
        fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
        plot_eye(ax, t_eye, segs, f"{args.label} (112G PAM4)")
        out = OUT_DIR / f"best_eye_{args.label}.png"

    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  figure: {out}")
    print("\n  An open eye (positive heights on all three sub-eyes) means the")
    print("  channel carries 112G PAM4 without equalisation. Matching target")
    print("  and generated eyes is the thesis finale result.")


if __name__ == "__main__":
    main()