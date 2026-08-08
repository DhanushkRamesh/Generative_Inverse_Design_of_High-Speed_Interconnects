"""
eye_diagram_validation.py
================================================================================
Stage 08 -- application-level eye-diagram comparison (112G PAM4, PRBS13Q).

WHAT THIS DOES
  Builds a 112 Gb/s PAM4 eye (56 GBaud, Nyquist ~= 28 GHz -- the eye band) from
  a via's differential insertion loss Sdd21, and compares the eye produced by
  the TARGET via against the eye produced by the GENERATED (inverse-designed)
  via. The comparison metric -- how closely the two eyes agree -- is the
  application-level proof that the generated design reproduces the target's
  signal-integrity behaviour in the time domain.

WHY A CHANNEL IS EMBEDDED
  A via in isolation is nearly transparent, so its bare eye carries little
  information (a via alone barely stresses the signal). To make the eye
  MEANINGFUL, the via is cascaded with a representative lossy transmission
  line on each side, creating realistic inter-symbol interference. Because
  the SAME channel is applied to target and generated vias, it does not bias
  the comparison -- it simply moves both into a regime where the eye is
  informative. (112G PAM4 links are equalised in practice; here we keep the
  channel light enough that a stressed-but-open eye is obtained WITHOUT
  equalisation, which is sufficient for a fidelity comparison.)

CHANNEL MODEL (first-order, standard)
  Sdd21_line(f) = exp(-(alpha(f) + j*beta(f)) * L),  applied on EACH side:
      alpha(f) : skin-effect-dominated loss, dB/inch scaling as sqrt(f)
      beta(f)  : 2*pi*f*sqrt(eps_r)/c   (phase / delay)
  Reference: transmission-line theory (Paul, Multiconductor Transmission
  Lines); eye/PAM4 methodology (Bogatin, Signal and Power Integrity).

USAGE
  # bare via (no channel) -- current behaviour:
  python3 eye_diagram_validation.py --npz TARGET.npz --label Target \
      --compare-npz DESIGN.npz --compare-pair 1 --compare-label Generated

  # WITH a representative channel (informative eye) -- tune length/loss so the
  # eye is stressed but still open (positive heights):
  python3 eye_diagram_validation.py --npz TARGET.npz --label Target \
      --compare-npz DESIGN.npz --compare-pair 1 --compare-label Generated \
      --channel-length 1.5 --channel-loss 0.7 --channel-eps 4.0

TUNING THE CHANNEL (important)
  If all eye heights are NEGATIVE, the eye is CLOSED -- reduce --channel-length
  and/or --channel-loss until the heights go positive (a stressed-but-open
  eye). If the eye is trivially wide-open, INCREASE them. Start at
  length 1.5 in, loss 0.7 dB/in and adjust.
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
    [[1, -1, 0, 0], [0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 1, 1]], float)


# ----------------------------------------------------------------------------
# Channel model: lossy transmission line
# ----------------------------------------------------------------------------
def lossy_line_sdd21(freq, length_in, loss_db_per_in_at_ghz,
                     ref_ghz=1.0, eps_r=4.0):
    """Sdd21 of a lossy differential line of `length_in` inches (PER SIDE).

    Loss scales as sqrt(f) (skin-effect-dominated) -- a standard first-order
    trace model. Returns complex Sdd21 on `freq`.
    """
    c = 2.998e8
    f = np.asarray(freq, float)
    loss_db_per_in = loss_db_per_in_at_ghz * np.sqrt(f / (ref_ghz * 1e9))
    alpha_db = loss_db_per_in * length_in            # total dB over the line
    mag = 10.0 ** (-alpha_db / 20.0)                 # linear magnitude
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
        s = (np.interp(F_GRID, f, s.real) + 1j * np.interp(F_GRID, f, s.imag))
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
# PRBS13Q PAM4 pattern
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
# Eye construction
# ----------------------------------------------------------------------------
def build_eye(sdd21: np.ndarray):
    t, h, dt = impulse_response(sdd21)
    sym = pam4_symbols()
    sps = UI / dt
    n_wave = int(len(sym) * sps)
    idx = np.floor(np.arange(n_wave) * dt / UI).astype(int)
    idx = np.clip(idx, 0, len(sym) - 1)
    tx = sym[idx]
    e = np.cumsum(h ** 2)
    n_keep = int(np.searchsorted(e, 0.9999 * e[-1])) + 1
    rx = np.convolve(tx, h[:n_keep])[:n_wave]
    seg_len = int(round(2 * UI / dt))
    start = int(50 * sps)
    n_seg = (n_wave - start) // seg_len
    segs = rx[start:start + n_seg * seg_len].reshape(n_seg, seg_len)
    t_eye = np.arange(seg_len) * dt * 1e12
    k_samp = int(round(UI / dt))
    v = np.sort(segs[:, k_samp])
    q = np.quantile(v, [0.125, 0.375, 0.625, 0.875])
    rails = q
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
    step = max(1, len(segs) // 800)
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
    args = ap.parse_args()

    def load(csv, npz, pair):
        return load_sdd21_csv(csv) if csv else load_sdd21_npz(npz, pair)

    print("=" * 70)
    print(f"Stage 08: 112G PAM4 eye  (56 GBaud, UI = {UI*1e12:.3f} ps, "
          f"PRBS13Q, un-equalised)")
    if args.channel_length > 0:
        print(f"  channel: {args.channel_length} in/side, "
              f"{args.channel_loss} dB/in@1GHz (sqrt-f), eps_r={args.channel_eps}")
    else:
        print("  channel: NONE (bare via)")
    print("=" * 70)

    # build the shared channel once (identical for target and generated)
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
        # FIX: divide by abs() so a closed (negative) eye does not blow the %
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
            print("    NOTE: eye is CLOSED under this channel -- the % is a")
            print("    fidelity measure (how equally both eyes closed), not an")
            print("    opening. Reduce --channel-length/--channel-loss for an")
            print("    OPEN eye where opening is meaningful.")
        elif pct < 10.0:
            print(f"    VERDICT: eyes agree within {pct:.1f}% (<10%) -- the")
            print(f"    generated design reproduces the target's eye. The")
            print(f"    inverse design preserves signal integrity.")
        else:
            print(f"    VERDICT: eyes differ by {pct:.1f}% -- report honestly")
            print(f"    and trace it to the band where Sdd21 disagrees.")

        ymax = 1.15 * max(np.abs(segs).max(), np.abs(segs2).max())
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), tight_layout=True)
        plot_eye(axes[0], t_eye, segs, f"{args.label} (112G PAM4)")
        plot_eye(axes[1], t_eye, segs2, f"{args.compare_label} (112G PAM4)")
        for a in axes:
            a.set_ylim(-ymax, ymax)
        fig.suptitle(f"112G PAM4 eye: {args.label} vs {args.compare_label}  "
                     f"(opening diff {pct:.1f}%)")
        out = OUT_DIR / f"best_eye_{args.label}_vs_{args.compare_label}.png"
    else:
        fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
        plot_eye(ax, t_eye, segs, f"{args.label} (112G PAM4)")
        out = OUT_DIR / f"best_eye_{args.label}.png"

    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  figure: {out}")


if __name__ == "__main__":
    main()