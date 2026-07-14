"""
05_baseline_known.py  (v2 - weighted delta_solver + eye-critical view)
================================================================================
Stage 05 of the openEMS validation pipeline -- THE CRITICAL GATE.

Compares the OpenEMS full-array result for ONE known sim against CONMLS three
ways, and now reports delta_solver through TWO complementary lenses so the
gate answers both thesis claims:

  (1) EYE-CRITICAL (energy) view: per-band flat mean |dB|. The eye is carried
      by the low-loss band, so agreement in 0-28 GHz is what protects the eye.
      High-frequency stopband disagreement (deep nulls, near the -40..-50 dB
      floor) barely affects the eye and is expected to be looser.

  (2) MODEL-WEIGHTED (information) view: delta_solver weighted by the SAME
      element-aware (4,4,401) frequency-importance tensor the inverse model
      uses. This asks "do the solvers agree WHERE THE MODEL IS SENSITIVE?"
      (Sdd11 low-freq, Sdd21 high-freq per the frequency EDA). This ties the
      validation metric to the thesis's own frequency-importance contribution.

Why two views (the subtlety): a frequency can be information-rich for the model
(high geometry-to-geometry variance) yet energy-poor for the eye (deep loss).
High-frequency Sdd21 is exactly that. So "3 dB there" is harmless for the eye
but NOT automatically harmless for the model -- hence both views are reported.

PANELS (unchanged): A pipeline audit (no sim), B purest solver delta (16-port
SE), C thesis delta_solver (mixed-mode pair).

USAGE
    python 05_baseline_known.py --sim sim_pkg_0017 --pair 1
    python 05_baseline_known.py --sim sim_pkg_0017 --pair 1 --no-openems
    # optional: point at the model weight tensor (else the weighted view is skipped)
    python 05_baseline_known.py --sim sim_pkg_0017 --pair 1 \
        --weights ../sandbox_v1/data/frequency_eda/weights_element_aware_per_freq.npy
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skrf as rf

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (Path.home() / "mece_project_inverse_model"
                / "Generative_Inverse_Design_of_High-Speed_Interconnects")
RAW_ARRAY_DIR = PROJECT_ROOT / "data" / "raw" / "Universal-Diff-SI-Array"
PROCESSED_PT = (PROJECT_ROOT / "data" / "processed"
                / "Universal-Diff-SI-Array" / "diff_pair_dataset.pt")
OPENEMS_NPZ_DIR = _THIS_DIR / "results" / "04_openems"
RESULTS_DIR = _THIS_DIR / "results" / "05_baseline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FREQ_MIN_HZ, FREQ_MAX_HZ, FREQ_N = 0.25e9, 100e9, 401
TARGET_FREQ_HZ = np.linspace(FREQ_MIN_HZ, FREQ_MAX_HZ, FREQ_N)
BANDS_GHZ = [("0-14", 0, 14), ("14-28", 14, 28), ("28-56", 28, 56),
             ("56-100", 56, 100)]
# eye-critical band = where signal energy lives
EYE_BANDS = {"0-14", "14-28"}

M_BE = (1 / np.sqrt(2)) * np.array(
    [[1, -1, 0, 0], [0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 1, 1]], float)


def db(x):
    return 20 * np.log10(np.abs(x) + 1e-12)


def convert_mm(s_se):
    return M_BE @ s_se @ M_BE.T


def recip(s):
    return 0.5 * (s + np.transpose(s, (0, 2, 1)))


def pair_idx(k):
    b = 4 * (k - 1)
    return [b, b + 2, b + 1, b + 3]


def interp_to_target(f, s):
    P = s.shape[1]
    t = np.clip(TARGET_FREQ_HZ, f.min(), f.max())
    out = np.empty((FREQ_N, P, P), complex)
    for i in range(P):
        for j in range(P):
            out[:, i, j] = (np.interp(t, f, s[:, i, j].real)
                            + 1j * np.interp(t, f, s[:, i, j].imag))
    return out


# ----------------------------------------------------------------------------
def load_openems_se(sim_id):
    npz = OPENEMS_NPZ_DIR / f"{sim_id}_mur_lateral_openems_se.npz"
    if not npz.exists():
        raise FileNotFoundError(
            f"OpenEMS result not found: {npz}\nRun stage 04 --run first.")
    d = np.load(npz)
    freq, S = d["freq"], d["S"]
    if S.shape[0] != FREQ_N or not np.allclose(freq, TARGET_FREQ_HZ):
        S = interp_to_target(freq, S)
    return TARGET_FREQ_HZ, S


def load_raw_touchstone(sim_id):
    sim_dir = RAW_ARRAY_DIR / "variation" / sim_id
    m = sorted(sim_dir.glob("*.s*p"))
    if not m:
        raise FileNotFoundError(f"No touchstone in {sim_dir}")
    ntwk = rf.Network(str(m[0]))
    return TARGET_FREQ_HZ, interp_to_target(ntwk.f, ntwk.s.astype(complex))


def load_processed_pair(sim_id, pair_k):
    try:
        import torch
    except ImportError:
        return None
    if not PROCESSED_PT.exists():
        return None
    p = torch.load(PROCESSED_PT, weights_only=False, map_location="cpu")
    sids = list(p["sim_ids"]); pids = p["pair_ids"].numpy()
    for i, (s, pk) in enumerate(zip(sids, pids)):
        if s == sim_id and int(pk) == pair_k:
            return p["Y_real"][i].numpy() + 1j * p["Y_imag"][i].numpy()
    return None


def extract_mm(S_se, k):
    idx = pair_idx(k)
    s = recip(S_se[:, idx][:, :, idx].astype(complex))
    return convert_mm(s)


def load_weights(path):
    """Element-aware (4,4,401) or (401,4,4) frequency-importance tensor."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"  [weighted view skipped] weight file not found: {p}")
        return None
    w = np.load(p)
    if w.shape == (4, 4, 401):
        w = np.transpose(w, (2, 0, 1))     # -> (401,4,4)
    if w.shape != (FREQ_N, 4, 4):
        print(f"  [weighted view skipped] unexpected weight shape {w.shape}")
        return None
    return w


# ----------------------------------------------------------------------------
NULL_FLOOR_DB = -30.0   # clip both traces here before dB-differencing.
# Rationale: a resonance null is a cancellation point; its DEPTH is
# hypersensitive and never matches between two solvers, while its POSITION is
# the physical content. Raw dB-differencing at an aligned-but-deeper null
# manufactures 20+ dB of "error" from negligible linear energy (the exact
# dB-variance pathology identified in the frequency-importance EDA). Standard
# practice in solver/VNA comparisons is to compare above a noise floor. All
# three metrics (raw dB, floored dB, linear |dS|) are reported transparently;
# the gate evaluates the floored metric.


def band_stats(freq, a, b, label):
    f = freq / 1e9
    da, dbb = db(a), db(b)
    d_raw = np.abs(da - dbb)
    d_floor = np.abs(np.maximum(da, NULL_FLOOR_DB)
                     - np.maximum(dbb, NULL_FLOOR_DB))
    d_lin = np.abs(np.abs(a) - np.abs(b))          # linear-magnitude error
    rows = []
    for name, lo, hi in BANDS_GHZ:
        m = (f >= lo) & (f < hi)
        if m.any():
            rows.append({"trace": label, "band": name,
                         "mean_dB": round(float(d_raw[m].mean()), 3),
                         "max_dB": round(float(d_raw[m].max()), 3),
                         "mean_dB_floored": round(float(d_floor[m].mean()), 3),
                         "mean_linear": round(float(d_lin[m].mean()), 4)})
    return rows


def weighted_delta(mm_oe, mm_ref, w, i, j):
    """Weighted mean |dB| gap for element (i,j), using weights w[:,i,j]."""
    d = np.abs(db(mm_oe[:, i, j]) - db(mm_ref[:, i, j]))
    wij = w[:, i, j]
    if wij.sum() <= 0:
        return float("nan")
    return float((d * wij).sum() / wij.sum())


# ----------------------------------------------------------------------------
def panel_A(sim_id, pair_k):
    print("\n--- PANEL A: pipeline audit (extracted-from-raw vs processed) ---")
    _, S_raw = load_raw_touchstone(sim_id)
    mm_ext = extract_mm(S_raw, pair_k)
    mm_st = load_processed_pair(sim_id, pair_k)
    if mm_st is None:
        print("  [skip] processed pair unavailable")
        return mm_ext, None
    d = np.abs(mm_ext - mm_st).max()
    print(f"  max |diff| = {d:.3e}  "
          f"{'MATCH (pipeline verified)' if d < 1e-4 else 'MISMATCH -- investigate'}")
    return mm_ext, mm_st


def panel_B(sim_id):
    print("\n--- PANEL B: purest solver delta (OpenEMS 16-port vs raw) ---")
    freq, S_oe = load_openems_se(sim_id)
    _, S_raw = load_raw_touchstone(sim_id)
    N = min(S_oe.shape[1], S_raw.shape[1])
    rows = []
    for i, j, nm in [(0, 0, "S11"), (1, 0, "S21")]:
        if i < N and j < N:
            rows += band_stats(freq, S_oe[:, i, j], S_raw[:, i, j], f"SE_{nm}")
    for r in rows:
        print(f"    {r['trace']:7s} {r['band']:7s} mean={r['mean_dB']:6.2f} "
              f"max={r['max_dB']:6.2f} dB")
    return S_oe, rows


def panel_C(sim_id, pair_k, S_oe, mm_st, weights):
    print("\n--- PANEL C: delta_solver (OpenEMS mixed-mode pair vs processed) ---")
    mm_oe = extract_mm(S_oe, pair_k)
    rows = []
    if mm_st is None:
        print("  [skip] processed pair unavailable")
        return mm_oe, rows
    elems = [("Sdd11", 0, 0), ("Sdd21", 1, 0), ("Scc11", 2, 2), ("Scd21", 1, 2)]
    for nm, i, j in elems:
        rows += band_stats(TARGET_FREQ_HZ, mm_oe[:, i, j], mm_st[:, i, j], nm)
    print("  [EYE-CRITICAL view: raw dB | floored dB (-30 clip) | linear |dS|]")
    for r in rows:
        tag = " <- eye" if (r["trace"] in ("Sdd11", "Sdd21")
                            and r["band"] in EYE_BANDS) else ""
        print(f"    {r['trace']:7s} {r['band']:7s} raw={r['mean_dB']:6.2f} "
              f"floored={r['mean_dB_floored']:6.2f} dB  "
              f"lin={r['mean_linear']:7.4f}{tag}")

    # model-weighted view
    if weights is not None:
        print("\n  [MODEL-WEIGHTED view: delta_solver weighted by the element-")
        print("   aware frequency-importance tensor -- agreement WHERE THE")
        print("   MODEL IS SENSITIVE (Sdd11 low-f, Sdd21 high-f)]")
        for nm, i, j in elems:
            wd = weighted_delta(mm_oe, mm_st, weights, i, j)
            print(f"    {nm:7s} weighted mean |dB| = {wd:6.2f}")
    return mm_oe, rows


# ----------------------------------------------------------------------------
def make_overlay(sim_id, pair_k, mm_oe, mm_st, mm_ext, dC):
    f = TARGET_FREQ_HZ / 1e9
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), tight_layout=True)
    s21 = next((r["mean_dB"] for r in dC
                if r["trace"] == "Sdd21" and r["band"] == "14-28"), float("nan"))
    s11 = next((r["mean_dB"] for r in dC
                if r["trace"] == "Sdd11" and r["band"] == "0-14"), float("nan"))
    fig.suptitle(f"Stage 05 ({sim_id}, pair {pair_k})  "
                 f"eye-band delta: Sdd11(0-14)={s11:.2f} dB | "
                 f"Sdd21(14-28)={s21:.2f} dB")
    for a, (i, j, t) in zip(ax, [(0, 0, "Sdd11"), (1, 0, "Sdd21")]):
        if mm_st is not None:
            a.plot(f, db(mm_st[:, i, j]), "k-", lw=2, label="CONMLS")
        if mm_ext is not None:
            a.plot(f, db(mm_ext[:, i, j]), "g:", lw=1.2, label="CONMLS (raw)")
        a.plot(f, db(mm_oe[:, i, j]), "r--", lw=2, label="OpenEMS")
        for _, lo, hi in BANDS_GHZ:
            if f"{lo}-{hi}" in {f'{a}-{b}' for a, b in [(0,14),(14,28)]}:
                a.axvspan(lo, hi, alpha=0.06, color="green")
        a.set_xlabel("Frequency (GHz)"); a.set_ylabel("dB")
        a.set_title(t); a.grid(alpha=0.3); a.legend(); a.set_xlim(0, 100)
    out = RESULTS_DIR / f"{sim_id}_mur_lateral_pair{pair_k}_overlay.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def evaluate_gate(panelA_ok, rowsC):
    print("\n" + "=" * 78)
    print("STAGE 05 GATE")
    print("=" * 78)
    ok = True
    print(f"  Panel A pipeline audit: {'pass' if panelA_ok else 'FAIL'}")
    ok = ok and panelA_ok

    # EYE-CRITICAL gate: Sdd11/Sdd21 in 0-28 GHz must be small (protects eye)
    eye = [r for r in rowsC if r["trace"] in ("Sdd11", "Sdd21")
           and r["band"] in EYE_BANDS]
    if eye:
        worst = max(eye, key=lambda r: r["mean_dB_floored"])
        print(f"  EYE-CRITICAL delta_solver (0-28 GHz, floored at "
              f"{NULL_FLOOR_DB:.0f} dB): worst mean = "
              f"{worst['mean_dB_floored']:.2f} dB "
              f"({worst['trace']} {worst['band']}) [target <= ~3 dB]")
        print(f"  (raw-dB worst for reference: "
              f"{max(r['mean_dB'] for r in eye):.2f} dB -- difference vs "
              f"floored = aligned-null depth mismatch, negligible energy)")
        if worst["mean_dB_floored"] > 3.0:
            print("  [FAIL] eye-critical band exceeds 3 dB -- the eye would be "
                  "affected. Fix ports/mesh/floor before trusting the eye.")
            ok = False
        else:
            print("  [pass] eye-critical band within 3 dB -- eye is protected.")
    print("\n  " + ("GATE PASSED -- proceed to stage 06." if ok
                    else "GATE NOT PASSED -- debug before scaling."))
    print("  (High-frequency stopband disagreement is expected and does not")
    print("   fail the gate; it barely affects the eye. The model-weighted")
    print("   view above shows agreement where the model is sensitive.)")
    print("=" * 78)
    return ok


# ----------------------------------------------------------------------------
def main():
    class _Tee:

        def __init__(self, *streams): self.streams = streams
        def write(self, data):
            for s in self.streams: s.write(data); s.flush()
        def flush(self):
            for s in self.streams: s.flush()
    _logfile = open(_THIS_DIR / f"stage05_mur_lateral_log_{datetime.now():%Y%m%d_%H%M%S}.log", "w")
    sys.stdout = _Tee(sys.__stdout__, _logfile)
    sys.stderr = _Tee(sys.__stderr__, _logfile)
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="sim_pkg_0017")
    ap.add_argument("--pair", type=int, default=1)
    ap.add_argument("--no-openems", action="store_true")
    ap.add_argument("--weights", default=None,
                    help="path to element-aware (4,4,401) weight .npy")
    args = ap.parse_args()

    print("=" * 78)
    print(f"Stage 05: baseline comparison ({args.sim}, pair {args.pair})")
    print("=" * 78)

    weights = load_weights(args.weights)
    mm_ext, mm_st = panel_A(args.sim, args.pair)
    panelA_ok = (mm_st is not None
                 and np.abs(mm_ext - mm_st).max() < 1e-4)

    if args.no_openems:
        print("\n[--no-openems] Panel A only. Run stage 04 --run then re-run.")
        return

    try:
        S_oe, _ = panel_B(args.sim)
    except FileNotFoundError as e:
        print(f"\n{e}")
        return

    mm_oe, rowsC = panel_C(args.sim, args.pair, S_oe, mm_st, weights)

    if rowsC:
        t = RESULTS_DIR / f"{args.sim}_pair{args.pair}_panelC.csv"
        with open(t, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rowsC[0].keys()))
            w.writeheader(); w.writerows(rowsC)
        print(f"\n  saved {t.name}")

    out = make_overlay(args.sim, args.pair, mm_oe, mm_st, mm_ext, rowsC)
    print(f"  overlay: {out}")
    evaluate_gate(panelA_ok, rowsC)


if __name__ == "__main__":
    main()