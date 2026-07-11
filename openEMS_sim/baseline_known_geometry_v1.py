"""
05_baseline_known.py
================================================================================
Stage 05 of the openEMS validation pipeline -- THE CRITICAL GATE.

PURPOSE
    Compare the OpenEMS full-array simulation of a KNOWN dataset geometry
    against the CONMLS ground truth, three ways, to (a) prove the whole chain
    (parser, builder, ports, mesh, mixed-mode) reproduces a known answer, and
    (b) measure delta_solver -- the CONMLS-vs-FDTD discrepancy floor that every
    later model-validation number is interpreted against.

    Nothing expensive downstream (stages 06-08) should run until this gate
    passes on at least one sim.

THE THREE PANELS
    Panel A -- pipeline audit (NO OpenEMS):
        extracted-from-raw mixed-mode pair  vs  processed .pt pair
        Already proven MATCH in stage 03b (max diff ~3.6e-08). Re-checked here
        so stage 05 is self-contained. If this fails, the data pipeline is the
        problem, not the solver.

    Panel B -- purest solver delta (full 16-port, single-ended):
        OpenEMS 16-port S  vs  raw .s16p
        Both are the complete array, no extraction. This is the cleanest
        CONMLS-vs-FDTD comparison. Reported per representative port.

    Panel C -- thesis-relevant delta_solver (mixed-mode pair):
        OpenEMS mixed-mode pair  vs  processed .pt pair
        This is the quantity the inverse model consumes and the eye diagram
        depends on. delta_solver(band) comes from here.

DIAGNOSTIC TREE (why three panels)
    A fails                 -> pipeline bug (extraction/normalization)
    A ok, B fails           -> OpenEMS geometry / mesh / port bug
    A ok, B ok, C fails     -> mixed-mode conversion / port-ordering mismatch
    all pass within delta   -> gate PASSED, geometry model validated

INPUTS
    - OpenEMS single-ended S-matrix: results/04_openems/{sim}_openems_se.npz
      (produced by stage 04 --run)
    - Raw CONMLS touchstone: data/raw/.../variation/{sim}/{sim}.s16p
    - Processed pairs: data/processed/.../diff_pair_dataset.pt

OUTPUTS
    - results/05_baseline/{sim}_panelA.csv / _panelB.csv / _panelC.csv
    - results/05_baseline/{sim}_overlay.png  (Sdd11/Sdd21, all panels)
    - delta_solver(band) table printed + saved to {sim}_delta_solver.csv

USAGE
    # after stage 04 --run has produced the .npz:
    python 05_baseline_known.py --sim sim_pkg_0017 --pair 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import skrf as rf

# ----------------------------------------------------------------------------
# Project paths
# ----------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (
    Path.home()
    / "mece_project_inverse_model"
    / "Generative_Inverse_Design_of_High-Speed_Interconnects"
)
RAW_ARRAY_DIR = PROJECT_ROOT / "data" / "raw" / "Universal-Diff-SI-Array"
PROCESSED_PT = (
    PROJECT_ROOT / "data" / "processed" / "Universal-Diff-SI-Array"
    / "diff_pair_dataset.pt"
)
OPENEMS_NPZ_DIR = _THIS_DIR / "results" / "04_openems"
RESULTS_DIR = _THIS_DIR / "results" / "05_baseline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Frequency grid + PAM4 bands (matches the processed dataset)
# ----------------------------------------------------------------------------
FREQ_MIN_HZ = 0.25e9
FREQ_MAX_HZ = 100e9
FREQ_N_POINTS = 401
TARGET_FREQ_HZ = np.linspace(FREQ_MIN_HZ, FREQ_MAX_HZ, FREQ_N_POINTS)

# Bands for delta_solver reporting (GHz). Nyquist for 112G PAM4 ~ 28 GHz.
BANDS_GHZ = [
    ("0-14 GHz",   0.0,  14.0),
    ("14-28 GHz",  14.0, 28.0),
    ("28-56 GHz",  28.0, 56.0),
    ("56-100 GHz", 56.0, 100.0),
]

# ----------------------------------------------------------------------------
# Bockelman-Eisenstadt transform (identical to physics_utils.py)
# ----------------------------------------------------------------------------
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
    return M_BE @ s_se @ M_BE.T


def enforce_reciprocity(s: np.ndarray) -> np.ndarray:
    return 0.5 * (s + np.transpose(s, (0, 2, 1)))


def diff_pair_port_indices_array(pair_k: int, num_ports: int) -> list[int]:
    base = 4 * (pair_k - 1)
    idx = [base, base + 2, base + 1, base + 3]
    if max(idx) >= num_ports:
        raise ValueError(f"pair {pair_k} out of range for {num_ports} ports")
    return idx


def db(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.abs(x) + 1e-12)


def interp_to_target(f_native: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Interpolate an (F,P,P) S-matrix onto the target grid (no extrapolation)."""
    P = s.shape[1]
    target = np.clip(TARGET_FREQ_HZ, f_native.min(), f_native.max())
    out = np.empty((FREQ_N_POINTS, P, P), dtype=complex)
    for i in range(P):
        for j in range(P):
            out[:, i, j] = (np.interp(target, f_native, s[:, i, j].real)
                            + 1j * np.interp(target, f_native, s[:, i, j].imag))
    return out


# ============================================================================
# Data loaders
# ============================================================================
def load_openems_se(sim_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the OpenEMS single-ended S-matrix from stage 04's npz."""
    npz = OPENEMS_NPZ_DIR / f"{sim_id}_openems_se.npz"
    if not npz.exists():
        raise FileNotFoundError(
            f"OpenEMS result not found: {npz}\n"
            f"Run stage 04 first:  python 04_build_array_model.py "
            f"--sim {sim_id} --run"
        )
    d = np.load(npz)
    freq, S = d["freq"], d["S"]
    # Ensure it is on the target grid
    if S.shape[0] != FREQ_N_POINTS or not np.allclose(freq, TARGET_FREQ_HZ):
        S = interp_to_target(freq, S)
    return TARGET_FREQ_HZ, S


def load_raw_touchstone(sim_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the raw CONMLS .s16p, return (freq_hz, S) on the target grid."""
    sim_dir = RAW_ARRAY_DIR / "variation" / sim_id
    matches = sorted(sim_dir.glob("*.s*p"))
    if not matches:
        raise FileNotFoundError(f"No touchstone in {sim_dir}")
    ntwk = rf.Network(str(matches[0]))
    S = interp_to_target(ntwk.f, ntwk.s.astype(complex))
    return TARGET_FREQ_HZ, S


def load_processed_pair(sim_id: str, pair_k: int) -> np.ndarray | None:
    """Load the stored mixed-mode pair from diff_pair_dataset.pt, or None."""
    try:
        import torch
    except ImportError:
        print("  [warn] torch not importable; cannot load processed .pt")
        return None
    if not PROCESSED_PT.exists():
        print(f"  [warn] {PROCESSED_PT} not found")
        return None
    payload = torch.load(PROCESSED_PT, weights_only=False, map_location="cpu")
    sim_ids = list(payload["sim_ids"])
    pair_ids = payload["pair_ids"].numpy()
    matches = [i for i, (s, p) in enumerate(zip(sim_ids, pair_ids))
               if s == sim_id and int(p) == pair_k]
    if not matches:
        avail = sorted(int(pair_ids[i]) for i, s in enumerate(sim_ids)
                       if s == sim_id)
        print(f"  [warn] sim {sim_id} pair {pair_k} not in processed dataset. "
              f"Available: {avail}")
        return None
    row = matches[0]
    return (payload["Y_real"][row].numpy()
            + 1j * payload["Y_imag"][row].numpy())


# ============================================================================
# Extraction (mirrors the pipeline exactly)
# ============================================================================
def extract_mixed_mode_pair(S_se_full: np.ndarray, pair_k: int) -> np.ndarray:
    """From a full (F,N,N) single-ended S, extract pair k as mixed-mode (F,4,4)."""
    num_ports = S_se_full.shape[1]
    idx = diff_pair_port_indices_array(pair_k, num_ports)
    s_se = S_se_full[:, idx][:, :, idx].astype(complex)
    s_se = enforce_reciprocity(s_se)
    return convert_to_mixed_mode(s_se)


# ============================================================================
# delta_solver reporting
# ============================================================================
def band_stats(freq_hz: np.ndarray, s_a: np.ndarray, s_b: np.ndarray,
               label: str) -> list[dict]:
    """Per-band mean/max |dB difference| between two complex traces."""
    f_ghz = freq_hz / 1e9
    da = db(s_a)
    dbb = db(s_b)
    diff = np.abs(da - dbb)
    rows = []
    for name, lo, hi in BANDS_GHZ:
        m = (f_ghz >= lo) & (f_ghz < hi)
        if not np.any(m):
            continue
        rows.append({
            "trace": label, "band": name,
            "mean_dB": float(diff[m].mean()),
            "max_dB": float(diff[m].max()),
        })
    return rows


# ============================================================================
# Panels
# ============================================================================
def panel_A(sim_id: str, pair_k: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Pipeline audit: extracted-from-raw vs processed .pt. No OpenEMS."""
    print("\n--- PANEL A: pipeline audit (extracted-from-raw vs processed) ---")
    _, S_raw = load_raw_touchstone(sim_id)
    mm_extracted = extract_mixed_mode_pair(S_raw, pair_k)
    mm_stored = load_processed_pair(sim_id, pair_k)
    if mm_stored is None:
        print("  [skip] processed pair unavailable")
        return mm_extracted, None
    diff = np.abs(mm_extracted - mm_stored)
    print(f"  max |diff| over 4x4xF : {diff.max():.3e}")
    print(f"  max |diff| Sdd11      : "
          f"{np.abs(mm_extracted[:,0,0]-mm_stored[:,0,0]).max():.3e}")
    print(f"  max |diff| Sdd21      : "
          f"{np.abs(mm_extracted[:,1,0]-mm_stored[:,1,0]).max():.3e}")
    if diff.max() < 1e-4:
        print("  VERDICT: MATCH (pipeline verified)")
    else:
        print("  VERDICT: MISMATCH -- investigate extraction before trusting C")
    return mm_extracted, mm_stored


def panel_B(sim_id: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Purest solver delta: OpenEMS 16-port vs raw .s16p (single-ended)."""
    print("\n--- PANEL B: purest solver delta (OpenEMS 16-port vs raw .s16p) ---")
    freq, S_oe = load_openems_se(sim_id)
    _, S_raw = load_raw_touchstone(sim_id)
    N = min(S_oe.shape[1], S_raw.shape[1])
    if S_oe.shape[1] != S_raw.shape[1]:
        print(f"  [warn] port count differs: OpenEMS {S_oe.shape[1]} vs "
              f"raw {S_raw.shape[1]}; comparing first {N}")

    # Report a few representative single-ended terms: S11, S21 (through the
    # first via pair), and a cross-coupling term.
    rows = []
    for (i, j, name) in [(0, 0, "S11"), (1, 0, "S21"),
                          (2, 0, "S31"), (0, 2, "S13")]:
        if i < N and j < N:
            rows += band_stats(freq, S_oe[:, i, j], S_raw[:, i, j],
                               f"SE_{name}")
    for r in rows:
        print(f"    {r['trace']:8s} {r['band']:10s} "
              f"mean={r['mean_dB']:6.2f} dB  max={r['max_dB']:6.2f} dB")
    return S_oe, S_raw, rows


def panel_C(sim_id: str, pair_k: int, S_oe: np.ndarray,
            mm_stored: np.ndarray | None
            ) -> tuple[np.ndarray, np.ndarray | None, list[dict]]:
    """Thesis-relevant delta_solver: OpenEMS mixed-mode pair vs processed."""
    print("\n--- PANEL C: delta_solver (OpenEMS mixed-mode pair vs processed) ---")
    mm_oe = extract_mixed_mode_pair(S_oe, pair_k)
    rows = []
    if mm_stored is None:
        print("  [skip] processed pair unavailable; reporting OpenEMS pair only")
        return mm_oe, None, rows
    for (i, j, name) in [(0, 0, "Sdd11"), (1, 0, "Sdd21"),
                          (2, 2, "Scc11"), (1, 2, "Scd21")]:
        rows += band_stats(TARGET_FREQ_HZ, mm_oe[:, i, j], mm_stored[:, i, j],
                           name)
    for r in rows:
        print(f"    {r['trace']:8s} {r['band']:10s} "
              f"mean={r['mean_dB']:6.2f} dB  max={r['max_dB']:6.2f} dB")
    return mm_oe, mm_stored, rows


# ============================================================================
# Plot
# ============================================================================
def make_overlay(sim_id: str, pair_k: int,
                 mm_oe: np.ndarray, mm_stored: np.ndarray | None,
                 mm_extracted: np.ndarray | None) -> Path:
    f_ghz = TARGET_FREQ_HZ / 1e9
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), tight_layout=True)

    for ax, (i, j, title) in zip(axes, [(0, 0, "Sdd11"), (1, 0, "Sdd21")]):
        if mm_stored is not None:
            ax.plot(f_ghz, db(mm_stored[:, i, j]), "k-", lw=2,
                    label="CONMLS (processed)")
        if mm_extracted is not None:
            ax.plot(f_ghz, db(mm_extracted[:, i, j]), "g:", lw=1.5,
                    label="CONMLS (extracted-from-raw)")
        ax.plot(f_ghz, db(mm_oe[:, i, j]), "r--", lw=2, label="OpenEMS")
        for _, lo, hi in BANDS_GHZ:
            ax.axvspan(lo, hi, alpha=0.04,
                       color="blue" if (lo // 28) % 2 == 0 else "orange")
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(f"{title}  ({sim_id}, pair {pair_k})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, 100)

    out = RESULTS_DIR / f"{sim_id}_pair{pair_k}_overlay.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ============================================================================
# Gate evaluation
# ============================================================================
def evaluate_gate(panelA_ok: bool, panelC_rows: list[dict]) -> bool:
    """Stage-05 gate: A matches, and delta_solver <= ~3 dB below 56 GHz on the
    primary paths (Sdd11, Sdd21).
    """
    print("\n" + "=" * 78)
    print("STAGE 05 GATE")
    print("=" * 78)
    ok = True

    if not panelA_ok:
        print("  [FAIL] Panel A pipeline audit did not match.")
        ok = False
    else:
        print("  [pass] Panel A pipeline audit matches.")

    # delta_solver criterion on Sdd11/Sdd21 below 56 GHz
    THRESH = 3.0
    primary = [r for r in panelC_rows
               if r["trace"] in ("Sdd11", "Sdd21")
               and r["band"] in ("0-14 GHz", "14-28 GHz", "28-56 GHz")]
    if not primary:
        print("  [warn] no Panel C data to evaluate delta_solver")
    else:
        worst = max(primary, key=lambda r: r["mean_dB"])
        print(f"  delta_solver (Sdd11/Sdd21, <56 GHz): worst mean = "
              f"{worst['mean_dB']:.2f} dB in {worst['band']} "
              f"({worst['trace']})   [threshold {THRESH} dB]")
        if worst["mean_dB"] > THRESH:
            print(f"  [FAIL] delta_solver exceeds {THRESH} dB below 56 GHz.")
            print("         Likely causes (check in this order):")
            print("           1. port numbering / mixed-mode mapping")
            print("           2. mesh density (under-resolved at high freq)")
            print("           3. de-normalization / stackup thickness")
            print("           4. port reference (systematic level offset)")
            ok = False
        else:
            print(f"  [pass] delta_solver within {THRESH} dB below 56 GHz.")

    print("\n  " + ("GATE PASSED -- geometry model validated; proceed to "
                    "stage 06." if ok else
                    "GATE NOT PASSED -- debug before scaling to more sims."))
    print("=" * 78)
    return ok


# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", type=str, default="sim_pkg_0017")
    ap.add_argument("--pair", type=int, default=1)
    ap.add_argument("--no-openems", action="store_true",
                    help="run only Panel A (no OpenEMS result needed)")
    args = ap.parse_args()

    print("=" * 78)
    print(f"Stage 05: baseline known-geometry comparison "
          f"({args.sim}, pair {args.pair})")
    print("=" * 78)

    # Panel A (always; no OpenEMS)
    mm_extracted, mm_stored = panel_A(args.sim, args.pair)
    panelA_ok = (mm_stored is not None
                 and np.abs(mm_extracted - mm_stored).max() < 1e-4)

    if args.no_openems:
        print("\n[--no-openems] Skipping Panels B/C. Run stage 04 --run, then "
              "re-run without --no-openems.")
        return

    # Panels B and C (need the OpenEMS result)
    try:
        S_oe, S_raw, rowsB = panel_B(args.sim)
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nPanel A done; Panels B/C need the OpenEMS run. "
              "Run stage 04 --run first.")
        return

    mm_oe, _, rowsC = panel_C(args.sim, args.pair, S_oe, mm_stored)

    # Save CSVs
    import csv
    def _write(rows, name):
        if not rows:
            return
        p = RESULTS_DIR / f"{args.sim}_pair{args.pair}_{name}.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  saved {p.name}")
    _write(rowsB, "panelB")
    _write(rowsC, "panelC")

    # Overlay figure
    out = make_overlay(args.sim, args.pair, mm_oe, mm_stored, mm_extracted)
    print(f"\n  overlay figure: {out}")

    # Gate
    evaluate_gate(panelA_ok, rowsC)


if __name__ == "__main__":
    main()