"""
03_touchstone_diagnostic.py
================================================================================
Stage 03b of the openEMS validation pipeline (pre-simulation diagnostic).

PURPOSE
    Read the raw CONMLS Touchstone file for a known sim, extract its
    differential pairs using the EXACT pipeline convention (port indices +
    Bockelman-Eisenstadt transform), and do three things -- all with zero
    OpenEMS involvement:

      1. COMPARISON A (extraction audit): compare the mixed-mode pair we
         extract here from the raw .s16p against the same pair stored in the
         processed diff_pair_dataset.pt. If they match, the whole extraction
         chain (port selection, reciprocity, mixed-mode transform, frequency
         interpolation) is verified independent of any solver. This is the
         pipeline-audit panel of stage 05, runnable now.

      2. PORT-REFERENCE DIAGNOSIS: print the low- and mid-frequency mixed-mode
         signatures (Sdd11, Sdd21, Scc11, common/diff isolation). The
         low-frequency behaviour reveals how CONMLS referenced its ports,
         which tells stage 04 how to define the OpenEMS lumped ports. A
         passive differential through-via referenced single-ended-to-ground
         has Sdd21(f->0) ~ 0 dB and Sdd11(f->0) deep; deviations from that
         signature flag a different port reference.

      3. GROUND-TRUTH EXPORT: save the extracted mixed-mode pair (Sdd11,
         Sdd21) to CSV so stage 05 has the exact target curves the OpenEMS
         build must reproduce.

REQUIREMENTS
    Reuses the user's own extraction convention verbatim:
      - port indices: [4(k-1), 4(k-1)+2, 4(k-1)+1, 4(k-1)+3]  (Array)
      - mixed-mode:   M_BE @ S_se @ M_BE.T  with [TX+,TX-,RX+,RX-] ordering
      - reciprocity:  (S + S^T)/2 before mixed-mode
    These are copied from physics_utils.py / parse_via_array.py so this script
    is self-contained (no import from sandbox_v1 required).

USAGE
    cd ~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/openEMS_Sim
    python 03b_touchstone_diagnostic.py --sim sim_pkg_0017
    python 03b_touchstone_diagnostic.py --sim sim_pkg_0017 --pair 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Touchstone reading via scikit-rf (already verified installed in stage 01)
import skrf as rf

# ----------------------------------------------------------------------------
# Project paths
# ----------------------------------------------------------------------------
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
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "03b_touchstone"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# The processed dataset's frequency grid (parse_diff_pairs PATCH 1)
FREQ_MIN_HZ = 0.25e9
FREQ_MAX_HZ = 100e9
FREQ_N_POINTS = 401
TARGET_FREQ_HZ = np.linspace(FREQ_MIN_HZ, FREQ_MAX_HZ, FREQ_N_POINTS)

# ----------------------------------------------------------------------------
# Bockelman-Eisenstadt transform (copied verbatim from physics_utils.py)
# input ports  [TX+, TX-, RX+, RX-]  ->  output [d_TX, d_RX, c_TX, c_RX]
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
    """(F,4,4) single-ended [TX+,TX-,RX+,RX-] -> mixed-mode [dTX,dRX,cTX,cRX]."""
    return M_BE @ s_se @ M_BE.T


def enforce_reciprocity(s: np.ndarray) -> np.ndarray:
    """(S + S^T)/2 per frequency."""
    return 0.5 * (s + np.transpose(s, (0, 2, 1)))


def diff_pair_port_indices_array(pair_k: int, num_ports: int) -> list[int]:
    """0-based [TX+, TX-, RX+, RX-] for Array pair k (1-indexed)."""
    base = 4 * (pair_k - 1)
    idx = [base, base + 2, base + 1, base + 3]
    if max(idx) >= num_ports:
        raise ValueError(f"pair {pair_k} out of range for {num_ports} ports")
    return idx


def db(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.abs(x) + 1e-12)


# ----------------------------------------------------------------------------
# Load + extract
# ----------------------------------------------------------------------------
def load_touchstone(sim_id: str) -> rf.Network:
    """Load the raw .s16p (or .sNp) for a sim as a scikit-rf Network."""
    sim_dir = RAW_ARRAY_DIR / "variation" / sim_id
    matches = sorted(sim_dir.glob("*.s*p"))
    if not matches:
        raise FileNotFoundError(f"No touchstone file in {sim_dir}")
    ntwk = rf.Network(str(matches[0]))
    return ntwk


def extract_pair_mixed_mode(
    ntwk: rf.Network, pair_k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Extract one differential pair, return (freq_hz, s_mm) on the TARGET grid.

    Mirrors parse_diff_pairs.py exactly:
      1. slice the 4 single-ended ports for pair k
      2. enforce reciprocity
      3. Bockelman-Eisenstadt mixed-mode transform
      4. interpolate onto the 0.25..100 GHz / 401-pt grid (no extrapolation)
    """
    num_ports = ntwk.nports
    idx = diff_pair_port_indices_array(pair_k, num_ports)

    # scikit-rf S is (F, P, P); slice the 4x4 sub-block for our ports
    s_full = ntwk.s
    s_se = s_full[:, idx][:, :, idx].astype(np.complex128)
    s_se = enforce_reciprocity(s_se)
    s_mm_native = convert_to_mixed_mode(s_se)   # on the file's native grid

    # Interpolate onto the target grid (parse_diff_pairs PATCH 1: no extrapolation)
    f_native = ntwk.f  # Hz
    # Guard: clip target grid into the native range to avoid extrapolation
    f_lo, f_hi = f_native.min(), f_native.max()
    target = np.clip(TARGET_FREQ_HZ, f_lo, f_hi)
    if not np.allclose(target, TARGET_FREQ_HZ):
        print(f"  [note] target grid clipped to native range "
              f"[{f_lo/1e9:.3f}, {f_hi/1e9:.3f}] GHz to avoid extrapolation")

    s_mm = np.empty((FREQ_N_POINTS, 4, 4), dtype=np.complex128)
    for i in range(4):
        for j in range(4):
            re = np.interp(target, f_native, s_mm_native[:, i, j].real)
            im = np.interp(target, f_native, s_mm_native[:, i, j].imag)
            s_mm[:, i, j] = re + 1j * im
    return TARGET_FREQ_HZ, s_mm


# ----------------------------------------------------------------------------
# Comparison A: extracted-from-raw vs processed .pt
# ----------------------------------------------------------------------------
def comparison_A(sim_id: str, pair_k: int, s_mm_extracted: np.ndarray) -> None:
    """Compare our freshly-extracted mixed-mode pair against the stored one."""
    try:
        import torch
    except ImportError:
        print("  [skip Comparison A] torch not importable")
        return

    if not PROCESSED_PT.exists():
        print(f"  [skip Comparison A] {PROCESSED_PT} not found")
        return

    payload = torch.load(PROCESSED_PT, weights_only=False, map_location="cpu")
    sim_ids = list(payload["sim_ids"])
    pair_ids = payload["pair_ids"].numpy()

    # Find the row matching this sim_id and pair_id.
    # pair_ids are 1-based pair indices in the pipeline.
    matches = [
        i for i, (s, p) in enumerate(zip(sim_ids, pair_ids))
        if s == sim_id and int(p) == pair_k
    ]
    if not matches:
        print(f"  [Comparison A] sim {sim_id} pair {pair_k} not in processed "
              f"dataset (may have been skipped). Available pairs for this sim: "
              f"{sorted(int(pair_ids[i]) for i,s in enumerate(sim_ids) if s==sim_id)}")
        return

    row = matches[0]
    y_real = payload["Y_real"][row].numpy()   # (401, 4, 4)
    y_imag = payload["Y_imag"][row].numpy()
    s_mm_stored = y_real + 1j * y_imag

    # Compare
    diff = np.abs(s_mm_extracted - s_mm_stored)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())

    # Focus on the two headline elements
    sdd11_diff = np.abs(s_mm_extracted[:, 0, 0] - s_mm_stored[:, 0, 0]).max()
    sdd21_diff = np.abs(s_mm_extracted[:, 1, 0] - s_mm_stored[:, 1, 0]).max()

    print(f"\n  COMPARISON A (extracted-from-raw vs processed .pt)")
    print(f"    matched processed row: {row}  (sim {sim_id}, pair {pair_k})")
    print(f"    max |diff| over all 4x4xF : {max_abs:.3e}")
    print(f"    mean |diff|               : {mean_abs:.3e}")
    print(f"    max |diff| Sdd11          : {float(sdd11_diff):.3e}")
    print(f"    max |diff| Sdd21          : {float(sdd21_diff):.3e}")
    if max_abs < 1e-4:
        print(f"    VERDICT: MATCH (extraction chain verified independent of solver)")
    elif max_abs < 1e-2:
        print(f"    VERDICT: CLOSE (small diff; likely interpolation/reciprocity "
              f"rounding). Inspect if this grows.")
    else:
        print(f"    VERDICT: MISMATCH -- extraction convention differs from the "
              f"pipeline. Investigate port indices / M_BE / interpolation BEFORE "
              f"any OpenEMS work.")


# ----------------------------------------------------------------------------
# Port-reference diagnosis
# ----------------------------------------------------------------------------
def diagnose_port_reference(freq_hz: np.ndarray, s_mm: np.ndarray) -> None:
    """Print the S-parameter signatures that reveal the port reference."""
    f_ghz = freq_hz / 1e9
    # indices in mixed-mode: [d_TX, d_RX, c_TX, c_RX] = [0,1,2,3]
    sdd11 = s_mm[:, 0, 0]
    sdd21 = s_mm[:, 1, 0]
    scc11 = s_mm[:, 2, 2]
    scd21 = s_mm[:, 1, 2]   # common->diff coupling (should be ~0 for symmetric)

    lo = 0                      # 0.25 GHz (first point)
    mid = int(np.argmin(np.abs(f_ghz - 14.0)))   # ~Nyquist
    hi = int(np.argmin(np.abs(f_ghz - 56.0)))

    print(f"\n  PORT-REFERENCE DIAGNOSIS (mixed-mode signatures)")
    print(f"    {'f[GHz]':>8s}  {'Sdd11':>8s}  {'Sdd21':>8s}  "
          f"{'Scc11':>8s}  {'Scd21':>8s}")
    for label, k in [("low", lo), ("~14", mid), ("~56", hi)]:
        print(f"    {f_ghz[k]:8.2f}  {db(sdd11[k]):8.2f}  {db(sdd21[k]):8.2f}  "
              f"{db(scc11[k]):8.2f}  {db(scd21[k]):8.2f}")

    # Interpretation heuristics
    sdd21_lo_db = db(sdd21[lo])
    sdd11_lo_db = db(sdd11[lo])
    scd21_lo_db = db(scd21[lo])
    print(f"\n    Interpretation:")
    if sdd21_lo_db > -1.0:
        print(f"      - Sdd21(low) = {sdd21_lo_db:.2f} dB ~ 0 dB: the through-path")
        print(f"        transmits at low frequency, consistent with a passive")
        print(f"        differential through-via referenced single-ended-to-ground.")
    else:
        print(f"      - Sdd21(low) = {sdd21_lo_db:.2f} dB is well below 0 dB. If this")
        print(f"        is a through-via, the port reference may differ from the")
        print(f"        simple via-end-to-ground assumption. Flag for stage 04.")
    if sdd11_lo_db < -10.0:
        print(f"      - Sdd11(low) = {sdd11_lo_db:.2f} dB (deep): well matched at low")
        print(f"        frequency, again consistent with 50 ohm-to-ground ports.")
    else:
        print(f"      - Sdd11(low) = {sdd11_lo_db:.2f} dB is not deep; unusual for a")
        print(f"        matched through-via. Note for stage 04 port setup.")
    if scd21_lo_db < -25.0:
        print(f"      - Scd21(low) = {scd21_lo_db:.2f} dB (small): mode conversion is")
        print(f"        low, so the pair is close to symmetric -- expected.")
    else:
        print(f"      - Scd21(low) = {scd21_lo_db:.2f} dB: notable mode conversion")
        print(f"        even at low frequency; the pair may be asymmetric.")
    print(f"\n    => Stage 04 will define lumped ports as via-end-to-nearest-"
          f"ground-plane\n       (50 ohm). Stage 05 Comparison C confirms or "
          f"refutes this reference.")


# ----------------------------------------------------------------------------
# Export ground-truth curves for stage 05
# ----------------------------------------------------------------------------
def export_ground_truth(sim_id: str, pair_k: int, freq_hz: np.ndarray,
                        s_mm: np.ndarray) -> Path:
    out = RESULTS_DIR / f"{sim_id}_pair{pair_k}_conmls_mixedmode.csv"
    header = ("freq_Hz,"
              "Sdd11_re,Sdd11_im,Sdd21_re,Sdd21_im,"
              "Sdc11_re,Sdc11_im,Scc11_re,Scc11_im")
    cols = [
        freq_hz,
        s_mm[:, 0, 0].real, s_mm[:, 0, 0].imag,
        s_mm[:, 1, 0].real, s_mm[:, 1, 0].imag,
        s_mm[:, 0, 2].real, s_mm[:, 0, 2].imag,
        s_mm[:, 2, 2].real, s_mm[:, 2, 2].imag,
    ]
    np.savetxt(out, np.column_stack(cols), delimiter=",", header=header,
               comments="")
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", type=str, default="sim_pkg_0017")
    ap.add_argument("--pair", type=int, default=1,
                    help="1-based differential pair index within the sim")
    args = ap.parse_args()

    print("=" * 78)
    print(f"Stage 03b: touchstone diagnostic  ({args.sim}, pair {args.pair})")
    print("=" * 78)

    ntwk = load_touchstone(args.sim)
    print(f"  Loaded {ntwk.nports}-port network, "
          f"{len(ntwk.f)} freq points "
          f"[{ntwk.f.min()/1e9:.3f}, {ntwk.f.max()/1e9:.3f}] GHz")

    freq_hz, s_mm = extract_pair_mixed_mode(ntwk, args.pair)

    # 1. Comparison A (extraction audit)
    comparison_A(args.sim, args.pair, s_mm)

    # 2. Port-reference diagnosis
    diagnose_port_reference(freq_hz, s_mm)

    # 3. Export ground-truth curves
    out = export_ground_truth(args.sim, args.pair, freq_hz, s_mm)
    print(f"\n  Ground-truth mixed-mode curves saved: {out}")

    print("\n" + "=" * 78)
    print("Stage 03b complete.")
    print("  - If Comparison A says MATCH, the extraction chain is verified and")
    print("    stage 05 Comparison A will pass by construction.")
    print("  - The port-reference diagnosis tells stage 04 how to set up ports.")
    print("  - Re-run with different --pair to inspect other pairs in this sim.")
    print("=" * 78)


if __name__ == "__main__":
    main()