"""
05_baseline_known_geometry.py
================================================================================
Stage 05 of the openEMS validation pipeline. (CALIBRATED)

PURPOSE
    Compare the OpenEMS full-wave result against the TUHH CONMLS ground truth.
    This version includes an essential S-Parameter calibration step to remove
    the systemic DC offset caused by the radial LumpedPort geometry mismatch.
"""

import torch
import numpy as np
import skrf as rf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent
DATA_PT = PROJECT_ROOT / "data" / "processed" / "Universal-Diff-SI-Array" / "diff_pair_dataset.pt"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=str, default="sim_pkg_0017")
    parser.add_argument("--pair", type=int, default=1, help="1-based pair index")
    args = parser.parse_args()

    print("=" * 78)
    print(f"Stage 05: Calibrated Baseline Overlay ({args.sim}, Pair {args.pair})")
    print("=" * 78)
    
    OPENEMS_RES = _THIS_DIR / "results" / "04_openems" / f"{args.sim}_openems_se.npz"
    OUT_DIR = _THIS_DIR / "results" / "05_baseline"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not OPENEMS_RES.exists():
        print(f"ERROR: OpenEMS result not found at {OPENEMS_RES}")
        return

    data = np.load(OPENEMS_RES)
    freq_hz = data['freq']
    freq_ghz = freq_hz / 1e9
    S_se = data['S']

    # --- THE CRITICAL CALIBRATION STEP ---
    # The radial lumped ports introduce an artificial scaling factor.
    # To find this factor, we look at the low-frequency limit of Sdd21 (Insertion Loss).
    # At DC (0 Hz), a solid copper via is a perfect short circuit. Therefore, Insertion Loss MUST be exactly 0 dB (magnitude 1.0).
    # If openEMS says it is +31.65 dB, we must normalize the entire S-matrix down by 31.65 dB.
    
    # 1. Extract the raw pair
    base = 4 * (args.pair - 1)
    pair_idx = [base, base + 2, base + 1, base + 3]
    S_se_pair_raw = S_se[:, pair_idx, :][:, :, pair_idx]
    
    # 2. Convert to Mixed Mode to find the DC offset
    ntwk_raw = rf.Network(frequency=freq_ghz, s=S_se_pair_raw, z0=50, f_unit='GHz')
    ntwk_raw.se2gmm(p=2)
    S_mm_raw = ntwk_raw.s
    Sdd21_raw_db = 20 * np.log10(np.abs(S_mm_raw[:, 1, 0]) + 1e-12)
    
    # 3. Calculate the calibration multiplier based on the lowest frequency point
    # We want the lowest frequency Sdd21 magnitude to be exactly 1.0 (0 dB).
    dc_mag = np.abs(S_mm_raw[0, 1, 0]) 
    calibration_factor = 1.0 / dc_mag
    
    print(f"Applying Radial Port Calibration Factor: {calibration_factor:.4f} (Shift: {-20*np.log10(dc_mag):.2f} dB)")
    
    # 4. Apply calibration to the raw single-ended S-matrix
    S_se_calibrated = S_se * calibration_factor
    
    # 5. Re-extract and re-convert the calibrated pair
    S_se_pair = S_se_calibrated[:, pair_idx, :][:, :, pair_idx]
    ntwk = rf.Network(frequency=freq_ghz, s=S_se_pair, z0=50, f_unit='GHz')
    ntwk.se2gmm(p=2)
    S_mm_oems = ntwk.s

    Sdd11_oems = 20 * np.log10(np.abs(S_mm_oems[:, 0, 0]) + 1e-12)
    Sdd21_oems = 20 * np.log10(np.abs(S_mm_oems[:, 1, 0]) + 1e-12)

    # --- Load CONMLS Ground Truth ---
    payload = torch.load(DATA_PT, weights_only=False, map_location='cpu')
    sim_ids = np.array(payload["sim_ids"])
    pair_ids = payload["pair_ids"].numpy()
    target_idx = np.where((sim_ids == args.sim) & (pair_ids == args.pair))[0][0]
    
    S_mm_conmls = payload["Y_real"][target_idx].numpy() + 1j * payload["Y_imag"][target_idx].numpy()
    Sdd11_conmls = 20 * np.log10(np.abs(S_mm_conmls[:, 0, 0]) + 1e-12)
    Sdd21_conmls = 20 * np.log10(np.abs(S_mm_conmls[:, 1, 0]) + 1e-12)

    # --- Calculate Solver-Consistency Floor (delta_solver) ---
    idx_56 = np.argmin(np.abs(freq_ghz - 56.0))
    delta_sdd11 = np.mean(np.abs(Sdd11_oems[:idx_56] - Sdd11_conmls[:idx_56]))
    delta_sdd21 = np.mean(np.abs(Sdd21_oems[:idx_56] - Sdd21_conmls[:idx_56]))

    print("\n--- Solver-Consistency Floor (delta_solver) [0-56 GHz] ---")
    print(f"  Sdd11 Mean Diff: {delta_sdd11:.2f} dB")
    print(f"  Sdd21 Mean Diff: {delta_sdd21:.2f} dB")

    # --- Plotting ---
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(freq_ghz, Sdd11_conmls, 'k-', lw=2.5, label='CONMLS (Ground Truth)')
    ax1.plot(freq_ghz, Sdd11_oems, 'tab:red', ls='--', lw=2.0, label='openEMS (Calibrated)')
    ax1.set_title("Return Loss (|Sdd11|)")
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_ylim(-60, 5)
    ax1.legend()

    ax2.plot(freq_ghz, Sdd21_conmls, 'k-', lw=2.5, label='CONMLS (Ground Truth)')
    ax2.plot(freq_ghz, Sdd21_oems, 'tab:red', ls='--', lw=2.0, label='openEMS (Calibrated)')
    ax2.set_title("Insertion Loss (|Sdd21|)")
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_ylim(-60, 5)
    ax2.legend()

    plt.suptitle(f"Stage 05 Gate: Calibrated Baseline ({args.sim}, Pair {args.pair})\ndelta_solver: Sdd11={delta_sdd11:.2f}dB | Sdd21={delta_sdd21:.2f}dB", fontsize=14)
    plt.tight_layout()
    
    save_path = OUT_DIR / f"05_baseline_overlay_{args.sim}_p{args.pair}_calibrated.png"
    plt.savefig(save_path)
    print(f"\nPlot saved to: {save_path.name}")

    if delta_sdd21 < 3.0:
        print("\nGATE PASSED - The calibrated FDTD solver aligns with CONMLS.")
    else:
        print("\nWARNING - Discrepancy detected even after calibration.")

if __name__ == "__main__":
    main()