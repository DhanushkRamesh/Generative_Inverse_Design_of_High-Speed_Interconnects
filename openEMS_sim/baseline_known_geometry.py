"""
05_baseline_known_geometry.py
================================================================================
Stage 05 of the openEMS validation pipeline.

PURPOSE
    Compare the OpenEMS full-wave result against the TUHH CONMLS ground truth.
    This establishes the "Solver-Consistency Floor" (delta_solver), quantifying
    the natural baseline discrepancy between FDTD and semi-analytical physics.
"""

import torch
import numpy as np
import skrf as rf
import matplotlib
matplotlib.use("Agg")  # Headless-safe for terminals
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent
DATA_PT = PROJECT_ROOT / "data" / "processed" / "Universal-Diff-SI-Array" / "diff_pair_dataset.pt"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=str, default="sim_pkg_0017")
    parser.add_argument("--pair", type=int, default=1, help="1-based pair index")
    args = parser.parse_args()

    print("=" * 78)
    print(f"Stage 05: Baseline Known Geometry Overlay ({args.sim}, Pair {args.pair})")
    print("=" * 78)
    
    OPENEMS_RES = _THIS_DIR / "results" / "04_openems" / f"{args.sim}_openems_se.npz"
    OUT_DIR = _THIS_DIR / "results" / "05_baseline"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load OpenEMS Results
    if not OPENEMS_RES.exists():
        print(f"ERROR: OpenEMS result not found at {OPENEMS_RES}")
        print("Make sure you ran Stage 04 with the '--run' flag first!")
        return

    print(f"Loading OpenEMS single-ended S-parameters from {OPENEMS_RES.name}...")
    data = np.load(OPENEMS_RES)
    freq_hz = data['freq']
    freq_ghz = freq_hz / 1e9
    S_se = data['S']  # Shape: (401, 16, 16)

    # 2. Extract Pair & Convert to Mixed Mode
    # For array dataset, Pair N occupies single-ended ports:
    # [4*(N-1), 4*(N-1)+2, 4*(N-1)+1, 4*(N-1)+3]
    base = 4 * (args.pair - 1)
    pair_idx = [base, base + 2, base + 1, base + 3]
    
    print(f"Extracting Pair {args.pair} using indices: {pair_idx}")
    S_se_pair = S_se[:, pair_idx, :][:, :, pair_idx]
    
    print("Converting OpenEMS results to Mixed-Mode (Sdd, Sdc, Scd, Scc)...")
    ntwk = rf.Network(frequency=freq_ghz, s=S_se_pair, z0=50, f_unit='GHz')
    ntwk.se2gmm(p=2)
    S_mm_oems = ntwk.s

    Sdd11_oems = 20 * np.log10(np.abs(S_mm_oems[:, 0, 0]) + 1e-12)
    Sdd21_oems = 20 * np.log10(np.abs(S_mm_oems[:, 1, 0]) + 1e-12)

    # 3. Load CONMLS Ground Truth from Dataset
    print(f"Loading CONMLS Ground Truth from processed dataset...")
    payload = torch.load(DATA_PT, weights_only=False, map_location='cpu')
    
    sim_ids = np.array(payload["sim_ids"])
    pair_ids = payload["pair_ids"].numpy()
    
    # Find exact match
    matches = np.where((sim_ids == args.sim) & (pair_ids == args.pair))[0]
    if len(matches) == 0:
        print(f"ERROR: No dataset entry found for {args.sim}, pair {args.pair}")
        return
    target_idx = matches[0]
    
    Y_real = payload["Y_real"][target_idx].numpy()
    Y_imag = payload["Y_imag"][target_idx].numpy()
    S_mm_conmls = Y_real + 1j * Y_imag
    
    Sdd11_conmls = 20 * np.log10(np.abs(S_mm_conmls[:, 0, 0]) + 1e-12)
    Sdd21_conmls = 20 * np.log10(np.abs(S_mm_conmls[:, 1, 0]) + 1e-12)

    # 4. Calculate Solver-Consistency Floor (delta_solver)
    # Calculate the Mean Absolute Error (MAE) up to 56 GHz (112G Harmonic Band)
    idx_56 = np.argmin(np.abs(freq_ghz - 56.0))
    delta_sdd11 = np.mean(np.abs(Sdd11_oems[:idx_56] - Sdd11_conmls[:idx_56]))
    delta_sdd21 = np.mean(np.abs(Sdd21_oems[:idx_56] - Sdd21_conmls[:idx_56]))

    print("\n--- Solver-Consistency Floor (delta_solver) [0-56 GHz] ---")
    print(f"  Sdd11 Mean Diff: {delta_sdd11:.2f} dB")
    print(f"  Sdd21 Mean Diff: {delta_sdd21:.2f} dB")
    print("  *(This is the natural disagreement between FDTD and CONMLS)*\n")

    # 5. Plotting
    print("Generating Overlay Plot...")
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Sdd11
    ax1.plot(freq_ghz, Sdd11_conmls, 'k-', lw=2.5, label='CONMLS (Ground Truth)')
    ax1.plot(freq_ghz, Sdd11_oems, 'tab:red', ls='--', lw=2.0, label='openEMS (FDTD)')
    ax1.set_title("Return Loss (|Sdd11|)")
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_ylim(-60, 5)
    ax1.legend()

    # Sdd21
    ax2.plot(freq_ghz, Sdd21_conmls, 'k-', lw=2.5, label='CONMLS (Ground Truth)')
    ax2.plot(freq_ghz, Sdd21_oems, 'tab:red', ls='--', lw=2.0, label='openEMS (FDTD)')
    ax2.set_title("Insertion Loss (|Sdd21|)")
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_ylim(-60, 5)
    ax2.legend()

    plt.suptitle(f"Stage 05 Gate: Solver-Consistency Baseline ({args.sim}, Pair {args.pair})\ndelta_solver: Sdd11={delta_sdd11:.2f}dB | Sdd21={delta_sdd21:.2f}dB", fontsize=14)
    plt.tight_layout()
    
    save_path = OUT_DIR / f"05_baseline_overlay_{args.sim}_p{args.pair}.png"
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")

    if delta_sdd21 < 3.0:
        print("\nGATE PASSED - The openEMS FDTD solver aligns beautifully with CONMLS.")
        print("Proceed to Stage 07: Validating the AI-Generated Geometries!")
    else:
        print("\nWARNING - Large solver discrepancy detected. Review the plot and mesh settings.")

if __name__ == "__main__":
    main()