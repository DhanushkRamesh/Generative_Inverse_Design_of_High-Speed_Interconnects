"""
debug_one_sim.py
================
Verbose end-to-end walkthrough of a SINGLE simulation, printing what
happens at every step.  Use this to diagnose why parse_diff_pairs.py
is dropping every pair on real data.

Run from sandbox_v1/:
    python debug_one_sim.py --dataset Array
    python debug_one_sim.py --dataset Array --sim sim_pkg_0017
    python debug_one_sim.py --dataset Link
"""

from __future__ import annotations
import sys
from pathlib import Path

SANDBOX_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SANDBOX_ROOT))

import argparse
import numpy as np
import pandas as pd
import skrf as rf

from utils.physics_utils import (
    convert_to_mixed_mode, enforce_reciprocity,
    check_passivity, reciprocity_residual,
)
from data.parse_via_array import (
    parse_via_array, identify_diff_pairs,
    diff_pair_port_indices, context_vector,
)


def hdr(s: str) -> None:
    print(f"\n{'-' * 6} {s} {'-' * (60 - len(s))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["Array", "Link"])
    ap.add_argument("--sim", default=None,
                    help="Specific sim_id to debug. If omitted, uses the first row of parameter.csv.")
    args = ap.parse_args()

    project_root = SANDBOX_ROOT.parent
    raw_dir = project_root / "data" / "raw" / f"Universal-Diff-SI-{args.dataset}"
    print(f"Raw data dir:  {raw_dir}")
    print(f"               exists: {raw_dir.exists()}")
    if not raw_dir.exists():
        print("ABORT: raw data dir not found.  Edit the script if your path differs.")
        return

    # --------------------------------------------------------------------- step 1
    hdr("STEP 1: load parameter.csv")
    df = pd.read_csv(raw_dir / "parameter.csv")
    if "LOSTANGENT" in df.columns and "LOSSTANGENT" not in df.columns:
        df = df.rename(columns={"LOSTANGENT": "LOSSTANGENT"})
    print(f"  Loaded {len(df)} rows, {df.shape[1]} columns.")

    if args.sim:
        rows = df[df["SIMULATION"].astype(str) == args.sim]
        if len(rows) == 0:
            print(f"ABORT: simulation '{args.sim}' not found in CSV.")
            return
        sim_row = rows.iloc[0]
    else:
        sim_row = df.iloc[0]
    sim_id = str(sim_row["SIMULATION"])
    print(f"  Debugging simulation: {sim_id}")
    print(f"  SIGNAL_AMOUNT = {sim_row['SIGNAL_AMOUNT']}")
    print(f"  VIAS_X x VIAS_Y = {int(sim_row['VIAS_X_AMOUNT'])} x {int(sim_row['VIAS_Y_AMOUNT'])}")

    sim_dir = raw_dir / "variation" / sim_id
    print(f"  sim_dir = {sim_dir}")
    print(f"  exists  = {sim_dir.exists()}")
    if not sim_dir.exists():
        print("ABORT: sim folder not found.")
        return
    print(f"  contents = {[p.name for p in sim_dir.iterdir()]}")

    # --------------------------------------------------------------------- step 2
    hdr("STEP 2: find and load touchstone")
    ts_paths = list(sim_dir.glob("*.s*p"))
    print(f"  touchstone files: {[p.name for p in ts_paths]}")
    if not ts_paths:
        print("ABORT: no touchstone file.")
        return
    ts_path = ts_paths[0]

    try:
        ntwk = rf.Network(str(ts_path))
        print(f"  loaded OK.  S shape = {ntwk.s.shape}, "
              f"freqs = {ntwk.f[0]/1e9:.3f} .. {ntwk.f[-1]/1e9:.1f} GHz "
              f"({ntwk.f.size} pts)")
    except Exception as e:
        print(f"ABORT: skrf failed to load network: {type(e).__name__}: {e}")
        return

    num_ports = ntwk.s.shape[1]
    expected = 2 * int(sim_row["SIGNAL_AMOUNT"])
    print(f"  num_ports = {num_ports}   expected (2*SIGNAL_AMOUNT) = {expected}   "
          f"{'OK' if num_ports == expected else 'MISMATCH'}")

    # --------------------------------------------------------------------- step 3
    hdr("STEP 3: interpolate to standard frequency grid")
    std_freq = rf.Frequency(0, 100, 401, unit="ghz")
    try:
        ntwk_interp = ntwk.interpolate(std_freq, bounds_error=False, fill_value="extrapolate")
        print(f"  interpolated OK.  new S shape = {ntwk_interp.s.shape}")
    except Exception as e:
        print(f"ABORT: interpolation failed: {type(e).__name__}: {e}")
        return

    # --------------------------------------------------------------------- step 4
    hdr("STEP 4: parse via_array.txt")
    via_path = sim_dir / "via_array.txt"
    print(f"  via_path = {via_path}  (exists: {via_path.exists()})")
    if not via_path.exists():
        print("ABORT: via_array.txt missing.")
        return

    try:
        parsed = parse_via_array(via_path)
        print(f"  ports parsed: {len(parsed['ports'])}")
        print(f"  grid shape:   {len(parsed['grid'])} rows x "
              f"{len(parsed['grid'][0]) if parsed['grid'] else 0} cols")
        print(f"  first port:   {parsed['ports'][0] if parsed['ports'] else 'NONE'}")
        print(f"  last port:    {parsed['ports'][-1] if parsed['ports'] else 'NONE'}")
    except Exception as e:
        print(f"ABORT: via_array parse failed: {type(e).__name__}: {e}")
        return

    # --------------------------------------------------------------------- step 5
    hdr("STEP 5: identify diff pairs")
    diff_pairs = identify_diff_pairs(parsed)
    expected_pairs = int(sim_row["SIGNAL_AMOUNT"]) // 2
    print(f"  found {len(diff_pairs)} pairs;  expected {expected_pairs}   "
          f"{'OK' if len(diff_pairs) == expected_pairs else 'MISMATCH'}")
    for p in diff_pairs[:3]:
        print(f"    pair {p['pair_id']}: sgn{p['sgn_a']}@({p['x_a']},{p['y_a']})  "
              f"+  sgn{p['sgn_b']}@({p['x_b']},{p['y_b']})")
    if len(diff_pairs) > 3:
        print(f"    ... ({len(diff_pairs) - 3} more)")

    # --------------------------------------------------------------------- step 6
    hdr("STEP 6: process each diff pair")
    n_passed, n_failed_passive, n_other_err = 0, 0, 0
    min_eig_list = []
    recip_res_list = []
    s_se_full_min_eig = None

    # Also compute passivity on the FULL N-port network for comparison
    is_full_passive, full_min_eig = check_passivity(
        ntwk_interp.s.astype(np.complex128), threshold=-1e-6
    )
    print(f"  FULL N-port passivity check  "
          f"(N={num_ports}):  min_eig = {full_min_eig:.4e}  "
          f"=> {'PASS' if is_full_passive else 'FAIL'}")

    for pair in diff_pairs:
        k = pair["pair_id"]
        try:
            port_idx = diff_pair_port_indices(args.dataset, num_ports, k)
        except ValueError as e:
            print(f"  pair {k}: port-index error: {e}")
            n_other_err += 1
            continue

        s_se = ntwk_interp.s[:, port_idx][:, :, port_idx].astype(np.complex128)
        recip_res = reciprocity_residual(s_se)
        s_se = enforce_reciprocity(s_se)

        is_passive, min_eig = check_passivity(s_se, threshold=-1e-6)
        min_eig_list.append(min_eig)
        recip_res_list.append(recip_res)
        if not is_passive:
            n_failed_passive += 1
        else:
            try:
                _ = convert_to_mixed_mode(s_se)
                _ = context_vector(pair, parsed["grid"],
                                   int(sim_row["VIAS_X_AMOUNT"]),
                                   int(sim_row["VIAS_Y_AMOUNT"]))
                n_passed += 1
            except Exception as e:
                print(f"  pair {k}: error in MM/context: {type(e).__name__}: {e}")
                n_other_err += 1

    print(f"\n  Summary across {len(diff_pairs)} pairs:")
    print(f"    accepted:                 {n_passed}")
    print(f"    failed passivity (-1e-6): {n_failed_passive}")
    print(f"    other errors:             {n_other_err}")
    if min_eig_list:
        m = np.array(min_eig_list)
        print(f"    sliced 4x4 min_eig:   min={m.min():.2e}  max={m.max():.2e}  "
              f"median={np.median(m):.2e}")
        # What threshold would catch X% of the pairs?
        for thresh in (-1e-6, -1e-5, -1e-4, -1e-3, -1e-2, -1e-1):
            kept = int((m >= thresh).sum())
            print(f"      with threshold {thresh:+.0e}:  {kept}/{len(m)} pairs would pass")
    if recip_res_list:
        r = np.array(recip_res_list)
        print(f"    reciprocity residual: min={r.min():.2e}  max={r.max():.2e}  "
              f"median={np.median(r):.2e}")

    print("\nDone.")


if __name__ == "__main__":
    main()