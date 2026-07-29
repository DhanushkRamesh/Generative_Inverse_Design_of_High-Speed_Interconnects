"""
06_delta_solver_batch.py  (v3 - matches stage 04 v6.2 + stage 05 v3 metrics)
================================================================================
Stage 06 of the openEMS validation pipeline.

PURPOSE
    Stage 05 proves OpenEMS reproduces ONE geometry. Stage 06 proves the
    agreement is CONSISTENT across SEVERAL geometries (different layer counts,
    via radii, materials). The output "across N geometries, OpenEMS and CONMLS
    agree to within X dB per band" is the solver-consistency floor -- a thesis
    table, and the baseline stage 07 subtracts when judging model designs.

    Run only after the stage-05 gate passes on sim_pkg_0017.

METRICS (identical to stage 05 v3, so the numbers are directly comparable):
    - raw dB, floored dB (-30 dB clip), linear |dS|  per band
    - MODEL-WEIGHTED delta (element-aware frequency-importance tensor)

EFFICIENCY
    --skip-run reuses existing results/04_openems/*.npz (instant comparison).
    You already simulated sim_pkg_0017 and sim_pkg_2871, so:
        python 06_delta_solver_batch.py --sims sim_pkg_0017 sim_pkg_2871 --skip-run
    only runs the COMPARISON; it does not re-simulate.

USAGE
    # compare already-simulated sims (fast, no FDTD):
    python 06_delta_solver_batch.py --sims sim_pkg_0017 sim_pkg_2871 --skip-run

    # simulate a NEW sim then compare (slow, runs stage 04):
    python 06_delta_solver_batch.py --sims sim_pkg_0101

PICKING SIMS
    Span the parameter range: different LAYER_AMOUNT, VIA_RADIUS, PITCH.
    Any sim that parsed into the processed dataset works.

NPZ NAMING
    The .npz written by stage 04 is looked up by trying several known suffixes
    (your files use '_mur_lateral_openems_se.npz'). New sims run here are saved
    with the same suffix so stage 05/07/08 find them.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------
# Load stage 04 (any of the user's renamed files)
# ----------------------------------------------------------------------------
def _load_module(candidates, alias):
    for name in candidates:
        p = _THIS_DIR / name
        if p.exists():
            spec = importlib.util.spec_from_file_location(alias, p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[alias] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(f"None of {candidates} found in {_THIS_DIR}")


stage04 = _load_module(
    ["build_array_model_mur_final.py", "04_build_array_model.py",
     "build_array_model_coax_final.py", "build_array_model_v6_2_FINAL.py",
     "04_build_array_model_v6_2_FINAL.py"], "stage04_mod")
ArrayModelBuilder = stage04.ArrayModelBuilder
from_sim_folder = stage04.from_sim_folder

PROJECT_ROOT = (Path.home() / "mece_project_inverse_model"
                / "Generative_Inverse_Design_of_High-Speed_Interconnects")
PROCESSED_PT = (PROJECT_ROOT / "data" / "processed"
                / "Universal-Diff-SI-Array" / "diff_pair_dataset.pt")
DEFAULT_WEIGHTS = (_THIS_DIR.parent / "sandbox_v1" / "data" / "frequency_eda"
                   / "weights_element_aware_per_freq.npy")

NPZ_DIR = _THIS_DIR / "results" / "04_openems"
OUT_DIR = _THIS_DIR / "results" / "06_delta_solver"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# npz filename suffixes to try when loading an existing result (yours first)
NPZ_SUFFIXES = ["_mur_lateral_openems_se.npz", "_openems_se.npz",
                "_final_coax_openems_se.npz", "_best_openems_se.npz"]
# suffix used when THIS script simulates a new sim
NPZ_SAVE_SUFFIX = "_mur_lateral_openems_se.npz"

FREQ = np.linspace(0.25e9, 100e9, 401)
BANDS = [("0-14", 0, 14), ("14-28", 14, 28), ("28-56", 28, 56), ("56-100", 56, 100)]
EYE_BANDS = {"0-14", "14-28"}
NULL_FLOOR_DB = -30.0
M_BE = (1 / np.sqrt(2)) * np.array(
    [[1, -1, 0, 0], [0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 1, 1]], float)


def db(x):
    return 20 * np.log10(np.abs(x) + 1e-12)


def extract_pair(S_se, k):
    b = 4 * (k - 1)
    idx = [b, b + 2, b + 1, b + 3]
    s = S_se[:, idx][:, :, idx]
    s = 0.5 * (s + np.transpose(s, (0, 2, 1)))
    return M_BE @ s @ M_BE.T


def find_npz(sim_id):
    """Locate an existing OpenEMS result for sim_id, trying known suffixes."""
    for suf in NPZ_SUFFIXES:
        p = NPZ_DIR / f"{sim_id}{suf}"
        if p.exists():
            return p
    return None


def load_processed(sim_id):
    import torch
    payload = torch.load(PROCESSED_PT, weights_only=False, map_location="cpu")
    sim_ids = list(payload["sim_ids"])
    pair_ids = payload["pair_ids"].numpy()
    out = {}
    for i, (s, p) in enumerate(zip(sim_ids, pair_ids)):
        if s == sim_id:
            out[int(p)] = (payload["Y_real"][i].numpy()
                           + 1j * payload["Y_imag"][i].numpy())
    return out


def load_weights(path):
    p = Path(path)
    if not p.exists():
        print(f"  [weighted view skipped] weight file not found: {p}")
        return None
    w = np.load(p)
    if w.shape == (4, 4, 401):
        w = np.transpose(w, (2, 0, 1))
    if w.shape != (401, 4, 4):
        print(f"  [weighted view skipped] unexpected weight shape {w.shape}")
        return None
    return w


def run_sim(sim_id, cells):
    geo = from_sim_folder(sim_id)
    builder = ArrayModelBuilder(geo, cells_per_wavelength=cells, verbose=True)
    sim_root = _THIS_DIR / "runs" / f"04_{sim_id}"
    freq, S = builder.run_and_extract(sim_root, run=True)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    out = NPZ_DIR / f"{sim_id}{NPZ_SAVE_SUFFIX}"
    np.savez(out, freq=freq, S=S)
    print(f"    saved {out.name}")
    return S


def three_metrics(mm_oe, mm_ref, i, j):
    da, dbb = db(mm_oe[:, i, j]), db(mm_ref[:, i, j])
    d_raw = np.abs(da - dbb)
    d_floor = np.abs(np.maximum(da, NULL_FLOOR_DB)
                     - np.maximum(dbb, NULL_FLOOR_DB))
    d_lin = np.abs(np.abs(mm_oe[:, i, j]) - np.abs(mm_ref[:, i, j]))
    return d_raw, d_floor, d_lin


def main():
    # ---- logging (timestamped, never overwrites) ----------------------------
    class _Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, data):
            for s in self.streams: s.write(data); s.flush()
        def flush(self):
            for s in self.streams: s.flush()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _logfile = open(_THIS_DIR / f"stage06_log_{stamp}.log", "w")
    sys.stdout = _Tee(sys.__stdout__, _logfile)
    sys.stderr = _Tee(sys.__stderr__, _logfile)

    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", nargs="+", required=True)
    ap.add_argument("--skip-run", action="store_true",
                    help="reuse existing results/04_openems/*.npz (no FDTD)")
    ap.add_argument("--cells", type=int,
                    default=getattr(stage04, "CELLS_PER_WAVELENGTH", 20))
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--tag", default="",
                    help="optional label appended to output filenames")
    args = ap.parse_args()

    weights = load_weights(args.weights)
    rows, wrows = [], []

    for sim_id in args.sims:
        print("=" * 78)
        print(f"Stage 06: {sim_id}")
        print("=" * 78)

        if args.skip_run:
            npz = find_npz(sim_id)
            if npz is None:
                print(f"  [skip] no OpenEMS npz found for {sim_id} "
                      f"(tried suffixes {NPZ_SUFFIXES})")
                continue
            print(f"  reusing {npz.name}")
            S_oe = np.load(npz)["S"]
        else:
            S_oe = run_sim(sim_id, args.cells)

        stored = load_processed(sim_id)
        if not stored:
            print(f"  [warn] no processed pairs for {sim_id}; skipping")
            continue

        f_ghz = FREQ / 1e9
        for pair_k, mm_ref in sorted(stored.items()):
            try:
                mm_oe = extract_pair(S_oe, pair_k)
            except Exception as e:
                print(f"  pair {pair_k}: extraction failed ({e})")
                continue
            for name, i, j in [("Sdd11", 0, 0), ("Sdd21", 1, 0)]:
                d_raw, d_floor, d_lin = three_metrics(mm_oe, mm_ref, i, j)
                for bn, lo, hi in BANDS:
                    m = (f_ghz >= lo) & (f_ghz < hi)
                    rows.append({
                        "sim": sim_id, "pair": pair_k, "trace": name,
                        "band": bn,
                        "mean_dB": round(float(d_raw[m].mean()), 3),
                        "mean_dB_floored": round(float(d_floor[m].mean()), 3),
                        "mean_linear": round(float(d_lin[m].mean()), 4),
                        "max_dB": round(float(d_raw[m].max()), 3),
                    })
                if weights is not None:
                    wij = weights[:, i, j]
                    wd = (float((d_floor * wij).sum() / wij.sum())
                          if wij.sum() > 0 else float("nan"))
                    wrows.append({"sim": sim_id, "pair": pair_k, "trace": name,
                                  "weighted_floored_dB": round(wd, 3)})
            print(f"  pair {pair_k}: compared")

    if not rows:
        print("\nNo comparisons produced.")
        return

    suffix = (f"_{args.tag}" if args.tag else "") + f"_{stamp}"

    # ---- per-comparison table ----------------------------------------------
    table = OUT_DIR / f"delta_solver_table{suffix}.csv"
    with open(table, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- aggregate: eye-critical (floored) per trace/band ------------------
    agg = {}
    for r in rows:
        agg.setdefault((r["trace"], r["band"]), []).append(r["mean_dB_floored"])
    sum_rows = []
    for (t, b), v in sorted(agg.items()):
        sum_rows.append({"trace": t, "band": b,
                         "mean_of_means_floored_dB": round(float(np.mean(v)), 3),
                         "worst_floored_dB": round(float(np.max(v)), 3),
                         "n": len(v)})
    sfile = OUT_DIR / f"delta_solver_summary{suffix}.csv"
    with open(sfile, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sum_rows[0].keys()))
        w.writeheader(); w.writerows(sum_rows)

    print("\n" + "=" * 78)
    print("delta_solver SUMMARY -- EYE-CRITICAL (floored dB, all sims & pairs)")
    print("=" * 78)
    print(f"  {'trace':7s} {'band':8s} {'mean-of-means':>14s} {'worst':>8s} {'N':>4s}")
    for r in sum_rows:
        eye = " <- eye" if (r["trace"] in ("Sdd11", "Sdd21")
                            and r["band"] in EYE_BANDS) else ""
        print(f"  {r['trace']:7s} {r['band']:8s} "
              f"{r['mean_of_means_floored_dB']:14.2f} "
              f"{r['worst_floored_dB']:8.2f} {r['n']:4d}{eye}")

    # ---- aggregate: model-weighted -----------------------------------------
    if wrows:
        wfile = OUT_DIR / f"delta_solver_weighted{suffix}.csv"
        with open(wfile, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(wrows[0].keys()))
            w.writeheader(); w.writerows(wrows)
        wagg = {}
        for r in wrows:
            wagg.setdefault(r["trace"], []).append(r["weighted_floored_dB"])
        print("\n" + "=" * 78)
        print("delta_solver SUMMARY -- MODEL-WEIGHTED (element-aware tensor)")
        print("=" * 78)
        for t, v in sorted(wagg.items()):
            print(f"  {t:7s} weighted mean-of-means (floored) = "
                  f"{np.nanmean(v):.2f} dB  (N={len(v)})")

    print(f"\n  tables written to {OUT_DIR}/ with suffix '{suffix}'")
    print("\n  This is the solver-consistency floor across all tested sims.")
    print("  Report the eye-critical 0-28 GHz rows (protect the eye) and the")
    print("  model-weighted numbers (agreement where the model is sensitive).")


if __name__ == "__main__":
    main()