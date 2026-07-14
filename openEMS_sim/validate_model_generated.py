"""
07_validate_generated.py  (v2 - aligned to actual TTO npz + from_feature_vector)
================================================================================
Stage 07 of the openEMS validation pipeline -- THE THESIS PAYOFF.

Validates geometries INVENTED by the inverse model (cVAE + TTO). For each
generated design it:
  1. de-normalizes x_local_norm via stage 03's from_feature_vector (mirrors
     exactly how the dataset was built)
  2. SANITY-CHECKS the physical dimensions BEFORE any simulation (so a malformed
     geometry never wastes 20-90 min of FDTD)
  3. builds + simulates the design in OpenEMS (same v6.x all-MUR engine)
  4. produces the two thesis comparisons:
       - OpenEMS vs TARGET  -> did the AI design achieve its goal in real physics?
       - OpenEMS vs FORWARD-MODEL PREDICTION -> was the surrogate honest, or did
         TTO exploit it? (gap >> the stage-06 solver floor = exploitation)

INPUT (--designs design_sample_XX.npz), from tto_interface_inverse.py:
  x_local_norm (1,8), x_global_norm (1,6), x_context_norm (1,7),
  target_real/imag (1,401,4,4), pred_real/imag (1,401,4,4),
  template_sim_id (1,), pair_id (1,)

USAGE
  # sanity-check the geometry only, NO simulation (fast, do this first):
  python 07_validate_generated.py --designs <path>/design_sample_100.npz --check-only

  # full validation (builds + simulates in OpenEMS):
  python 07_validate_generated.py --designs <path>/design_sample_100.npz

  # reuse an existing OpenEMS result (skip the FDTD):
  python 07_validate_generated.py --designs <path>/design_sample_100.npz --skip-run
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent


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
     "build_array_model_coax_final.py", "04_build_array_model_v6_2_FINAL.py"],
    "stage04_mod7")
stage03 = stage04.stage03
ArrayModelBuilder = stage04.ArrayModelBuilder
from_feature_vector = stage03.from_feature_vector

PROJECT_ROOT = (Path.home() / "mece_project_inverse_model"
                / "Generative_Inverse_Design_of_High-Speed_Interconnects")
PROCESSED_PT = (PROJECT_ROOT / "data" / "processed"
                / "Universal-Diff-SI-Array" / "diff_pair_dataset.pt")
OUT_DIR = _THIS_DIR / "results" / "07_generated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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


def load_norm_stats():
    import torch
    p = torch.load(PROCESSED_PT, weights_only=False, map_location="cpu")
    def arr(k):
        v = p[k]
        return v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
    return {
        "X_local_mean": arr("X_local_mean"), "X_local_std": arr("X_local_std"),
        "X_global_mean": arr("X_global_mean"), "X_global_std": arr("X_global_std"),
        "X_context_mean": arr("X_context_mean"), "X_context_std": arr("X_context_std"),
        "local_features": list(p["local_features"]),
        "global_features": list(p["global_features"]),
        "context_features": list(p["context_features"]),
        "log_features": list(p["log_features"]),
    }


def sanity_check_geometry(geo, label):
    """Print physical dimensions and flag anything unphysical BEFORE simulating."""
    print(f"\n  --- geometry sanity check [{label}] ---")
    vr = geo.via_radius_mil
    ar = geo.antipad_radius_mil
    pitch = geo.pitch_mil
    print(f"    via_radius   = {vr:.3f} mil")
    print(f"    antipad_rad  = {ar:.3f} mil")
    print(f"    pitch        = {pitch:.3f} mil")
    print(f"    TDIEL/TMET   = {geo.total_thickness_mil:.1f} total, "
          f"eps_r={geo.permittivity:.3f}, tan_d={geo.loss_tangent:.4f}")
    print(f"    n_ports      = {geo.n_ports}")
    problems = []
    if vr <= 0: problems.append("via_radius <= 0")
    if ar <= vr: problems.append(f"antipad ({ar:.2f}) <= via_radius ({vr:.2f}) "
                                 "-- antipad must clear the via")
    if pitch <= 2 * ar: problems.append(f"pitch ({pitch:.2f}) <= 2*antipad "
                                        f"({2*ar:.2f}) -- vias/antipads overlap")
    if geo.permittivity <= 1: problems.append("eps_r <= 1 (unphysical)")
    if geo.loss_tangent < 0: problems.append("tan_d < 0")
    if problems:
        print("    [WARNING] unphysical geometry:")
        for p in problems:
            print(f"      - {p}")
        print("    Simulating this may fail or give garbage. Review the design.")
        return False
    print("    geometry looks physical (via < antipad < pitch/2, eps_r>1).")
    return True


def three_metrics_rows(case, cmp_name, mm_a, mm_b):
    f_ghz = FREQ / 1e9
    rows = []
    for nm, i, j in [("Sdd11", 0, 0), ("Sdd21", 1, 0)]:
        da, dbv = db(mm_a[:, i, j]), db(mm_b[:, i, j])
        d_raw = np.abs(da - dbv)
        d_flr = np.abs(np.maximum(da, NULL_FLOOR_DB) - np.maximum(dbv, NULL_FLOOR_DB))
        d_lin = np.abs(np.abs(mm_a[:, i, j]) - np.abs(mm_b[:, i, j]))
        for bn, lo, hi in BANDS:
            m = (f_ghz >= lo) & (f_ghz < hi)
            rows.append({"case": case, "comparison": cmp_name, "trace": nm,
                         "band": bn,
                         "mean_dB": round(float(d_raw[m].mean()), 3),
                         "mean_dB_floored": round(float(d_flr[m].mean()), 3),
                         "mean_linear": round(float(d_lin[m].mean()), 4)})
    return rows


def main():
    class _Tee:
        def __init__(self, *s): self.s = s
        def write(self, d):
            for x in self.s: x.write(d); x.flush()
        def flush(self):
            for x in self.s: x.flush()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log = open(_THIS_DIR / f"stage07_log_{stamp}.log", "w")
    sys.stdout = _Tee(sys.__stdout__, _log)
    sys.stderr = _Tee(sys.__stderr__, _log)

    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", required=True, help="design_sample_XX.npz")
    ap.add_argument("--check-only", action="store_true",
                    help="de-normalize + sanity-check geometry, NO simulation")
    ap.add_argument("--skip-run", action="store_true",
                    help="reuse existing OpenEMS result npz")
    ap.add_argument("--use-cvae", action="store_true",
                    help="validate the cVAE guess instead of the TTO geometry")
    ap.add_argument("--cells", type=int,
                    default=getattr(stage04, "CELLS_PER_WAVELENGTH", 20))
    args = ap.parse_args()

    d = np.load(args.designs, allow_pickle=True)
    case = Path(args.designs).stem
    template = str(d["template_sim_id"][0])
    pair_k = int(d["pair_id"][0])
    which = "x_local_cvae" if args.use_cvae else "x_local_norm"
    x_local = d[which][0]
    x_global = d["x_global_norm"][0]
    x_context = d["x_context_norm"][0]
    target = (d["target_real"] + 1j * d["target_imag"])[0]
    pred = (d["pred_real"] + 1j * d["pred_imag"])[0]

    print("=" * 78)
    print(f"Stage 07: {case}  (template {template}, pair {pair_k}, "
          f"geometry='{which}')")
    print("=" * 78)

    norm = load_norm_stats()
    geo = from_feature_vector(
        x_local_norm=x_local, x_global_norm=x_global,
        x_context_norm=x_context, norm_stats=norm, template_sim_id=template)

    ok = sanity_check_geometry(geo, case)
    if args.check_only:
        print("\n[--check-only] stopping before simulation. If the geometry "
              "looks good, re-run without --check-only.")
        return
    if not ok:
        print("\n[abort] geometry failed the sanity check; not simulating. "
              "Use --check-only to inspect, or fix the design.")
        return

    npz = OUT_DIR / f"{case}_openems_se.npz"
    if args.skip_run and npz.exists():
        print(f"\n  reusing {npz.name}")
        S_oe = np.load(npz)["S"]
    else:
        print(f"\n  building + simulating in OpenEMS ({geo.n_ports} ports) ...")
        builder = ArrayModelBuilder(geo, cells_per_wavelength=args.cells,
                                    verbose=True)
        sim_root = _THIS_DIR / "runs" / f"07_{case}"
        freq, S_oe = builder.run_and_extract(sim_root, run=True)
        np.savez(npz, freq=freq, S=S_oe)
        print(f"  saved {npz.name}")

    mm_oe = extract_pair(S_oe, pair_k)

    rows = []
    rows += three_metrics_rows(case, "openems_vs_target", mm_oe, target)
    rows += three_metrics_rows(case, "openems_vs_pred", mm_oe, pred)

    # overlay figure
    f_ghz = FREQ / 1e9
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), tight_layout=True)
    for ax, (i, j, ttl) in zip(axes, [(0, 0, "Sdd11"), (1, 0, "Sdd21")]):
        ax.plot(f_ghz, db(target[:, i, j]), "k-", lw=2, label="Target")
        ax.plot(f_ghz, db(pred[:, i, j]), "b:", lw=1.5, label="Forward-model pred")
        ax.plot(f_ghz, db(mm_oe[:, i, j]), "r--", lw=2, label="OpenEMS (design)")
        ax.set_xlabel("Frequency (GHz)"); ax.set_ylabel("dB")
        ax.set_title(f"{ttl} -- {case}"); ax.grid(alpha=0.3); ax.legend()
        ax.set_xlim(0, 100)
    fig_path = OUT_DIR / f"{case}_overlay_{stamp}.png"
    fig.savefig(fig_path, dpi=150); plt.close(fig)

    table = OUT_DIR / f"{case}_validation_{stamp}.csv"
    with open(table, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 78)
    print("STAGE 07 RESULT  (mean floored dB, eye band)")
    print("=" * 78)
    for cmp_name in ("openems_vs_target", "openems_vs_pred"):
        print(f"  [{cmp_name}]")
        for r in rows:
            if r["comparison"] == cmp_name and r["band"] in EYE_BANDS:
                print(f"    {r['trace']:6s} {r['band']:7s} "
                      f"floored={r['mean_dB_floored']:6.2f} dB  "
                      f"lin={r['mean_linear']:.4f}")
    print(f"\n  figure: {fig_path.name}")
    print(f"  table : {table.name}")
    print("\nINTERPRETATION:")
    print("  openems_vs_target small  -> the design achieves its goal in real")
    print("     physics: the inverse model works. THE thesis result.")
    print("  openems_vs_pred within the stage-06 solver floor -> the forward")
    print("     surrogate was honest about this design.")
    print("  openems_vs_pred >> floor (esp. where TTO optimised) -> surrogate")
    print("     exploitation, caught in independent physics. Motivates latent TTO.")


if __name__ == "__main__":
    main()