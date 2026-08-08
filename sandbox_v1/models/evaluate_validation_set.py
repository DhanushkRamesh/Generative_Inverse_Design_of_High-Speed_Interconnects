"""
evaluate_validation_set.py
--------------------------------------------------------------------------
BATCH MODEL-ACCURACY EVALUATION over a random subset of validation samples.

WHY THIS EXISTS
  Judging the inverse model from a handful of hand-picked (often deliberately
  hard) targets is a biased test. This script draws a RANDOM sample of
  validation targets and evaluates every one through BOTH stages, so the
  reported success rates are an unbiased estimate of model performance --
  exactly like reporting test-set accuracy in image classification, where a
  few dozen random test images give a percentage with a tight confidence
  interval (you do NOT need the entire test set).

WHAT IT MEASURES  (two DISTINCT stages, reported separately)
  STAGE 1 - RECONSTRUCTION ACCURACY (the pure inverse-model quality):
      fit-only latent TTO  ->  eye-band fit (dB).  Independent of yield.
      Success threshold: fit <= FIT_OK dB.
  STAGE 2 - MANUFACTURABILITY (usefulness of the produced design):
      worst-case yield TTO ->  yield (%), max|x|, physical geometry?
      Success threshold: yield >= YIELD_OK %.

  A sample is FULLY SUCCESSFUL iff it passes BOTH thresholds.

OUTPUT
  - per-sample rows printed live (so you can watch progress)
  - a CSV with every sample's fit / yield / max|x| / physical / verdicts
  - a SUMMARY block with the headline percentages + median stats

USAGE (from sandbox_v1/models/, same dir as the yield script)
  # rename/point IMPORT_NAME below to your working yield-script filename first
  python3 evaluate_validation_set.py --n 50 --seed 123

  # fit-only quick pass (stage 1 only -- fast, pure model accuracy):
  python3 evaluate_validation_set.py --n 50 --stage1-only

NOTE ON RUNTIME
  Each sample runs 12 restarts x 150 steps per stage. Two stages => ~2x.
  50 samples is a sensible default: statistically solid, finishes in a
  reasonable time. Running the *entire* 1828-sample validation set gives the
  SAME percentage (within a couple of points) at ~35x the cost -- not worth it.
"""

import argparse
import csv
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# CONFIG -- EDIT THESE TWO LINES to match your environment
# ---------------------------------------------------------------------------
# The module name (no .py) of your WORKING yield script. If your file is
# tto_yield_aware_inverse.py, set this to "tto_yield_aware_inverse".
IMPORT_NAME = "tto_yield_aware_inverse"
# Path to the processed dataset (same one the yield script loads).
DATA_PT = (Path(__file__).resolve().parent.parent.parent / "data" / "processed"
           / "Universal-Diff-SI-Array" / "diff_pair_dataset.pt")

# Success thresholds (define BEFORE running -- do not move goalposts after)
FIT_OK = 1.5      # dB: reconstruction counts as "good" if eye-fit <= this
YIELD_OK = 50.0   # %:  design counts as "manufacturable" if yield >= this

# Split reproduction (MUST match train_direct_sequence_resnet_best_v1.py)
SPLIT_SEED = 42
TRAIN_FRAC = 0.85


def get_validation_indices():
    """Reproduce the simulation-level 85/15 split and return VAL indices."""
    payload = torch.load(DATA_PT, map_location="cpu", weights_only=False)
    sim_ids = payload["sim_ids"]
    sim_ids = sim_ids.cpu().numpy() if hasattr(sim_ids, "cpu") else np.asarray(sim_ids)
    unique = np.unique(sim_ids)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(unique)
    n_train = int(TRAIN_FRAC * len(unique))
    train_sims = set(unique[:n_train])
    val_idx = np.array([i for i, s in enumerate(sim_ids) if s not in train_sims])
    return val_idx, sim_ids


def check_geometry_valid(npz_path, payload):
    """De-normalize a saved design's geometry and apply the SAME buildability
    rule as validate_model_generated.py --check-only:

        via_radius < antipad_radius < pitch/2   AND   eps_r > 1
        (equivalently 2*antipad < pitch, no via/antipad overlap)

    Returns (is_valid: bool, detail: str). A design can have great fit and
    yield yet still be geometrically UNBUILDABLE (e.g. antipad inflated past
    pitch/2) -- this closes that false-positive gap in the success criterion.
    """
    try:
        d = np.load(npz_path, allow_pickle=True)
        x_norm = d["x_local_norm"].reshape(-1)  # normalized local features
    except Exception as e:
        return False, f"load-fail:{str(e)[:40]}"

    names = list(payload.get("local_features", []))
    Xmean = payload["X_local_mean"]
    Xstd = payload["X_local_std"]
    Xmean = Xmean.cpu().numpy() if hasattr(Xmean, "cpu") else np.asarray(Xmean)
    Xstd = Xstd.cpu().numpy() if hasattr(Xstd, "cpu") else np.asarray(Xstd)
    log_feats = set(payload.get("log_features", []))

    def phys(nm):
        i = names.index(nm)
        v = x_norm[i] * float(Xstd[i]) + float(Xmean[i])
        return 10.0 ** v if nm in log_feats else v

    try:
        via = phys("VIA_RADIUS")
        antipad = phys("ANTIPAD_RADIUS")
        pitch = phys("PITCH")
        eps_r = phys("PERMITTIVITY")
    except ValueError as e:
        return False, f"missing-feature:{e}"

    problems = []
    if not (via < antipad):
        problems.append(f"via({via:.2f})>=antipad({antipad:.2f})")
    if not (2 * antipad < pitch):
        problems.append(f"2*antipad({2*antipad:.2f})>=pitch({pitch:.2f})")
    if not (eps_r > 1.0):
        problems.append(f"eps_r({eps_r:.2f})<=1")
    if problems:
        return False, "; ".join(problems)
    return True, f"via{via:.1f}<ap{antipad:.1f}<p/2 eps{eps_r:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50,
                    help="number of RANDOM validation samples to evaluate")
    ap.add_argument("--seed", type=int, default=123,
                    help="RNG seed for which validation samples are drawn "
                         "(separate from the split seed; change to resample)")
    ap.add_argument("--restarts", type=int, default=12)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--fit-gate", type=float, default=5.0,
                    help="fit gate passed to the yield stage. Kept loose (5.0) "
                         "so poor reconstructions are still SCORED (yield read) "
                         "rather than gate-rejected -- we WANT their real yield.")
    ap.add_argument("--stage1-only", action="store_true",
                    help="only run reconstruction (fit-only); skip yield stage")
    ap.add_argument("--out", type=str, default="validation_eval.csv")
    args = ap.parse_args()

    # import the yield module (provides run_yield_tto)
    try:
        mod = importlib.import_module(IMPORT_NAME)
    except ModuleNotFoundError:
        sys.exit(f"[fatal] could not import '{IMPORT_NAME}'. Edit IMPORT_NAME "
                 f"at the top of this script to your yield-script filename "
                 f"(without .py), and run from the same directory.")
    if not hasattr(mod, "run_yield_tto"):
        sys.exit(f"[fatal] '{IMPORT_NAME}' has no run_yield_tto(); is it the "
                 f"right file?")

    # draw the random validation subset
    val_idx, sim_ids = get_validation_indices()
    if args.n > len(val_idx):
        args.n = len(val_idx)
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(val_idx, size=args.n, replace=False)
    chosen = np.sort(chosen)

    # payload (for geometry de-normalization) + design output directory
    _payload = torch.load(DATA_PT, map_location="cpu", weights_only=False)
    design_dir = (Path(__file__).resolve().parent / "evaluation_results"
                  / "generated_designs")

    print("=" * 74)
    print(f"  VALIDATION-SET EVALUATION  |  {args.n} random samples  "
          f"(seed {args.seed})")
    print(f"  success criteria: reconstruction fit <= {FIT_OK} dB, "
          f"yield >= {YIELD_OK}%")
    print("=" * 74)

    results = []
    t0 = time.time()
    for k, idx in enumerate(chosen):
        idx = int(idx)
        sid = sim_ids[idx]
        row = {"sample": idx, "sim_id": str(sid)}

        # ---- STAGE 1: reconstruction accuracy (fit-only) -----------------
        # lambda_j=0, mode=variance == plain latent TTO; we read rank-1 fit.
        try:
            r1 = mod.run_yield_tto(
                idx, tto_steps=args.steps, restarts=args.restarts,
                mode="variance", lambda_j=0.0, fit_gate=args.fit_gate)
            best_fit = min(r["fit_eye_dB"] for r in r1)
            # yield of the fit-only design (its nominal robustness)
            fitonly_yield = max(r["yield_pct"] for r in r1
                                if abs(r["fit_eye_dB"] - best_fit) < 1e-6)
            row["recon_fit_dB"] = round(best_fit, 3)
            row["fitonly_yield_pct"] = round(fitonly_yield, 1)
        except Exception as e:
            row["recon_fit_dB"] = None
            row["fitonly_yield_pct"] = None
            row["error_stage1"] = str(e)[:120]

        # ---- STAGE 2: manufacturability (worst-case yield) ---------------
        if not args.stage1_only:
            try:
                r2 = mod.run_yield_tto(
                    idx, tto_steps=args.steps, restarts=args.restarts,
                    mode="worstcase", fit_gate=args.fit_gate)
                # rank-1 is best yield among (loosely) gated designs
                top = r2[0]
                row["wc_yield_pct"] = round(top["yield_pct"], 1)
                row["wc_fit_dB"] = round(top["fit_eye_dB"], 3)
                row["wc_maxabs_x"] = top.get("max_abs_xnorm")
                # GEOMETRY VALIDITY of the winning design (buildability check)
                top_npz = design_dir / top["npz"]
                gv, gdetail = check_geometry_valid(top_npz, _payload)
                row["wc_geom_valid"] = gv
                row["wc_geom_detail"] = gdetail
            except Exception as e:
                row["wc_yield_pct"] = None
                row["wc_fit_dB"] = None
                row["wc_geom_valid"] = None
                row["error_stage2"] = str(e)[:120]

        # ---- verdicts ----------------------------------------------------
        rf = row.get("recon_fit_dB")
        wy = row.get("wc_yield_pct")
        gv = row.get("wc_geom_valid")
        row["recon_ok"] = (rf is not None and rf <= FIT_OK)
        # manufacturable requires BOTH sufficient yield AND buildable geometry
        # (stage1-only runs have no geometry check -> treat as N/A True)
        geom_ok = True if args.stage1_only else bool(gv)
        row["mfg_ok"] = (wy is not None and wy >= YIELD_OK and geom_ok)
        row["full_ok"] = bool(row["recon_ok"] and row["mfg_ok"])

        results.append(row)
        elapsed = time.time() - t0
        eta = elapsed / (k + 1) * (args.n - k - 1)
        print(f"[{k+1:>3d}/{args.n}] sample {idx:>5d} ({sid}) | "
              f"recon_fit={row.get('recon_fit_dB')} dB | "
              f"wc_yield={row.get('wc_yield_pct')}% | "
              f"geom={row.get('wc_geom_valid')} | "
              f"recon_ok={row['recon_ok']} mfg_ok={row['mfg_ok']} | "
              f"ETA {eta/60:.1f} min")

    # ---- write CSV -------------------------------------------------------
    keys = sorted({k for r in results for k in r.keys()})
    out_path = Path(args.out)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)

    # ---- SUMMARY ---------------------------------------------------------
    def pct(cond):
        n = sum(1 for r in results if cond(r))
        return 100.0 * n / len(results), n

    fits = [r["recon_fit_dB"] for r in results if r.get("recon_fit_dB") is not None]
    yields = [r["wc_yield_pct"] for r in results if r.get("wc_yield_pct") is not None]

    print("\n" + "=" * 74)
    print(f"  SUMMARY over {len(results)} random validation samples")
    print("=" * 74)
    print("\n  STAGE 1 -- RECONSTRUCTION ACCURACY (inverse model quality):")
    if fits:
        print(f"    median fit = {np.median(fits):.2f} dB   "
              f"mean = {np.mean(fits):.2f} dB   "
              f"best = {min(fits):.2f}   worst = {max(fits):.2f}")
        p, n = pct(lambda r: r["recon_ok"])
        print(f"    reconstructed within {FIT_OK} dB : {n}/{len(results)} "
              f"= {p:.0f}%")
    print("\n  STAGE 2 -- MANUFACTURABILITY (worst-case yield):")
    if yields:
        print(f"    median yield = {np.median(yields):.1f}%   "
              f"mean = {np.mean(yields):.1f}%")
        p, n = pct(lambda r: r.get("wc_yield_pct") is not None
                   and r["wc_yield_pct"] >= YIELD_OK)
        print(f"    achieved yield >= {YIELD_OK}% : {n}/{len(results)} = {p:.0f}%")
        pg, ng = pct(lambda r: r.get("wc_geom_valid") is True)
        print(f"    geometry buildable         : {ng}/{len(results)} = {pg:.0f}%")
        p2, n2 = pct(lambda r: r.get("wc_yield_pct") is not None
                     and r["wc_yield_pct"] >= 70.0)
        print(f"    (stricter) yield >= 70%    : {n2}/{len(results)} = {p2:.0f}%")
    print("\n  COMBINED:")
    p, n = pct(lambda r: r["full_ok"])
    print(f"    FULLY successful (both criteria): {n}/{len(results)} = {p:.0f}%")

    # simple Wilson-ish note (95% ~ +/- 1.96*sqrt(p(1-p)/n))
    if results:
        p_full = n / len(results)
        se = (p_full * (1 - p_full) / len(results)) ** 0.5
        print(f"    (approx 95% CI on full-success rate: "
              f"{100*(p_full-1.96*se):.0f}% - {100*(p_full+1.96*se):.0f}%)")

    print(f"\n  per-sample CSV written: {out_path}")
    print("=" * 74)


if __name__ == "__main__":
    main()