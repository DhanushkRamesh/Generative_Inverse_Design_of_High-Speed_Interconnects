"""
parse_diff_pairs.py  (v2 - patched after debug_one_sim.py findings)
==================================================================
Main parsing pipeline.  Produces one training sample per differential pair
across the TUHH SI/PI Universal-Diff-SI-{Array,Link} datasets.

Patches relative to v1
----------------------
1. Frequency grid is now  0.25 .. 100 GHz, 401 points.  The original
   0..100 GHz grid required linear extrapolation from 0.25 GHz down to
   DC, which produced |S| slightly above 1 on the through-paths (because
   S21(0.25 GHz) is essentially 1 for a passive through-via and the
   linear-slope extrapolation overshoots).  That destroyed passivity
   even though the raw simulation data was perfectly fine.  Eliminating
   extrapolation eliminates the artefact.

2. Passivity check is performed BEFORE interpolation, on the FULL N-port
   network, with a relaxed threshold (default -1e-3).  Slicing a 4x4
   submatrix from an N-port network does not preserve passivity even when
   the full network is strictly passive: the sub-system can show
   min(eig(I - S^H S)) around -0.2 because the excluded ports are not
   terminated in matched loads.  This is not a "violation" -- it is
   what slicing always produces -- and checking passivity in the right
   place (full N-port, before extraction) avoids the spurious drops.
   Refs:
     Triverio et al., "Stability, Causality, and Passivity in
       Electrical Interconnect Models," IEEE TADVP 2007, sec. 4.2.
     Grivet-Talocia & Gustavsen, "Passive Macromodeling: Theory and
       Applications," Wiley 2016, sec. 11.4.

3. skipped_pairs.csv is written via try/finally so we always get the
   diagnostic file, even when zero pairs make it through.

Run as:
    cd sandbox_v1
    python data/parse_diff_pairs.py --dataset Array
    python data/parse_diff_pairs.py --dataset Link
"""

from __future__ import annotations
import sys
from pathlib import Path

# sys.path bootstrap so we can run as `python data/parse_diff_pairs.py`
# from inside sandbox_v1/ without needing __init__.py files anywhere.
_SANDBOX_ROOT = Path(__file__).resolve().parent.parent
if str(_SANDBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(_SANDBOX_ROOT))

import argparse
import datetime as _dt
import subprocess

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import skrf as rf

from utils.physics_utils import (                                       # noqa: E402
    convert_to_mixed_mode,
    enforce_reciprocity,
    check_passivity,
    reciprocity_residual,
)
from data.parse_via_array import (                                      # noqa: E402
    parse_via_array,
    identify_diff_pairs,
    diff_pair_port_indices,
    context_vector,
    CONTEXT_FEATURE_NAMES,
)


# ---------------------------------------------------------------------------
# Feature contract (unchanged)
# ---------------------------------------------------------------------------
LOCAL_FEATURES_BASE = [
    "VIA_RADIUS", "ANTIPAD_RADIUS", "PITCH",
    "TDIEL", "TMET",
    "PERMITTIVITY", "CONDUCTIVITY", "LOSSTANGENT",
]
GLOBAL_FEATURES_BASE = [
    "LAYER_AMOUNT", "VIAS_X_AMOUNT", "VIAS_Y_AMOUNT",
    "SIGNAL_AMOUNT", "GROUND_AMOUNT", "POWER_AMOUNT",
]
LOG_FEATURES = {"CONDUCTIVITY", "LOSSTANGENT", "SL_WIDTH", "LENGTH"}

# ---------------------------------------------------------------------------
# PATCH (1): standard frequency grid.  Was 0..100 GHz, 401 pts;  now
# 0.25..100 GHz, 401 pts.  This stays WITHIN the raw data's range so no
# extrapolation happens during scipy interp1d.
# ---------------------------------------------------------------------------
FREQ_MIN_HZ   = 0.25e9
FREQ_MAX_HZ   = 100e9
FREQ_N_POINTS = 401


def _get_git_hash() -> str:
    """Return short git hash for traceability, or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, cwd=_SANDBOX_ROOT,
        ).decode().strip()
    except Exception:
        return "unknown"


def _feature_lists(dataset_type: str) -> tuple[list[str], list[str]]:
    if dataset_type == "Array":
        return LOCAL_FEATURES_BASE.copy(), GLOBAL_FEATURES_BASE.copy()
    if dataset_type == "Link":
        return (LOCAL_FEATURES_BASE + ["SL_WIDTH"],
                GLOBAL_FEATURES_BASE + ["LENGTH"])
    raise ValueError(f"Unknown dataset_type: {dataset_type!r}")


def _load_parameter_csv(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "parameter.csv")
    if "LOSTANGENT" in df.columns and "LOSSTANGENT" not in df.columns:
        df = df.rename(columns={"LOSTANGENT": "LOSSTANGENT"})
    if "CONDUCTI" in df.columns and "CONDUCTIVITY" not in df.columns:
        df = df.rename(columns={"CONDUCTI": "CONDUCTIVITY"})
    return df


def _find_touchstone(sim_dir: Path) -> Path | None:
    matches = list(sim_dir.glob("*.s*p"))
    return matches[0] if matches else None


def _apply_log10_inplace(
    arr: np.ndarray, feature_names: list[str], log_features: set
) -> None:
    for col_idx, name in enumerate(feature_names):
        if name in log_features:
            arr[:, col_idx] = np.log10(np.clip(arr[:, col_idx], 1e-12, None))


def _zscore_normalise(
    arr: np.ndarray, stats_indices: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = arr if stats_indices is None else arr[stats_indices]
    mean = subset.mean(axis=0)
    std = np.maximum(subset.std(axis=0), 1e-12)
    return (arr - mean) / std, mean, std


def _save_skipped_csv(skipped: list[dict], out_dir: Path) -> Path:
    """Always-callable; writes the skipped CSV even on early abort."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "skipped_pairs.csv"
    if not skipped:
        pd.DataFrame(columns=["sim_id", "pair_id", "reason"]).to_csv(path, index=False)
    else:
        pd.DataFrame(skipped).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def parse_dataset(
    dataset_type: str,
    raw_dir: Path,
    out_dir: Path,
    full_passivity_threshold: float = -1e-3,
    max_sims: int | None = None,
    reciprocity_warn_threshold: float = 1e-2,
) -> dict:
    """Parse a full dataset and save a .pt with all diff-pair samples."""
    out_dir.mkdir(parents=True, exist_ok=True)
    local_features, global_features = _feature_lists(dataset_type)

    print(f"\n{'=' * 70}")
    print(f"  Parsing dataset: {dataset_type}")
    print(f"  Raw dir:         {raw_dir}")
    print(f"  Output dir:      {out_dir}")
    print(f"  Freq grid:       {FREQ_MIN_HZ/1e9:.2f} .. {FREQ_MAX_HZ/1e9:.1f} GHz, "
          f"{FREQ_N_POINTS} points (NO extrapolation)")
    print(f"  Full-network passivity threshold: {full_passivity_threshold:+.0e}")
    print(f"  Local features:  {len(local_features)}  {local_features}")
    print(f"  Global features: {len(global_features)} {global_features}")
    print(f"  Context features:{len(CONTEXT_FEATURE_NAMES)} {CONTEXT_FEATURE_NAMES}")
    print(f"{'=' * 70}\n")

    df = _load_parameter_csv(raw_dir)
    if max_sims is not None:
        df = df.head(max_sims)
        print(f"  [DEBUG] Limited to first {max_sims} simulations.\n")

    std_freq = rf.Frequency(
        FREQ_MIN_HZ / 1e9, FREQ_MAX_HZ / 1e9, FREQ_N_POINTS, unit="ghz"
    )

    rows: list[dict] = []
    skipped: list[dict] = []
    diagnostic_min_eigs: list[float] = []

    # try/finally so skipped_pairs.csv ALWAYS gets written.
    try:
        for _, sim_row in tqdm(df.iterrows(), total=len(df), desc="Sims"):
            sim_id = str(sim_row["SIMULATION"])
            sim_dir = raw_dir / "variation" / sim_id

            ts_path = _find_touchstone(sim_dir)
            if ts_path is None:
                skipped.append(dict(sim_id=sim_id, pair_id=None,
                                    reason="no touchstone file"))
                continue
            try:
                ntwk = rf.Network(str(ts_path))
            except Exception as exc:
                skipped.append(dict(sim_id=sim_id, pair_id=None,
                                    reason=f"skrf load error: {exc}"))
                continue

            num_ports = ntwk.s.shape[1]
            
            if dataset_type == "Array" and num_ports != 2 * int(sim_row["SIGNAL_AMOUNT"]):
                skipped.append(dict(
                    sim_id=sim_id, pair_id=None,
                    reason=f"port count {num_ports} != 2*SIGNAL_AMOUNT "
                           f"({2 * int(sim_row['SIGNAL_AMOUNT'])})"
                ))
                continue

            # ----- PATCH (2): full-N-port passivity BEFORE interpolation ---
            # This is the only physically meaningful passivity check.
            is_full_passive, full_min_eig = check_passivity(
                ntwk.s.astype(np.complex128), full_passivity_threshold
            )
            diagnostic_min_eigs.append(full_min_eig)
            if not is_full_passive:
                skipped.append(dict(
                    sim_id=sim_id, pair_id=None,
                    reason=f"full N-port non-passive: min_eig={full_min_eig:.2e}"
                ))
                continue

            # Interpolate to standard grid (in-band, no extrapolation needed).
            try:
                ntwk = ntwk.interpolate(
                    std_freq, bounds_error=False, fill_value="extrapolate"
                )
            except Exception as exc:
                skipped.append(dict(sim_id=sim_id, pair_id=None,
                                    reason=f"interpolation error: {exc}"))
                continue

            via_path = sim_dir / "via_array.txt"
            if not via_path.exists():
                skipped.append(dict(sim_id=sim_id, pair_id=None,
                                    reason="no via_array.txt"))
                continue
            try:
                parsed = parse_via_array(via_path)
            except Exception as exc:
                skipped.append(dict(sim_id=sim_id, pair_id=None,
                                    reason=f"via_array parse error: {exc}"))
                continue

            diff_pairs = identify_diff_pairs(parsed, dataset_type)
            
            if dataset_type == "Array":
                expected_pairs = int(sim_row["SIGNAL_AMOUNT"]) // 2
                if len(diff_pairs) != expected_pairs:
                    skipped.append(dict(
                        sim_id=sim_id, pair_id=None,
                        reason=f"diff-pair count {len(diff_pairs)} != expected {expected_pairs}"
                    ))
                    continue
            elif dataset_type == "Link":
                if len(diff_pairs) == 0:
                    skipped.append(dict(
                        sim_id=sim_id, pair_id=None,
                        reason="No valid MTL diff links found"
                    ))
                    continue

            vias_x = int(sim_row["VIAS_X_AMOUNT"])
            vias_y = int(sim_row["VIAS_Y_AMOUNT"])

            x_local_raw  = np.array([sim_row[f] for f in local_features],  dtype=np.float64)
            x_global_raw = np.array([sim_row[f] for f in global_features], dtype=np.float64)

            for pair in diff_pairs:
                pair_id = pair["pair_id"]
                try:
                    port_idx = pair["port_idx"]
                    if max(port_idx) >= num_ports:
                        raise ValueError(f"Pair {pair_id} out of range for {dataset_type} file with {num_ports} ports (computed indices {port_idx})")
                except ValueError as exc:
                    skipped.append(dict(sim_id=sim_id, pair_id=pair_id,
                                        reason=f"port-index error: {exc}"))
                    continue

                s_se = ntwk.s[:, port_idx][:, :, port_idx].astype(np.complex128)

                # Reciprocity audit (data is already very clean; this is
                # essentially a no-op but harmless).
                recip_raw = reciprocity_residual(s_se)
                s_se = enforce_reciprocity(s_se)

                # NO per-pair passivity check (see PATCH 2 explanation).
                s_mm = convert_to_mixed_mode(s_se)

                try:
                    x_context_raw = context_vector(
                        pair, parsed["grid"], vias_x, vias_y
                    )
                except Exception as exc:
                    skipped.append(dict(
                        sim_id=sim_id, pair_id=pair_id,
                        reason=f"context-feature error: {exc}"
                    ))
                    continue

                if recip_raw > reciprocity_warn_threshold:
                    skipped.append(dict(
                        sim_id=sim_id, pair_id=pair_id,
                        reason=f"WARN large recip residual {recip_raw:.2e} (kept)"
                    ))

                rows.append(dict(
                    sim_id=sim_id, pair_id=pair_id, num_ports=num_ports,
                    x_local=x_local_raw, x_global=x_global_raw,
                    x_context=x_context_raw,
                    y_real=s_mm.real.astype(np.float32),
                    y_imag=s_mm.imag.astype(np.float32),
                ))

    finally:
        skipped_path = _save_skipped_csv(skipped, out_dir)
        print(f"\n  Skipped log -> {skipped_path}  ({len(skipped)} rows)")
        if diagnostic_min_eigs:
            d = np.array(diagnostic_min_eigs)
            print(f"  Full-network min_eig over {len(d)} sims:  "
                  f"min={d.min():.2e}  max={d.max():.2e}  median={np.median(d):.2e}")

    if not rows:
        raise RuntimeError(
            "No diff pairs successfully parsed.  "
            f"See {skipped_path} for reasons."
        )

    # ----- Stack into arrays -----------------------------------------------
    n_pairs = len(rows)
    print(f"\n  Successfully parsed {n_pairs} diff pairs from "
          f"{df.shape[0]} simulations.")

    sim_ids       = [r["sim_id"] for r in rows]
    pair_ids      = np.array([r["pair_id"]   for r in rows], dtype=np.int64)
    num_ports_arr = np.array([r["num_ports"] for r in rows], dtype=np.int64)
    X_local       = np.stack([r["x_local"]   for r in rows])
    X_global      = np.stack([r["x_global"]  for r in rows])
    X_context     = np.stack([r["x_context"] for r in rows])
    Y_real        = np.stack([r["y_real"]    for r in rows])
    Y_imag        = np.stack([r["y_imag"]    for r in rows])

    _apply_log10_inplace(X_local,  local_features,  LOG_FEATURES)
    _apply_log10_inplace(X_global, global_features, LOG_FEATURES)

    sim_id_array = np.array(sim_ids)
    _, first_idx_per_sim = np.unique(sim_id_array, return_index=True)

    X_local_n,   X_local_mean,   X_local_std   = _zscore_normalise(
        X_local,  stats_indices=first_idx_per_sim)
    X_global_n,  X_global_mean,  X_global_std  = _zscore_normalise(
        X_global, stats_indices=first_idx_per_sim)
    X_context_n, X_context_mean, X_context_std = _zscore_normalise(X_context)

    save_path = out_dir / "diff_pair_dataset.pt"
    payload = {
        "dataset_type": dataset_type,
        "X_local":   torch.from_numpy(X_local_n.astype(np.float32)),
        "X_global":  torch.from_numpy(X_global_n.astype(np.float32)),
        "X_context": torch.from_numpy(X_context_n.astype(np.float32)),
        "Y_real":    torch.from_numpy(Y_real),
        "Y_imag":    torch.from_numpy(Y_imag),
        "sim_ids":      sim_ids,
        "pair_ids":     torch.from_numpy(pair_ids),
        "num_ports":    torch.from_numpy(num_ports_arr),
        "X_local_mean":   torch.from_numpy(X_local_mean.astype(np.float32)),
        "X_local_std":    torch.from_numpy(X_local_std.astype(np.float32)),
        "X_global_mean":  torch.from_numpy(X_global_mean.astype(np.float32)),
        "X_global_std":   torch.from_numpy(X_global_std.astype(np.float32)),
        "X_context_mean": torch.from_numpy(X_context_mean.astype(np.float32)),
        "X_context_std":  torch.from_numpy(X_context_std.astype(np.float32)),
        "local_features":   local_features,
        "global_features":  global_features,
        "context_features": list(CONTEXT_FEATURE_NAMES),
        "log_features":     [f for f in (local_features + global_features)
                             if f in LOG_FEATURES],
        "frequencies": torch.from_numpy(
            np.linspace(FREQ_MIN_HZ, FREQ_MAX_HZ, FREQ_N_POINTS).astype(np.float64)
        ),
        "metadata": {
            "creation_date": _dt.datetime.now().isoformat(timespec="seconds"),
            "git_hash": _get_git_hash(),
            "num_simulations_processed": int(df.shape[0]),
            "num_pairs_accepted": int(n_pairs),
            "num_entries_skipped_or_warned": int(len(skipped)),
            "full_passivity_threshold": float(full_passivity_threshold),
            "freq_min_hz": float(FREQ_MIN_HZ),
            "freq_max_hz": float(FREQ_MAX_HZ),
            "freq_n_points": int(FREQ_N_POINTS),
            "max_sims": max_sims,
        },
    }
    torch.save(payload, save_path)
    print(f"\n  Saved -> {save_path}  ({save_path.stat().st_size / 1e6:.1f} MB)")

    print(f"\n  Tensor shapes:")
    for k in ("X_local", "X_global", "X_context", "Y_real", "Y_imag"):
        print(f"    {k:11s} {tuple(payload[k].shape)}")

    sdd21_db = 20 * np.log10(np.abs(Y_real[:, 0, 1, 0] + 1j * Y_imag[:, 0, 1, 0]) + 1e-12)
    sdd11_db = 20 * np.log10(np.abs(Y_real[:, 0, 0, 0] + 1j * Y_imag[:, 0, 0, 0]) + 1e-12)
    print(f"\n  Physics sanity at f={FREQ_MIN_HZ/1e9:.2f} GHz:")
    print(f"    Sdd21 mean = {sdd21_db.mean():+.2f} dB    "
          f"(expected near 0 dB for a passive through-path)")
    print(f"    Sdd11 mean = {sdd11_db.mean():+.2f} dB    "
          f"(expected well below 0 dB)")

    return {"n_pairs": n_pairs, "n_skipped": len(skipped),
            "save_path": save_path, "skipped_path": skipped_path}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _default_paths(dataset_type: str) -> tuple[Path, Path]:
    project_root = _SANDBOX_ROOT.parent
    raw_dir = project_root / "data" / "raw" / f"Universal-Diff-SI-{dataset_type}"
    out_dir = project_root / "data" / "processed" / f"Universal-Diff-SI-{dataset_type}"
    return raw_dir, out_dir


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["Array", "Link"])
    ap.add_argument("--raw-dir",  type=Path, default=None)
    ap.add_argument("--out-dir",  type=Path, default=None)
    ap.add_argument("--max-sims", type=int, default=None,
                    help="Limit to first N sims (testing).")
    ap.add_argument("--passivity-threshold", type=float, default=-1e-3,
                    help="Lower bound on full-N-port min_eig (default -1e-3).")
    args = ap.parse_args()

    raw_dir, out_dir = _default_paths(args.dataset)
    if args.raw_dir is not None: raw_dir = args.raw_dir
    if args.out_dir is not None: out_dir = args.out_dir

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw dataset directory not found: {raw_dir}\n"
            f"Use --raw-dir to override."
        )

    parse_dataset(
        dataset_type=args.dataset,
        raw_dir=raw_dir,
        out_dir=out_dir,
        full_passivity_threshold=args.passivity_threshold,
        max_sims=args.max_sims,
    )


if __name__ == "__main__":
    _cli()