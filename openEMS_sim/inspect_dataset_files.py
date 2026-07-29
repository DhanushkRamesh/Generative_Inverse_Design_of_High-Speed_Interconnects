"""
inspect_datafiles.py
================================================================================
Stage 00 of the openEMS validation pipeline.

PURPOSE
    Locate and dump the raw TUHH dataset definition files (stackup, via array,
    parameter variations) so the geometry parser (stage 03) can be written
    against the REAL file formats instead of assumptions.

    The TUHH documentation states that structural files contain parameter
    placeholders (capital letters in brackets) whose per-simulation values
    live in a parameter CSV. This script finds those files and prints their
    structure so the substitution mechanism can be replicated exactly.

WHAT IT DOES
    1. Recursively searches the project data directories for candidate files:
       stackup*, via*, parameter*, *.csv, *.txt, *.json (excluding the large
       touchstone / processed tensors).
    2. For each candidate: prints path, size, and the first lines (text) so
       the format is visible.
    3. Loads the processed diff_pair_dataset.pt and prints its keys, shapes,
       and normalization metadata if present - needed later to de-normalize
       inverse-model outputs back to physical dimensions.
    4. Prints a compact summary to paste back for writing stage 03.

USAGE
    cd ~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/openEMS_Sim
    python inspect_dataset_files.py
    python inspect_dataset_files.py --max-lines 40      # more lines per file
    python inspect_dataset_files.py --search-root /path # override search root

This script is read-only. It modifies nothing.
"""

import argparse
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
PROJECT_ROOT = (
    Path.home()
    / "mece_project_inverse_model"
    / "Generative_Inverse_Design_of_High-Speed_Interconnects"
)
DATA_ROOT = PROJECT_ROOT / "data"
PROCESSED_PT = DATA_ROOT / "processed" / "Universal-Diff-SI-Array" / "diff_pair_dataset.pt"

# Filename patterns that likely hold geometry / stackup / parameter definitions
CANDIDATE_PATTERNS = [
    "*stackup*", "*Stackup*", "*STACKUP*",
    "*via*", "*Via*", "*VIA*",
    "*parameter*", "*Parameter*", "*PARAMETER*",
    "*material*", "*Material*",
    "*.csv", "*.txt", "*.json", "*.info", "*.cfg", "*.ini",
]

# Extensions/names to skip (big data files, not definitions)
SKIP_SUFFIXES = {".pt", ".npy", ".npz", ".png", ".pdf", ".zip", ".tar", ".gz",
                 ".s4p", ".s8p", ".s16p", ".s32p", ".s80p", ".snp", ".h5",
                 ".hdf5", ".pkl", ".ipynb"}
# Touchstone files can have arbitrary .sNp suffixes - catch them generically
def is_touchstone(p: Path) -> bool:
    s = p.suffix.lower()
    return s.startswith(".s") and s.endswith("p") and s[2:-1].isdigit()

MAX_FILES_PER_DIR_LISTING = 20   # cap directory listings
MAX_TOTAL_DUMPS = 40             # cap number of files dumped in full


def human_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def dump_text_head(path: Path, max_lines: int) -> None:
    """Print the first max_lines of a text file, escaping decode issues."""
    try:
        with open(path, "r", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    print(f"      ... (truncated at {max_lines} lines)")
                    break
                # rstrip to keep output tidy; repr-ish for weird whitespace
                print(f"      {i:3d} | {line.rstrip()}")
    except Exception as e:
        print(f"      [could not read as text: {e}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-lines", type=int, default=25,
                    help="lines to show per candidate file")
    ap.add_argument("--search-root", type=str, default=str(DATA_ROOT),
                    help="root directory to search (default: project data/)")
    args = ap.parse_args()

    search_root = Path(args.search_root)
    print("=" * 78)
    print("Stage 00: raw dataset definition file inspection")
    print("=" * 78)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Search root:  {search_root}")

    if not search_root.exists():
        print(f"\nERROR: search root does not exist: {search_root}")
        print("Pass the correct location with --search-root /path/to/raw/data")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1) Top-level map of the data directory (2 levels), so we can see how
    #    the raw dataset is organised (per-sim folders? flat? archives?)
    # ------------------------------------------------------------------
    print("\n--- Directory map (2 levels) ---")
    try:
        for child in sorted(search_root.iterdir()):
            print(f"  {child.name}{'/' if child.is_dir() else ''}")
            if child.is_dir():
                entries = sorted(child.iterdir())
                for sub in entries[:MAX_FILES_PER_DIR_LISTING]:
                    print(f"      {sub.name}{'/' if sub.is_dir() else ''}")
                if len(entries) > MAX_FILES_PER_DIR_LISTING:
                    print(f"      ... (+{len(entries) - MAX_FILES_PER_DIR_LISTING} more)")
    except PermissionError as e:
        print(f"  [permission error walking directory: {e}]")

    # ------------------------------------------------------------------
    # 2) Find candidate definition files
    # ------------------------------------------------------------------
    print("\n--- Candidate definition files ---")
    seen = set()
    candidates = []
    for pattern in CANDIDATE_PATTERNS:
        for p in search_root.rglob(pattern):
            if not p.is_file():
                continue
            if p in seen:
                continue
            if p.suffix.lower() in SKIP_SUFFIXES or is_touchstone(p):
                continue
            seen.add(p)
            candidates.append(p)

    if not candidates:
        print("  No candidate files found under the search root.")
        print("  If the raw dataset lives elsewhere, re-run with --search-root.")
    else:
        # Sort: shallower paths first (top-level definitions), then by name
        candidates.sort(key=lambda p: (len(p.parts), str(p)))
        print(f"  Found {len(candidates)} candidate file(s). Dumping up to "
              f"{MAX_TOTAL_DUMPS}:\n")
        for idx, p in enumerate(candidates[:MAX_TOTAL_DUMPS]):
            rel = p.relative_to(search_root)
            print(f"  [{idx:02d}] {rel}   ({human_size(p.stat().st_size)})")
            dump_text_head(p, args.max_lines)
            print()
        if len(candidates) > MAX_TOTAL_DUMPS:
            print(f"  ... (+{len(candidates) - MAX_TOTAL_DUMPS} more files not "
                  f"dumped; re-run with --search-root narrowed if needed)")

    # ------------------------------------------------------------------
    # 3) Processed dataset metadata (needed for de-normalisation later)
    # ------------------------------------------------------------------
    print("\n--- Processed dataset (diff_pair_dataset.pt) ---")
    if not PROCESSED_PT.exists():
        print(f"  Not found at {PROCESSED_PT} (skip)")
    else:
        try:
            import torch
            payload = torch.load(PROCESSED_PT, weights_only=False,
                                 map_location="cpu")
            print(f"  Keys: {sorted(payload.keys())}")
            for k, v in payload.items():
                try:
                    import numpy as np
                    if hasattr(v, "shape"):
                        print(f"    {k:20s} shape={tuple(v.shape)} "
                              f"dtype={getattr(v, 'dtype', '?')}")
                    elif isinstance(v, (list, tuple)):
                        print(f"    {k:20s} {type(v).__name__} len={len(v)} "
                              f"first={v[0] if len(v) else None}")
                    elif isinstance(v, dict):
                        print(f"    {k:20s} dict keys={sorted(v.keys())[:12]}")
                    else:
                        print(f"    {k:20s} {type(v).__name__}: {str(v)[:80]}")
                except Exception as e:
                    print(f"    {k:20s} [inspect error: {e}]")
            # Normalisation metadata is essential for stage 04's
            # feature -> physical-dimension inversion. Flag it explicitly.
            norm_keys = [k for k in payload.keys()
                         if any(t in k.lower()
                                for t in ("mean", "std", "scale", "norm", "feature"))]
            print(f"\n  Normalisation-related keys detected: {norm_keys if norm_keys else 'NONE'}")
            if not norm_keys:
                print("  NOTE: no mean/std stored in the payload. The z-score"
                      " statistics must be recovered from the preprocessing"
                      " script - flag this when pasting output back.")
        except ImportError:
            print("  torch not importable in this environment (skip)")
        except Exception as e:
            print(f"  Could not load: {e}")

   

if __name__ == "__main__":
    main()