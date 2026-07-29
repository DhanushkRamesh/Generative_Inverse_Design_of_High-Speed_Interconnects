"""
test_parse_diff_pairs_synthetic.py
==================================
Builds a tiny synthetic dataset on disk (fake parameter.csv, fake touchstone
files, fake via_array.txt) and runs parse_diff_pairs.py against it.

This is a pure integration test: it does not exercise the *physics* of the
parser (synthetic touchstones are not realistic PCB responses), only the
*plumbing* — that we can read the inputs, slice the right ports, compute
context features, and write a well-formed .pt file.

Run from sandbox_v1/:
    python data/test_parse_diff_pairs_synthetic.py
"""

import sys
import shutil
from pathlib import Path

# Make the sandbox importable without __init__.py files
SANDBOX_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SANDBOX_ROOT))

import numpy as np
import pandas as pd
import skrf as rf
import torch

from data.parse_diff_pairs import parse_dataset


def make_synthetic_touchstone(num_ports: int, path: Path, seed: int) -> None:
    """Write a small passive reciprocal 4-port touchstone at the given path."""
    rng = np.random.default_rng(seed)
    n_freqs = 21
    freqs = np.linspace(0.25e9, 100e9, n_freqs)

    # Build a passive S = 0.1 * (random symmetric complex).  |S| small enough
    # to be passive in I - S^H S.
    s = np.zeros((n_freqs, num_ports, num_ports), dtype=complex)
    for f in range(n_freqs):
        A = (rng.standard_normal((num_ports, num_ports))
             + 1j * rng.standard_normal((num_ports, num_ports))) * 0.05
        s[f] = 0.5 * (A + A.T)   # symmetrise for reciprocity

    freq_obj = rf.Frequency.from_f(freqs, unit="hz")
    ntwk = rf.Network(frequency=freq_obj, s=s, z0=50.0)
    ntwk.write_touchstone(str(path.with_suffix("")))


def make_fake_array_via_array(path: Path, signal_amount: int,
                              vias_x: int, vias_y: int) -> None:
    """Write a minimal via_array.txt for an Array sim with `signal_amount` signals."""
    # Place signal vias along a row, ground vias filling the rest of the grid
    lines = [
        f"layer list: ['G', 'D/2', 'S', 'D/2', 'G']",
        "amount of layers: 3",
        "[layer-drawing]",
        "(omitted)",
    ]
    # Port list: 2 ports per signal via, interleaved (pup then plw)
    # Place sgn1 at (1,1), sgn2 at (1,2), sgn3 at (2,1), sgn4 at (2,2), etc.
    for sgn in range(1, signal_amount + 1):
        # Walk a snake-pattern through the grid
        x = ((sgn - 1) // vias_y) + 1
        y = ((sgn - 1) %  vias_y) + 1
        lines.append(f"arr_1_via_{x}_{y}_sgn{sgn}pup")
        lines.append(f"arr_1_via_{x}_{y}_sgn{sgn}plw")

    # ARRAY grid: fill with G, then overwrite with S where we placed signals
    lines.append("[ARRAY]")
    grid = [["G1"] * vias_x for _ in range(vias_y)]
    for sgn in range(1, signal_amount + 1):
        x = ((sgn - 1) // vias_y) + 1
        y = ((sgn - 1) %  vias_y) + 1
        # row index from top = vias_y - y (since y=1 is bottom)
        row_idx_top = vias_y - y
        col_idx = x - 1
        grid[row_idx_top][col_idx] = f"S{sgn}"
    for row in grid:
        lines.append("  ".join(row))

    path.write_text("\n".join(lines))


def make_fake_link_via_array(path: Path, signal_amount: int,
                             vias_x: int, vias_y: int) -> None:
    """Write a minimal via_array.txt for a Link sim."""
    lines = [
        f"layer list: ['G', 'D/2', 'S', 'D/2', 'G']",
        "amount of layers: 3",
        "[layer-drawing]",
        "(omitted)",
    ]
    # Link: 1 port per signal in each of 2 arrays.  Total ports = 2 * signal_amount.
    # Order: arr_1 sgn1..sgnK, then arr_2 sgn1..sgnK.
    for arr in (1, 2):
        for sgn in range(1, signal_amount + 1):
            x = ((sgn - 1) // vias_y) + 1
            y = ((sgn - 1) %  vias_y) + 1
            link = (sgn + 1) // 2   # pair index
            lines.append(f"arr_{arr}_via_{x}_{y}_sgn{sgn}_link{link}_pup")

    # ARRAY grid (used for context features; pick arr_1 layout — same as arr_2)
    lines.append("[ARRAY]")
    grid = [["G1"] * vias_x for _ in range(vias_y)]
    for sgn in range(1, signal_amount + 1):
        x = ((sgn - 1) // vias_y) + 1
        y = ((sgn - 1) %  vias_y) + 1
        row_idx_top = vias_y - y
        col_idx = x - 1
        grid[row_idx_top][col_idx] = f"S{sgn}"
    for row in grid:
        lines.append("  ".join(row))

    path.write_text("\n".join(lines))


def build_synthetic_dataset(root: Path, dataset_type: str,
                            n_sims: int = 3) -> None:
    """Build a tiny on-disk dataset under <root>/."""
    if root.exists():
        shutil.rmtree(root)
    var_dir = root / "variation"
    var_dir.mkdir(parents=True)

    # parameter.csv rows
    rows = []
    for i in range(n_sims):
        sig = 2 + 2 * i          # 2, 4, 6 signals -> 1, 2, 3 pairs
        vx, vy = 3, max(sig, 3)  # ensure grid is large enough for signals
        sim_id = f"sim_pkg_{i:03d}"

        # Touchstone
        sim_dir = var_dir / sim_id
        sim_dir.mkdir()
        ts = sim_dir / f"{sim_id}.s{2*sig}p"
        make_synthetic_touchstone(num_ports=2 * sig, path=ts, seed=i + 1)

        # via_array.txt
        if dataset_type == "Array":
            make_fake_array_via_array(sim_dir / "via_array.txt",
                                      signal_amount=sig, vias_x=vx, vias_y=vy)
        else:
            make_fake_link_via_array(sim_dir / "via_array.txt",
                                     signal_amount=sig, vias_x=vx, vias_y=vy)

        # parameter.csv row
        row = {
            "SIM_ID": 100 + i,
            "SIMULATION": sim_id,
            "PERMITTIVITY": 3.5 + i * 0.1,
            "CONDUCTIVITY": 5e7,
            "LOSSTANGENT":  0.005,
            "TDIEL":        20.0 + i,
            "TMET":         1.4,
            "LAYER_AMOUNT": 8 + 4 * i,
            "VIAS_X_AMOUNT": vx,
            "VIAS_Y_AMOUNT": vy,
            "VIA_RADIUS":   5.0,
            "ANTIPAD_RADIUS": 12.0,
            "PITCH":         30.0,
            "SIGNAL_AMOUNT": sig,
            "GROUND_AMOUNT": vx * vy - sig,
            "POWER_AMOUNT":  0,
        }
        if dataset_type == "Link":
            row["LENGTH"]   = 1000.0 * (i + 1)
            row["SL_WIDTH"] = 4.5
        rows.append(row)

    pd.DataFrame(rows).to_csv(root / "parameter.csv", index=False)


# ---------------------------------------------------------------------------
def main():
    tmp = Path("/tmp/sandbox_synthetic_data")
    if tmp.exists():
        shutil.rmtree(tmp)

    print("=" * 70)
    print(" Building synthetic Array dataset and parsing")
    print("=" * 70)
    raw = tmp / "raw" / "Universal-Diff-SI-Array"
    out = tmp / "processed" / "Universal-Diff-SI-Array"
    build_synthetic_dataset(raw, "Array", n_sims=3)
    result = parse_dataset(dataset_type="Array", raw_dir=raw, out_dir=out)

    print("\n  Loading saved .pt to verify contents...")
    data = torch.load(result["save_path"], weights_only=False)
    expected_pairs = 1 + 2 + 3  # 6
    assert data["X_local"].shape  == (expected_pairs, 8),  f"X_local: {data['X_local'].shape}"
    assert data["X_global"].shape == (expected_pairs, 6),  f"X_global: {data['X_global'].shape}"
    assert data["X_context"].shape == (expected_pairs, 7), f"X_context: {data['X_context'].shape}"
    assert data["Y_real"].shape   == (expected_pairs, 401, 4, 4)
    assert data["Y_imag"].shape   == (expected_pairs, 401, 4, 4)
    assert len(data["sim_ids"])   == expected_pairs
    print(f"  Array test PASSED.  {expected_pairs} pairs, all shapes correct.")

    print("\n" + "=" * 70)
    print(" Building synthetic Link dataset and parsing")
    print("=" * 70)
    raw = tmp / "raw" / "Universal-Diff-SI-Link"
    out = tmp / "processed" / "Universal-Diff-SI-Link"
    build_synthetic_dataset(raw, "Link", n_sims=3)
    result = parse_dataset(dataset_type="Link", raw_dir=raw, out_dir=out)

    print("\n  Loading saved .pt to verify contents...")
    data = torch.load(result["save_path"], weights_only=False)
    assert data["X_local"].shape  == (6, 9),  f"X_local: {data['X_local'].shape}"  # 8 + SL_WIDTH
    assert data["X_global"].shape == (6, 7),  f"X_global: {data['X_global'].shape}" # 6 + LENGTH
    assert data["X_context"].shape == (6, 7)
    assert data["Y_real"].shape   == (6, 401, 4, 4)
    print(f"  Link test PASSED.  6 pairs, all shapes correct.")

    print("\n" + "=" * 70)
    print(" Both synthetic integration tests passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()