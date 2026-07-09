"""
parse_geometry.py
================================================================================
Stage 03 of the openEMS validation pipeline.

PURPOSE
    Turn a TUHH via-array simulation into a solver-neutral geometry
    description that stage 04 can build in OpenEMS. Two front-ends produce the
    SAME GeometryDescription object:

        from_sim_folder(sim_id)       - reads the raw dataset files (for the
                                        known-geometry baseline, stages 05/06)
        from_feature_vector(...)      - de-normalizes an inverse-model output
                                        back to physical dimensions (stage 07)

    Both converge on one GeometryDescription, so stage 04 has a single code
    path regardless of geometry source.

CONVENTIONS (locked against the user's data pipeline; do not change)
    Via xy position  : x = (col_index + 1), y = (n_rows - row_index),
                       multiplied by PITCH. y=1 is the BOTTOM grid row.
                       (matches parse_via_array.grid_to_position_dict)
    Layer z heights  : the stackup "layer list" alternates metal tokens
                       (G/S/P) with 'D/2' dielectric half-spacers. Metal
                       thickness = TMET; each 'D/2' = TDIEL/2.
                       (matches stackup.txt + parameter.csv)
    Port ordering    : diff pair k (1-indexed) uses single-ended ports
                       [4(k-1), 4(k-1)+2, 4(k-1)+1, 4(k-1)+3]
                       in Bockelman-Eisenstadt [TX+, TX-, RX+, RX-] order.
                       (matches parse_via_array.diff_pair_port_indices, Array)
    Log features     : CONDUCTIVITY, LOSSTANGENT, SL_WIDTH, LENGTH are stored
                       log10'd then z-scored; de-normalization inverts both.
                       (matches parse_diff_pairs.LOG_FEATURES)

    All physical lengths are in MICRONS (um), consistent with the raw dataset
    parameter.csv (VIA_RADIUS ~ 3-18, PITCH ~ 40-70, TDIEL ~ 6-69, TMET ~ 1.5-4).

USAGE
    cd ~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/openEMS_Sim
    python 03_parse_geometry.py --sim sim_pkg_0017          # dump one geometry
    python 03_parse_geometry.py --sim sim_pkg_0017 --json   # + JSON dump
    python 03_parse_geometry.py --selftest                  # run internal checks

This stage has NO openEMS dependency; it is pure parsing and can be tested
without the solver.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

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

# ----------------------------------------------------------------------------
# Feature contract (mirrors parse_diff_pairs.py exactly)
# ----------------------------------------------------------------------------
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

# Regex for via_array.txt port lines (mirrors parse_via_array.VIA_LINE_REGEX)
VIA_LINE_REGEX = re.compile(
    r"^arr_(?P<arr>\d+)"
    r"_via_(?P<x>\d+)_(?P<y>\d+)"
    r"_sgn(?P<sgn>\d+)"
    r"(?:_link(?P<link>\d+))?"
    r"_?(?P<half>pup|plw)?"
    r"\s*$"
)


# ============================================================================
# GeometryDescription: the solver-neutral output of this stage
# ============================================================================
@dataclass
class ViaSpec:
    """One physical via in the array."""
    name: str                 # e.g. 'S1', 'G3', 'P1'
    net: str                  # 'S', 'G', or 'P'
    grid_x: int               # 1-based grid column coordinate
    grid_y: int               # 1-based grid row coordinate (y=1 is bottom)
    x_um: float               # physical x in microns
    y_um: float               # physical y in microns
    sgn_index: Optional[int]  # signal number (1-based) if net == 'S', else None


@dataclass
class LayerSpec:
    """One physical layer in the vertical stackup."""
    index: int                # 1-based layer index from the top
    kind: str                 # 'metal' (G/S/P) or 'dielectric'
    token: str                # original token: 'G','S','P','D/2'
    z_top_um: float           # top boundary (microns), z increasing downward
    z_bot_um: float           # bottom boundary (microns)
    thickness_um: float


@dataclass
class GeometryDescription:
    """Everything stage 04 needs to build the OpenEMS model."""
    sim_id: str
    # scalar physical parameters (microns / SI)
    via_radius_um: float
    antipad_radius_um: float
    pitch_um: float
    tdiel_um: float           # full dielectric thickness (each D/2 is half)
    tmet_um: float
    permittivity: float
    conductivity_spm: float   # S/m
    loss_tangent: float
    # counts
    layer_amount: int
    vias_x: int
    vias_y: int
    signal_amount: int
    ground_amount: int
    power_amount: int
    # structure
    vias: list[ViaSpec] = field(default_factory=list)
    layers: list[LayerSpec] = field(default_factory=list)
    # port mapping: single-ended touchstone order -> (via_name, half)
    # index i (0-based) is touchstone port i+1
    se_port_map: list[dict] = field(default_factory=list)
    # bookkeeping
    total_thickness_um: float = 0.0
    n_ports: int = 0
    source: str = "sim_folder"   # or "feature_vector"

    def pair_port_indices(self, pair_k: int) -> list[int]:
        """0-based [TX+, TX-, RX+, RX-] for diff pair k (1-indexed).
        Array convention, identical to parse_via_array.diff_pair_port_indices.
        """
        base = 4 * (pair_k - 1)
        idx = [base, base + 2, base + 1, base + 3]
        if max(idx) >= self.n_ports:
            raise ValueError(
                f"pair {pair_k} out of range for {self.n_ports}-port array "
                f"(indices {idx})"
            )
        return idx

    def num_pairs(self) -> int:
        return self.signal_amount // 2

    def summary(self) -> str:
        lines = [
            f"GeometryDescription  sim_id={self.sim_id}  source={self.source}",
            f"  via_radius   = {self.via_radius_um:.3f} um",
            f"  antipad_rad  = {self.antipad_radius_um:.3f} um",
            f"  pitch        = {self.pitch_um:.3f} um",
            f"  TDIEL/TMET   = {self.tdiel_um:.3f} / {self.tmet_um:.3f} um",
            f"  eps_r        = {self.permittivity:.4f}",
            f"  sigma        = {self.conductivity_spm:.3e} S/m",
            f"  tan(delta)   = {self.loss_tangent:.5f}",
            f"  layers(metal)= {self.layer_amount}  "
            f"(stackup entries: {len(self.layers)})",
            f"  grid         = {self.vias_x} x {self.vias_y}",
            f"  S/G/P vias   = {self.signal_amount}/{self.ground_amount}/"
            f"{self.power_amount}  (total placed: {len(self.vias)})",
            f"  n_ports      = {self.n_ports}  ({self.num_pairs()} diff pairs)",
            f"  total_thick  = {self.total_thickness_um:.2f} um",
        ]
        return "\n".join(lines)


# ============================================================================
# Parsing helpers
# ============================================================================
def _load_parameter_row(sim_id: str, raw_dir: Path = RAW_ARRAY_DIR) -> dict:
    """Return the parameter.csv row for one sim as a dict (raw physical units).
    Matches parse_diff_pairs._load_parameter_csv column handling.
    """
    df = pd.read_csv(raw_dir / "parameter.csv")
    if "LOSTANGENT" in df.columns and "LOSSTANGENT" not in df.columns:
        df = df.rename(columns={"LOSTANGENT": "LOSSTANGENT"})
    # The SIMULATION column holds the sim_pkg_XXXX folder name
    match = df[df["SIMULATION"] == sim_id]
    if len(match) == 0:
        raise ValueError(f"sim_id {sim_id!r} not found in parameter.csv")
    return match.iloc[0].to_dict()


def _parse_via_array_file(path: Path) -> dict:
    """Parse via_array.txt -> {'layers', 'ports', 'grid'}.
    Mirrors parse_via_array.parse_via_array exactly.
    """
    text = Path(path).read_text()

    # Stackup layer list
    m = re.search(r"layer list: \[(.*?)\]", text)
    layers: list[str] = []
    if m:
        layers = [s.strip().strip("'\"") for s in m.group(1).split(",")]

    # Port lines (touchstone order)
    ports: list[dict] = []
    for line in text.splitlines():
        match = VIA_LINE_REGEX.match(line.strip())
        if match:
            ports.append({
                "port_idx_1based": len(ports) + 1,
                "arr": int(match.group("arr")),
                "x": int(match.group("x")),
                "y": int(match.group("y")),
                "sgn": int(match.group("sgn")),
                "link": int(match.group("link")) if match.group("link") else None,
                "half": match.group("half"),
            })

    # [ARRAY] grid
    grid: list[list[str]] = []
    in_grid = False
    for line in text.splitlines():
        if line.strip().startswith("[ARRAY]"):
            in_grid = True
            continue
        if in_grid:
            row = line.strip()
            if not row:
                break
            tokens = re.findall(r"[GSP]\d+", row)
            if tokens:
                grid.append(tokens)

    return {"layers": layers, "ports": ports, "grid": grid}


def _stackup_from_via_array(path: Path) -> list[str]:
    """Some sims store the full stackup only in stackup.txt; prefer that if
    present (longer, authoritative), else fall back to via_array's layer list.
    """
    sim_dir = path.parent
    stk = sim_dir / "stackup.txt"
    if stk.exists():
        m = re.search(r"layer list: \[(.*?)\]", stk.read_text())
        if m:
            return [s.strip().strip("'\"") for s in m.group(1).split(",")]
    # fall back
    return _parse_via_array_file(path)["layers"]


def _build_layers(layer_tokens: list[str], tmet_um: float,
                  tdiel_um: float) -> list[LayerSpec]:
    """Convert the layer-token list into physical LayerSpecs with z-heights.

    Metal tokens (G/S/P) get thickness = TMET.
    'D/2' tokens get thickness = TDIEL / 2.
    z increases downward from 0 at the top.
    """
    layers: list[LayerSpec] = []
    z = 0.0
    for i, tok in enumerate(layer_tokens):
        if tok in ("G", "S", "P"):
            th = tmet_um
            kind = "metal"
        elif tok == "D/2":
            th = tdiel_um / 2.0
            kind = "dielectric"
        else:
            raise ValueError(f"Unknown stackup token {tok!r} at index {i}")
        layers.append(LayerSpec(
            index=i + 1, kind=kind, token=tok,
            z_top_um=z, z_bot_um=z + th, thickness_um=th,
        ))
        z += th
    return layers


def _build_vias(grid: list[list[str]], ports: list[dict],
                pitch_um: float) -> list[ViaSpec]:
    """Place every via (S/G/P) from the [ARRAY] grid at its physical xy.

    Grid coordinate convention (mirrors grid_to_position_dict):
        x = col_index_0based + 1
        y = n_rows - row_index_0based     (y=1 is bottom row)
    Physical position: (x-1)*pitch, (y-1)*pitch  (origin at grid corner).

    Signal vias get their sgn_index from the port list (position match).
    """
    n_rows = len(grid)
    # Build sgn -> grid position from the port list, so signal vias get indexed.
    pos_to_sgn: dict[tuple[int, int], int] = {}
    for p in ports:
        pos_to_sgn[(p["x"], p["y"])] = p["sgn"]

    vias: list[ViaSpec] = []
    for row_idx, row in enumerate(grid):
        for col_idx, token in enumerate(row):
            net = token[0]                 # 'G','S','P'
            gx = col_idx + 1
            gy = n_rows - row_idx
            x_um = (gx - 1) * pitch_um
            y_um = (gy - 1) * pitch_um
            sgn = pos_to_sgn.get((gx, gy)) if net == "S" else None
            vias.append(ViaSpec(
                name=token, net=net, grid_x=gx, grid_y=gy,
                x_um=x_um, y_um=y_um, sgn_index=sgn,
            ))
    return vias


def _build_se_port_map(ports: list[dict]) -> list[dict]:
    """Single-ended touchstone port map: index i (0-based) -> port descriptor.
    For Array files each signal via contributes two ports (pup, plw).
    """
    return [
        {
            "se_index_0based": i,
            "touchstone_port_1based": p["port_idx_1based"],
            "sgn": p["sgn"],
            "half": p["half"],           # 'pup' (top) or 'plw' (bottom)
            "grid_x": p["x"],
            "grid_y": p["y"],
        }
        for i, p in enumerate(ports)
    ]


# ============================================================================
# Front-end 1: from a dataset simulation folder
# ============================================================================
def from_sim_folder(sim_id: str,
                    raw_dir: Path = RAW_ARRAY_DIR) -> GeometryDescription:
    """Build a GeometryDescription from the raw dataset files for one sim."""
    sim_dir = raw_dir / "variation" / sim_id
    via_array_path = sim_dir / "via_array.txt"
    if not via_array_path.exists():
        raise FileNotFoundError(f"via_array.txt not found in {sim_dir}")

    row = _load_parameter_row(sim_id, raw_dir)
    parsed = _parse_via_array_file(via_array_path)
    layer_tokens = _stackup_from_via_array(via_array_path)

    tmet = float(row["TMET"])
    tdiel = float(row["TDIEL"])
    pitch = float(row["PITCH"])

    layers = _build_layers(layer_tokens, tmet, tdiel)
    vias = _build_vias(parsed["grid"], parsed["ports"], pitch)
    se_port_map = _build_se_port_map(parsed["ports"])

    geo = GeometryDescription(
        sim_id=sim_id,
        via_radius_um=float(row["VIA_RADIUS"]),
        antipad_radius_um=float(row["ANTIPAD_RADIUS"]),
        pitch_um=pitch,
        tdiel_um=tdiel,
        tmet_um=tmet,
        permittivity=float(row["PERMITTIVITY"]),
        conductivity_spm=float(row["CONDUCTIVITY"]),
        loss_tangent=float(row["LOSSTANGENT"]),
        layer_amount=int(row["LAYER_AMOUNT"]),
        vias_x=int(row["VIAS_X_AMOUNT"]),
        vias_y=int(row["VIAS_Y_AMOUNT"]),
        signal_amount=int(row["SIGNAL_AMOUNT"]),
        ground_amount=int(row["GROUND_AMOUNT"]),
        power_amount=int(row["POWER_AMOUNT"]),
        vias=vias,
        layers=layers,
        se_port_map=se_port_map,
        total_thickness_um=sum(l.thickness_um for l in layers),
        n_ports=len(parsed["ports"]),
        source="sim_folder",
    )
    return geo


# ============================================================================
# Front-end 2: from an inverse-model feature vector
# ============================================================================
def _denormalize_features(x_norm: np.ndarray, mean: np.ndarray,
                          std: np.ndarray, feature_names: list[str]) -> dict:
    """Invert z-scoring, then invert log10 for LOG_FEATURES.
    Returns {feature_name: physical_value}.
    """
    x_raw = x_norm * std + mean            # invert z-score
    out = {}
    for i, name in enumerate(feature_names):
        val = float(x_raw[i])
        if name in LOG_FEATURES:
            val = 10.0 ** val              # invert log10
        out[name] = val
    return out


def from_feature_vector(
    x_local_norm: np.ndarray,
    x_global_norm: np.ndarray,
    x_context_norm: np.ndarray,
    norm_stats: dict,
    template_sim_id: str,
    raw_dir: Path = RAW_ARRAY_DIR,
) -> GeometryDescription:
    """Build a GeometryDescription from a (normalized) model output.

    The scalar physical parameters come from de-normalizing the feature
    vectors. The ARRAY TOPOLOGY (grid layout, which positions are S/G/P, the
    port ordering) is NOT encoded in the per-pair feature vector, so it is
    taken from a template simulation that has the same grid dimensions. This
    mirrors how the dataset was built: the model predicts per-pair scalar
    geometry, conditioned on the array context; the surrounding array layout
    is contextual, not generated.

    Parameters
    ----------
    x_*_norm : the normalized feature vectors (as the model emits them)
    norm_stats : dict with X_local_mean/std, X_global_mean/std,
                 X_context_mean/std, and the feature-name lists, as stored in
                 diff_pair_dataset.pt
    template_sim_id : a dataset sim whose grid (vias_x, vias_y, S/G/P layout)
                      matches the intended design; supplies topology + ports
    """
    local_names = norm_stats["local_features"]
    global_names = norm_stats["global_features"]

    loc = _denormalize_features(
        np.asarray(x_local_norm, dtype=np.float64),
        np.asarray(norm_stats["X_local_mean"], dtype=np.float64),
        np.asarray(norm_stats["X_local_std"], dtype=np.float64),
        local_names,
    )
    glob = _denormalize_features(
        np.asarray(x_global_norm, dtype=np.float64),
        np.asarray(norm_stats["X_global_mean"], dtype=np.float64),
        np.asarray(norm_stats["X_global_std"], dtype=np.float64),
        global_names,
    )

    # Topology from the template sim (grid + ports + stackup token pattern)
    sim_dir = raw_dir / "variation" / template_sim_id
    parsed = _parse_via_array_file(sim_dir / "via_array.txt")
    layer_tokens = _stackup_from_via_array(sim_dir / "via_array.txt")

    tmet = loc["TMET"]
    tdiel = loc["TDIEL"]
    pitch = loc["PITCH"]

    layers = _build_layers(layer_tokens, tmet, tdiel)
    vias = _build_vias(parsed["grid"], parsed["ports"], pitch)
    se_port_map = _build_se_port_map(parsed["ports"])

    geo = GeometryDescription(
        sim_id=f"generated_from_{template_sim_id}",
        via_radius_um=loc["VIA_RADIUS"],
        antipad_radius_um=loc["ANTIPAD_RADIUS"],
        pitch_um=pitch,
        tdiel_um=tdiel,
        tmet_um=tmet,
        permittivity=loc["PERMITTIVITY"],
        conductivity_spm=loc["CONDUCTIVITY"],
        loss_tangent=loc["LOSSTANGENT"],
        layer_amount=int(round(glob["LAYER_AMOUNT"])),
        vias_x=int(round(glob["VIAS_X_AMOUNT"])),
        vias_y=int(round(glob["VIAS_Y_AMOUNT"])),
        signal_amount=int(round(glob["SIGNAL_AMOUNT"])),
        ground_amount=int(round(glob["GROUND_AMOUNT"])),
        power_amount=int(round(glob["POWER_AMOUNT"])),
        vias=vias,
        layers=layers,
        se_port_map=se_port_map,
        total_thickness_um=sum(l.thickness_um for l in layers),
        n_ports=len(parsed["ports"]),
        source="feature_vector",
    )
    return geo


# ============================================================================
# CLI / self-test
# ============================================================================
def _selftest() -> None:
    """Parse sim_pkg_0017 and assert the decoded facts match what we know."""
    print("Running self-test on sim_pkg_0017 ...")
    geo = from_sim_folder("sim_pkg_0017")
    print(geo.summary())

    # Known facts about sim_pkg_0017 from the inspector output:
    #   8 signal vias -> 16 ports -> 4 diff pairs
    #   grid is 5 rows x 3 cols (G6 S7 S5 / P1 S8 S6 / G5 S1 S3 / G4 S2 S4 / G1 G2 G3)
    assert geo.n_ports == 16, f"expected 16 ports, got {geo.n_ports}"
    assert geo.signal_amount == 8, f"expected 8 signals, got {geo.signal_amount}"
    assert geo.num_pairs() == 4, f"expected 4 pairs, got {geo.num_pairs()}"
    n_signal_vias = sum(1 for v in geo.vias if v.net == "S")
    assert n_signal_vias == 8, f"expected 8 S vias placed, got {n_signal_vias}"

    # Port-index formula check (Array): pair 1 -> [0, 2, 1, 3]
    assert geo.pair_port_indices(1) == [0, 2, 1, 3], geo.pair_port_indices(1)
    # pair 2 -> [4, 6, 5, 7]
    assert geo.pair_port_indices(2) == [4, 6, 5, 7], geo.pair_port_indices(2)

    # Layer count: LAYER_AMOUNT metal layers; stackup has metal + D/2 spacers.
    n_metal = sum(1 for l in geo.layers if l.kind == "metal")
    print(f"\n  metal layers in stackup: {n_metal} "
          f"(parameter.csv LAYER_AMOUNT = {geo.layer_amount})")

    # z monotonic + total thickness consistency
    for a, b in zip(geo.layers, geo.layers[1:]):
        assert b.z_top_um >= a.z_top_um, "layer z not monotonic"
    print(f"  total stack thickness: {geo.total_thickness_um:.2f} um")

    print("\nSELF-TEST PASSED")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", type=str, default=None,
                    help="sim_pkg id to parse and dump (e.g. sim_pkg_0017)")
    ap.add_argument("--json", action="store_true",
                    help="also print the GeometryDescription as JSON")
    ap.add_argument("--selftest", action="store_true",
                    help="run internal consistency checks on sim_pkg_0017")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    if args.sim is None:
        print("Nothing to do. Pass --sim <id> or --selftest.")
        print("Example: python 03_parse_geometry.py --sim sim_pkg_0017")
        return

    geo = from_sim_folder(args.sim)
    print(geo.summary())

    if args.json:
        # asdict handles the nested dataclasses; print compactly
        d = asdict(geo)
        # Trim the long via/layer lists for readability in the dump
        print("\n--- JSON (vias/layers truncated to first 3) ---")
        d_short = dict(d)
        d_short["vias"] = d["vias"][:3] + [{"...": f"+{len(d['vias'])-3} more"}]
        d_short["layers"] = d["layers"][:3] + [{"...": f"+{len(d['layers'])-3} more"}]
        d_short["se_port_map"] = d["se_port_map"][:3] + [{"...": "..."}]
        print(json.dumps(d_short, indent=2))


if __name__ == "__main__":
    main()