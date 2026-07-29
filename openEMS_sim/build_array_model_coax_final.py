"""
04_build_array_model.py  (v4 - native CoaxialPort launches)
================================================================================
Stage 04 of the openEMS validation pipeline.

WHY v4 (port history - read this before changing anything)
    v1 ports: lumped box centred ON the via -> the PEC via ran through the port
        and (with port priority above via priority) replaced a slice of the via
        with a 50-ohm resistor. Result: every port reflected ~5 dB, through
        path 12 dB down. WRONG.
    v2/v3 ports: hand-rolled radial lumped port across the antipad annulus
        (via wall -> plane edge). Terminals only touched the curved metal at
        tangent points; on the staircased FDTD grid the inner terminal could
        float entirely. Result: near-zero incident wave -> S-parameters at
        +28..+47 dB (impossible positive gain for a passive structure). WRONG.
    INFINITE BOARD: dielectric + planes extend into the lateral PML_8 to
        reproduce CONMLS's infinite planes (adopted from the radial-port
        experiments, which showed this corrects the Sdd21 HF slope).
    v4 ports: openEMS's NATIVE CoaxialPort (openEMS/ports.py class
        CoaxialPort). A via end referenced to its ground plane through an
        antipad IS a coaxial launch: via = inner conductor, antipad rim =
        shield. The class builds the conductors itself, excites the true
        radial-TEM mode (1/r weighted E), measures V and I with the
        differential transmission-line method, and computes the line's own
        Z_ref(f) and beta(f) from the fields. This is the reference-backed
        implementation, used in the same canonical pattern as the official
        MSL tutorial: the feed line runs INTO the z-PML, which absorbs the
        backward wave.

REFERENCE-PLANE / IMPEDANCE HANDLING (why stage 05 needs no changes)
    The air coax (r_i = via radius, r_o = antipad radius) is ~110 ohm, and the
    port measures part-way up the stub. Two corrections make the saved result
    directly comparable to the CONMLS 50-ohm touchstone at the via ends:
      1. de-embed: CalcPort(..., ref_plane_shift = stub_to_surface_distance)
         applies the exact transmission-line transform (openEMS Port.CalcPort
         lines 81-90) moving the reference plane to the board surface;
      2. renormalize: the full 16-port matrix is renormalized from the coax's
         field-computed Z_ref(f) to 50 ohm with scikit-rf.
    The npz written to results/04_openems/ is therefore 50-ohm-referenced at
    the board surface -- stages 05/06/07/08 consume it unchanged.

TUHH CONVENTIONS (unchanged, verified against the dataset PDFs)
    Units MIL (1 mil = 25.4e-6 m). Vias: solid PEC cylinders, no pads.
    Ground vias -> all ground planes; power vias -> all power planes; signal
    vias clear every plane with an antipad. Full-height through-vias.
    Antipads by priority override (dielectric cylinder punches plane).
    Priority ladder: dielectric 0 < plane 10 < antipad 20 < via/coax 30.

TIMING (the time-constraint work is kept)
    v3's lean board mesh is retained (z densified only inside the board,
    15 cells/lambda lateral, refinement at via walls/antipad edges). The z air
    margin is 100 mil per side with uniform 5-mil cells because PML_8 and the
    coax launch need real cells there. Estimate ~2.3M cells, ~8 min/port,
    ~2.2 h per full 16-port sim (vs ~4.5 h originally).

USAGE
    python 04_build_array_model.py --sim sim_pkg_0017 --dry-run
    python 04_build_array_model.py --sim sim_pkg_0017 --run
    python 04_build_array_model.py --sim sim_pkg_0017 --run --mur   # fallback
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Import stage 03 by path (tolerate the user's renamed files)
# ----------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent


def _load_stage03():
    for name in ("03_parse_geometry.py", "parse_geometry.py"):
        p = _THIS_DIR / name
        if p.exists():
            spec = importlib.util.spec_from_file_location("stage03", p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["stage03"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("stage 03 not found (03_parse_geometry.py / "
                            "parse_geometry.py)")


stage03 = _load_stage03()
GeometryDescription = stage03.GeometryDescription
from_sim_folder = stage03.from_sim_folder
from_feature_vector = stage03.from_feature_vector

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
MIL_TO_M = 25.4e-6
C0 = 299792458.0

PRIO_DIELECTRIC = 0
PRIO_PLANE = 10
PRIO_ANTIPAD = 20
PRIO_VIA = 30          # coax launch conductors use this too

F_MAX_HZ = 100e9
F_MIN_HZ = 0.25e9
N_FREQ = 401
CELLS_PER_WAVELENGTH = 15
END_CRITERIA = 1e-5
MAX_TIMESTEPS = 60000

AIR_MARGIN_XY_FRAC = 1.0   # times pitch, lateral air beyond board edge
AIR_Z_MIL = 100.0          # air stub above/below the board: hosts the coax
                           # launch and the z-PML (8 cells) behind the feed
AIR_CELL_MIL = 5.0         # uniform z cell in the air stubs (20 cells/side)
SHIELD_THICK_MIL = 3.0     # coax outer-shell radial thickness
FEED_FRAC = 0.55           # excitation position along the stub (past PML_8)
MEAS_FRAC = 0.72           # measurement plane along the stub (past the feed)


# ============================================================================
class ArrayModelBuilder:
    """Full via-array OpenEMS model with native CoaxialPort launches."""

    def __init__(self, geo: GeometryDescription,
                 f_max_hz: float = F_MAX_HZ,
                 cells_per_wavelength: int = CELLS_PER_WAVELENGTH,
                 use_mur_z: bool = False,
                 verbose: bool = True):
        self.geo = geo
        self.f_max_hz = f_max_hz
        self.cells_per_wavelength = cells_per_wavelength
        self.use_mur_z = use_mur_z
        self.verbose = verbose

        xs = [v.x_mil for v in geo.vias]
        ys = [v.y_mil for v in geo.vias]
        pitch = geo.pitch_mil
        self.board_x0 = min(xs) - pitch
        self.board_x1 = max(xs) + pitch
        self.board_y0 = min(ys) - pitch
        self.board_y1 = max(ys) + pitch
        self.board_z0 = 0.0
        self.board_z1 = geo.total_thickness_mil

        self.air_xy = AIR_MARGIN_XY_FRAC * pitch
        self.dom_x0 = self.board_x0 - self.air_xy
        self.dom_x1 = self.board_x1 + self.air_xy
        self.dom_y0 = self.board_y0 - self.air_xy
        self.dom_y1 = self.board_y1 + self.air_xy
        self.dom_z0 = self.board_z0 - AIR_Z_MIL
        self.dom_z1 = self.board_z1 + AIR_Z_MIL

        self.metal_layers = [l for l in geo.layers if l.kind == "metal"]
        self.ground_layers = [l for l in geo.layers if l.token == "G"]
        self.power_layers = [l for l in geo.layers if l.token == "P"]

        lam_min_m = C0 / (self.f_max_hz * np.sqrt(geo.permittivity))
        self.resolution_mil = (lam_min_m / MIL_TO_M) / self.cells_per_wavelength

        # de-embedding bookkeeping, filled at port creation
        self._port_deembed_mil = [0.0] * geo.n_ports

        if self.verbose:
            self._print_summary()

    def _print_summary(self):
        g = self.geo
        print(f"  ArrayModelBuilder v4 (CoaxialPort) for {g.sim_id}")
        print(f"    board (mil) : x[{self.board_x0:.1f},{self.board_x1:.1f}] "
              f"y[{self.board_y0:.1f},{self.board_y1:.1f}] z[0,{self.board_z1:.1f}]")
        print(f"    domain (mil): z[{self.dom_z0:.1f},{self.dom_z1:.1f}] "
              f"(coax stubs of {AIR_Z_MIL:.0f} mil above/below)")
        zc = 60.0 / 1.0 * np.log(g.antipad_radius_mil / g.via_radius_mil)
        print(f"    coax launch : r_i={g.via_radius_mil:.3f} r_o="
              f"{g.antipad_radius_mil:.3f} (air, Z0 ~ {zc:.0f} ohm; "
              f"renormalized to 50 ohm on output)")
        print(f"    mesh res    : {self.resolution_mil:.3f} mil "
              f"({self.cells_per_wavelength} cells/lambda @ "
              f"{self.f_max_hz/1e9:.0f} GHz)")
        print(f"    boundaries  : lateral PML_8 (infinite planes), z "
              f"{'MUR (fallback)' if self.use_mur_z else 'PML_8'}")
        print(f"    n_ports     : {g.n_ports}")

    # ------------------------------------------------------------------
    def build_geometry(self, FDTD, CSX, excited_port):
        geo = self.geo

        eps0 = 8.8541878128e-12
        kappa = (2.0 * np.pi * self.f_max_hz * eps0
                 * geo.permittivity * geo.loss_tangent)
        diel = CSX.AddMaterial("dielectric", epsilon=geo.permittivity,
                               kappa=kappa)
        pec = CSX.AddMetal("pec")

        # INFINITE PLANES: dielectric + metal planes fill the ENTIRE x/y
        # domain and run into the lateral PML. This reproduces the CONMLS
        # infinite-plane assumption and absorbs parallel-plate modes at the
        # edges instead of reflecting them (the fix that corrected the Sdd21
        # high-frequency slope in the radial-port experiments).
        diel.AddBox([self.dom_x0, self.dom_y0, self.board_z0],
                    [self.dom_x1, self.dom_y1, self.board_z1],
                    priority=PRIO_DIELECTRIC)

        for layer in self.metal_layers:
            if layer.token in ("G", "P"):
                pec.AddBox([self.dom_x0, self.dom_y0, layer.z_top_mil],
                           [self.dom_x1, self.dom_y1, layer.z_bot_mil],
                           priority=PRIO_PLANE)

        # Antipads: signal clears G+P; ground clears P; power clears G
        antipad_r = geo.antipad_radius_mil
        for via in geo.vias:
            clear = (self.ground_layers + self.power_layers if via.net == "S"
                     else self.power_layers if via.net == "G"
                     else self.ground_layers if via.net == "P" else [])
            for layer in clear:
                diel.AddCylinder([via.x_mil, via.y_mil, layer.z_top_mil],
                                 [via.x_mil, via.y_mil, layer.z_bot_mil],
                                 antipad_r, priority=PRIO_ANTIPAD)

        # Vias: solid PEC, full board height
        for via in geo.vias:
            pec.AddCylinder([via.x_mil, via.y_mil, self.board_z0],
                            [via.x_mil, via.y_mil, self.board_z1],
                            geo.via_radius_mil, priority=PRIO_VIA)

        # Mesh BEFORE ports: CoaxialPort snaps its probe/feed planes to
        # existing mesh lines, so the grid must exist first.
        self._build_mesh(CSX)

        ports = self._add_ports(FDTD, CSX, pec, excited_port)
        return ports

    def _add_ports(self, FDTD, CSX, pec, excited_port):
        """Native CoaxialPort launch at each signal-via end.

        pup: stub from z = -AIR_Z (start, in/through the z-PML) down to the
             board top surface (stop). plw: mirrored below the board.
        Inner conductor r_i = via radius (continuous with the via metal);
        shield r_o = antipad radius (lands on the ground plane around the
        antipad hole), thickness SHIELD_THICK_MIL. Air-filled (mat_prop=None).
        Excitation at FEED_FRAC of the stub (outside the 8-cell PML);
        measurement plane at MEAS_FRAC; the remaining MEAS->surface length is
        de-embedded in run_and_extract via CalcPort(ref_plane_shift=...).
        """
        geo = self.geo
        r_i = geo.via_radius_mil
        r_o = geo.antipad_radius_mil
        r_os = r_o + SHIELD_THICK_MIL
        stub = AIR_Z_MIL

        ports = [None] * geo.n_ports
        sgn_to_via = {v.sgn_index: v for v in geo.vias if v.net == "S"}

        for pm in geo.se_port_map:
            se_idx = pm["se_index_0based"]
            via = sgn_to_via[pm["sgn"]]
            half = pm["half"]                       # 'pup' | 'plw'

            if half == "pup":
                start = [via.x_mil, via.y_mil, self.board_z0 - stub]
                stop = [via.x_mil, via.y_mil, self.board_z0]
            else:
                start = [via.x_mil, via.y_mil, self.board_z1 + stub]
                stop = [via.x_mil, via.y_mil, self.board_z1]

            exc = 1.0 if (excited_port is not None
                          and se_idx == excited_port) else 0.0
            port = FDTD.AddCoaxialPort(
                se_idx + 1, pec, None, start, stop, "z",
                r_i, r_o, r_os,
                excite_amp=exc,
                FeedShift=FEED_FRAC * stub,
                MeasPlaneShift=MEAS_FRAC * stub,
                priority=PRIO_VIA,
            )
            ports[se_idx] = port
            # distance from the (snapped) measurement plane to the DUT surface
            mps = getattr(port, "measplane_shift", MEAS_FRAC * stub)
            self._port_deembed_mil[se_idx] = stub - mps

        return ports

    def _build_mesh(self, CSX):
        mesh = CSX.GetGrid()
        mesh.SetDeltaUnit(MIL_TO_M)
        geo = self.geo
        res = self.resolution_mil

        # ---- z: exact lines at every layer boundary, densified only inside
        #      the board; UNIFORM cells in the air stubs (PML + coax launch).
        z_lines = set()
        for layer in geo.layers:
            z_lines.add(round(layer.z_top_mil, 6))
            z_lines.add(round(layer.z_bot_mil, 6))
        zs = sorted(z_lines)
        dense = list(zs)
        for a, b in zip(zs, zs[1:]):
            if b - a > res:
                n = int(np.ceil((b - a) / res))
                dense.extend(np.linspace(a, b, n + 1).tolist())
        n_air = int(round(AIR_Z_MIL / AIR_CELL_MIL))
        dense.extend((self.board_z0
                      - np.arange(1, n_air + 1) * AIR_CELL_MIL).tolist())
        dense.extend((self.board_z1
                      + np.arange(1, n_air + 1) * AIR_CELL_MIL).tolist())
        mesh.AddLine("z", sorted(set(round(z, 6) for z in dense)))

        # ---- x / y: board edges, via walls, antipad edges, via centres, and
        #      radial refinement of the coax annulus at SIGNAL vias (the port
        #      probes integrate E from r_i to r_o -- they need cells there).
        x_lines = {self.dom_x0, self.dom_x1, self.board_x0, self.board_x1}
        y_lines = {self.dom_y0, self.dom_y1, self.board_y0, self.board_y1}
        r_i = geo.via_radius_mil
        r_o = geo.antipad_radius_mil
        r_mid = 0.5 * (r_i + r_o)
        r_os = r_o + SHIELD_THICK_MIL
        for v in geo.vias:
            offs = [-r_o, -r_i, 0.0, r_i, r_o]
            if v.net == "S":
                offs += [-r_os, -r_mid, r_mid, r_os]
            for d in offs:
                x_lines.add(round(v.x_mil + d, 6))
                y_lines.add(round(v.y_mil + d, 6))
        mesh.AddLine("x", sorted(x_lines))
        mesh.AddLine("y", sorted(y_lines))
        mesh.SmoothMeshLines("x", res)
        mesh.SmoothMeshLines("y", res)
        # no z-smoothing: board z is exact, air z is deliberately uniform
        return mesh

    # ------------------------------------------------------------------
    def run_and_extract(self, sim_root: Path, run: bool = True):
        """Excite each port in turn; assemble the 16x16 S-matrix.

        Output normalization chain (see file header):
          native Z_ref(f) wave separation  ->  ref_plane_shift de-embed to the
          board surface  ->  scikit-rf renormalize to 50 ohm.
        Saved npz is 50-ohm referenced: stages 05-08 consume it unchanged.
        """
        from CSXCAD import ContinuousStructure
        from openEMS import openEMS

        geo = self.geo
        N = geo.n_ports
        freq = np.linspace(F_MIN_HZ, F_MAX_HZ, N_FREQ)
        S = np.zeros((N_FREQ, N, N), dtype=complex)
        Z0 = np.zeros((N_FREQ, N), dtype=float)
        sim_root.mkdir(parents=True, exist_ok=True)

        for p_exc in range(N):
            if self.verbose:
                print(f"    [excite port {p_exc + 1}/{N}]")
            sim_dir = sim_root / f"excite_{p_exc:02d}"

            FDTD = openEMS(NrTS=MAX_TIMESTEPS, EndCriteria=END_CRITERIA)
            FDTD.SetGaussExcite((F_MAX_HZ + F_MIN_HZ) / 2,
                                (F_MAX_HZ - F_MIN_HZ) / 2)
            # Board extends into the lateral PML (infinite planes), so x/y
            # must be PML_8, not MUR. z is PML_8 (the coax stubs run into it);
            # --mur switches z to MUR only as an instability fallback.
            zbc = "MUR" if self.use_mur_z else "PML_8"
            FDTD.SetBoundaryCond(["MUR", "MUR", "MUR", "MUR", zbc, zbc])

            CSX = ContinuousStructure()
            FDTD.SetCSX(CSX)
            ports = self.build_geometry(FDTD, CSX, excited_port=p_exc)

            if not run:
                xml = sim_root / "model.xml"
                CSX.Write2XML(str(xml))
                if self.verbose:
                    print(f"    [dry-run] wrote {xml}")
                    print(f"    Inspect: AppCSXCAD {xml}")
                return None, None

            FDTD.Run(str(sim_dir), cleanup=True, verbose=0)

            # Wave separation with each port's OWN field-computed Z_ref, and
            # the built-in TL transform moving the reference plane from the
            # measurement plane to the board surface (exact, uses beta+Z_ref).
            for m in range(N):
                ports[m].CalcPort(str(sim_dir), freq,
                                  ref_plane_shift=self._port_deembed_mil[m])
                if p_exc == 0:
                    zr = np.real(np.asarray(ports[m].Z_ref))
                    Z0[:, m] = np.clip(zr, 1.0, None)
            a_inc = ports[p_exc].uf_inc
            for m in range(N):
                S[:, m, p_exc] = ports[m].uf_ref / a_inc

        S50 = self._renormalize_to_50(S, Z0, freq)
        return freq, S50

    @staticmethod
    def _renormalize_to_50(S, Z0, freq):
        """Renormalize an (F,N,N) S-matrix from per-port Z0(f) to 50 ohm."""
        import skrf as rf
        ntwk = rf.Network(frequency=rf.Frequency.from_f(freq, unit="Hz"),
                          s=S, z0=Z0)
        ntwk.renormalize(50.0)
        return ntwk.s


# ============================================================================
def main():
    class _Tee:

        def __init__(self, *streams): self.streams = streams
        def write(self, data):
            for s in self.streams: s.write(data); s.flush()
        def flush(self):
            for s in self.streams: s.flush()
    _logfile = open(_THIS_DIR / f"stage05_log_{datetime.now():%Y%m%d_%H%M%S}.log", "w")
    sys.stdout = _Tee(sys.__stdout__, _logfile)
    sys.stderr = _Tee(sys.__stderr__, _logfile)
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", type=str, default="sim_pkg_0017")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--cells", type=int, default=CELLS_PER_WAVELENGTH)
    ap.add_argument("--mur", action="store_true",
                    help="fallback: MUR instead of PML_8 in z (use only if "
                         "the PML shows late-time energy growth)")
    args = ap.parse_args()

    print("=" * 78)
    print(f"Stage 04 v4 (CoaxialPort): build array model  ({args.sim})")
    print("=" * 78)

    geo = from_sim_folder(args.sim)
    print(geo.summary())
    print()
    builder = ArrayModelBuilder(geo, cells_per_wavelength=args.cells,
                                use_mur_z=args.mur)
    sim_root = _THIS_DIR / "runs" / f"04_{args.sim}"

    if args.dry_run:
        builder.run_and_extract(sim_root, run=False)
        print("\nDry run complete. In AppCSXCAD confirm:")
        print("  - coax stubs above AND below the board at every signal via")
        print("    (thin inner rod continuing the via + a shield tube of")
        print("    antipad radius landing on the outer ground planes)")
        print("  - the stubs reach the z domain edges (they run into the PML)")
    elif args.run:
        print("\nFull 16-port solve (est. ~8 min/port with the v4 mesh).\n")
        freq, S = builder.run_and_extract(sim_root, run=True)
        out = _THIS_DIR / "results" / "04_openems"
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / f"{args.sim}_final_coax_openems_se.npz", freq=freq, S=S)
        print(f"\nSaved (50-ohm referenced, de-embedded to board surface):")
        print(f"  {out / (args.sim + '_final_coax_openems_se.npz')}")
        print("Run stage 05 next.")
    else:
        print("\nPass --dry-run or --run.")


if __name__ == "__main__":
    main()