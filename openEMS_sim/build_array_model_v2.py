"""
04_build_array_model.py  (v2 - radial ports + air margin)
================================================================================
Stage 04 of the openEMS validation pipeline.

CHANGES vs v1 (after stage-05 port debugging)
    1. AIR MARGIN: the board no longer fills the whole domain. Air is added
       above/below (z) and around (x/y) the board, and the absorbing boundary is
       set back. v1 had the structure flush against the MUR boundary, producing
       "Excitation inside the Mur-ABC" warnings that corrupted every port.
    2. RADIAL COAXIAL PORTS: each signal-via end is referenced to its nearest
       ground plane by a RADIAL lumped port spanning the antipad annulus (from
       the via wall out to the plane edge). v1 used a box centred on the via, so
       the PEC via ran through the port and shorted it (every port reflected
       ~5 dB, through-path 12 dB down). The radial port launches the coaxial
       via-to-plane mode, matching the CONMLS -0.03 dB through-signature.
    3. PML_8 boundaries in z (with air margin) instead of MUR.

TUHH CONVENTIONS (from Universal-Diff-SI-Array.pdf, verified)
    Units          : MIL. 1 mil = 25.4e-6 m.
    Vias           : solid PEC cylinders, radius = VIA_RADIUS, no pads.
    Connectivity   : ground vias -> all ground planes; power vias -> all power
                     planes; signal vias clear every plane with an antipad.
    Array vias     : full-height through-vias.
    Ports          : radial, via-wall-to-plane-edge across the antipad annulus,
                     at the top and bottom ground planes. 50 ohm.

PRIORITY LADDER (higher wins in CSXCAD)
    dielectric 0 < plane 10 < antipad 20 < via 30 < port 40

USAGE
    python 04_build_array_model.py --sim sim_pkg_0017 --dry-run
    python 04_build_array_model.py --sim sim_pkg_0017 --run

DEPENDS ON
    03_parse_geometry.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Import stage 03 by path (filename starts with a digit)
# ----------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent


def _load_stage03():
    """Load stage 03 regardless of its exact filename on disk."""
    for name in ("03_parse_geometry.py", "parse_geometry.py"):
        p = _THIS_DIR / name
        if p.exists():
            spec = importlib.util.spec_from_file_location("stage03", p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["stage03"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        "Could not find stage 03 (looked for 03_parse_geometry.py / "
        "parse_geometry.py in this directory)."
    )


stage03 = _load_stage03()
GeometryDescription = stage03.GeometryDescription
from_sim_folder = stage03.from_sim_folder

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
MIL_TO_M = 25.4e-6
C0 = 299792458.0

PRIO_DIELECTRIC = 0
PRIO_PLANE = 10
PRIO_ANTIPAD = 20
PRIO_VIA = 30
PRIO_PORT = 40

F_MAX_HZ = 100e9
F_MIN_HZ = 0.25e9
N_FREQ = 401
CELLS_PER_WAVELENGTH = 20
END_CRITERIA = 1e-4
MAX_TIMESTEPS = 60000

# Air margin (in mil) added around the board before the absorbing boundary.
# Chosen as a fraction of the free-space quarter-wavelength at f_min so the
# boundary never sees near fields. At 0.25 GHz lambda0 ~ 1.2 m -> we don't need
# that much; a few board-thicknesses of air plus a lambda/20-at-fmax cushion is
# plenty for the port not to touch the ABC. We use a fixed generous margin.
AIR_MARGIN_XY_FRAC = 1.0     # times pitch, lateral air beyond outer vias
AIR_MARGIN_Z_FRAC = 0.5      # times total board thickness, air above/below


# ============================================================================
class ArrayModelBuilder:
    def __init__(self, geo: GeometryDescription,
                 f_max_hz: float = F_MAX_HZ,
                 cells_per_wavelength: int = CELLS_PER_WAVELENGTH,
                 verbose: bool = True):
        self.geo = geo
        self.f_max_hz = f_max_hz
        self.cells_per_wavelength = cells_per_wavelength
        self.verbose = verbose

        xs = [v.x_mil for v in geo.vias]
        ys = [v.y_mil for v in geo.vias]
        self.x_min, self.x_max = min(xs), max(xs)
        self.y_min, self.y_max = min(ys), max(ys)

        # Board footprint: one pitch of dielectric beyond the outer vias
        pitch = geo.pitch_mil
        self.board_x0 = self.x_min - pitch
        self.board_x1 = self.x_max + pitch
        self.board_y0 = self.y_min - pitch
        self.board_y1 = self.y_max + pitch

        # Board z extent
        self.board_z0 = 0.0
        self.board_z1 = geo.total_thickness_mil

        # Air margins and domain (boundary) extent
        self.air_xy = AIR_MARGIN_XY_FRAC * pitch
        self.air_z = AIR_MARGIN_Z_FRAC * geo.total_thickness_mil
        self.dom_x0 = self.board_x0 - self.air_xy
        self.dom_x1 = self.board_x1 + self.air_xy
        self.dom_y0 = self.board_y0 - self.air_xy
        self.dom_y1 = self.board_y1 + self.air_xy
        self.dom_z0 = self.board_z0 - self.air_z
        self.dom_z1 = self.board_z1 + self.air_z

        self.metal_layers = [l for l in geo.layers if l.kind == "metal"]
        self.ground_layers = [l for l in geo.layers if l.token == "G"]
        self.power_layers = [l for l in geo.layers if l.token == "P"]
        self.signal_layers = [l for l in geo.layers if l.token == "S"]

        eps_r = geo.permittivity
        lam_min_m = C0 / (self.f_max_hz * np.sqrt(eps_r))
        self.resolution_mil = (lam_min_m / MIL_TO_M) / self.cells_per_wavelength

        if self.verbose:
            self._print_summary()

    def _print_summary(self):
        g = self.geo
        print(f"  ArrayModelBuilder v2 for {g.sim_id}")
        print(f"    board (mil) : x[{self.board_x0:.1f},{self.board_x1:.1f}] "
              f"y[{self.board_y0:.1f},{self.board_y1:.1f}] "
              f"z[0,{self.board_z1:.1f}]")
        print(f"    domain (mil): x[{self.dom_x0:.1f},{self.dom_x1:.1f}] "
              f"y[{self.dom_y0:.1f},{self.dom_y1:.1f}] "
              f"z[{self.dom_z0:.1f},{self.dom_z1:.1f}]  (air margins added)")
        print(f"    metal layers: {len(self.metal_layers)} "
              f"(G={len(self.ground_layers)} S={len(self.signal_layers)} "
              f"P={len(self.power_layers)})")
        print(f"    via radius  : {g.via_radius_mil:.3f} mil   "
              f"antipad: {g.antipad_radius_mil:.3f} mil")
        print(f"    mesh res    : {self.resolution_mil:.3f} mil "
              f"({self.cells_per_wavelength} cells/lambda @ "
              f"{self.f_max_hz/1e9:.0f} GHz)")
        print(f"    n_ports     : {g.n_ports}  (radial coaxial ports)")

    # ------------------------------------------------------------------
    def build_geometry(self, FDTD, CSX, excited_port):
        geo = self.geo

        eps0 = 8.8541878128e-12
        eps_r = geo.permittivity
        tan_d = geo.loss_tangent
        kappa = 2.0 * np.pi * self.f_max_hz * eps0 * eps_r * tan_d
        diel = CSX.AddMaterial("dielectric", epsilon=eps_r, kappa=kappa)
        pec = CSX.AddMetal("pec")

        # Dielectric block = the BOARD only (air fills the rest of the domain)
        diel.AddBox(
            [self.board_x0, self.board_y0, self.board_z0],
            [self.board_x1, self.board_y1, self.board_z1],
            priority=PRIO_DIELECTRIC,
        )

        # Metal planes (G and P) across the board footprint
        for layer in self.metal_layers:
            if layer.token in ("G", "P"):
                pec.AddBox(
                    [self.board_x0, self.board_y0, layer.z_top_mil],
                    [self.board_x1, self.board_y1, layer.z_bot_mil],
                    priority=PRIO_PLANE,
                )

        # Antipad clearances (dielectric cylinders punch holes in planes)
        antipad_r = geo.antipad_radius_mil
        for via in geo.vias:
            if via.net == "S":
                clear = self.ground_layers + self.power_layers
            elif via.net == "G":
                clear = self.power_layers
            elif via.net == "P":
                clear = self.ground_layers
            else:
                clear = []
            for layer in clear:
                diel.AddCylinder(
                    [via.x_mil, via.y_mil, layer.z_top_mil],
                    [via.x_mil, via.y_mil, layer.z_bot_mil],
                    antipad_r, priority=PRIO_ANTIPAD,
                )

        # Vias (solid PEC cylinders, full board height)
        via_r = geo.via_radius_mil
        for via in geo.vias:
            pec.AddCylinder(
                [via.x_mil, via.y_mil, self.board_z0],
                [via.x_mil, via.y_mil, self.board_z1],
                via_r, priority=PRIO_VIA,
            )

        ports = self._add_ports(FDTD, CSX, excited_port)
        self._build_mesh(CSX)
        return ports

    def _add_ports(self, FDTD, CSX, excited_port):
        """Radial coaxial ports: for each signal via, one at the top ground
        plane and one at the bottom ground plane. The port spans radially (+x)
        from the via wall to the plane edge across the antipad annulus, exc_dir
        = x, R = 50 ohm. caps=True (openEMS default) bridges via wall and plane
        edge, launching the coaxial via-to-plane mode.

        Numbering matches geo.se_port_map (se_index 0..N-1); openEMS is 1-based.
        """
        geo = self.geo
        R = 50.0
        via_r = geo.via_radius_mil
        antipad_r = geo.antipad_radius_mil

        # Top ground plane = first G layer; bottom ground plane = last G layer
        top_g = self.ground_layers[0]
        bot_g = self.ground_layers[-1]
        # thin lateral extent (y) of the radial port strip
        thin = via_r

        ports = [None] * geo.n_ports
        sgn_to_via = {v.sgn_index: v for v in geo.vias if v.net == "S"}

        for pm in geo.se_port_map:
            se_idx = pm["se_index_0based"]
            sgn = pm["sgn"]
            half = pm["half"]                 # 'pup' (top) or 'plw' (bottom)
            via = sgn_to_via[sgn]

            if half == "pup":
                zlo, zhi = top_g.z_top_mil, top_g.z_bot_mil
            else:
                zlo, zhi = bot_g.z_top_mil, bot_g.z_bot_mil

            # Radial port on +x side of the via: from via wall to plane edge.
            start = [via.x_mil + via_r, via.y_mil - thin, 0.5 * (zlo + zhi)]
            stop = [via.x_mil + antipad_r, via.y_mil + thin, 0.5 * (zlo + zhi)]
            # exc_dir = x; start/stop differ in x (radial), equal in z.
            # (A lumped port needs start != stop only along exc_dir.)
            # Give the port a small z-thickness by spanning the plane slab:
            start[2] = zlo
            stop[2] = zhi
            # But exc_dir must have start != stop and others define the box.
            # We keep exc along x: x differs (via_r -> antipad_r), y and z span.

            exc = 1.0 if (excited_port is not None and se_idx == excited_port) else 0.0
            port = FDTD.AddLumpedPort(
                se_idx + 1, R, start, stop, "x",
                excite=exc, priority=PRIO_PORT,
            )
            ports[se_idx] = port

        return ports

    def _build_mesh(self, CSX):
        mesh = CSX.GetGrid()
        mesh.SetDeltaUnit(MIL_TO_M)
        geo = self.geo
        res = self.resolution_mil

        # z lines: layer boundaries within the board + air region + densify
        z_lines = {self.dom_z0, self.dom_z1}
        for layer in geo.layers:
            z_lines.add(round(layer.z_top_mil, 6))
            z_lines.add(round(layer.z_bot_mil, 6))
        z_sorted = sorted(z_lines)
        dense_z = list(z_sorted)
        for a, b in zip(z_sorted, z_sorted[1:]):
            if b - a > res:
                n = int(np.ceil((b - a) / res))
                dense_z.extend(np.linspace(a, b, n + 1).tolist())
        mesh.AddLine("z", sorted(set(round(z, 6) for z in dense_z)))

        # x / y lines: domain edges, via walls, antipad edges, port spans
        x_lines = {self.dom_x0, self.dom_x1, self.board_x0, self.board_x1}
        y_lines = {self.dom_y0, self.dom_y1, self.board_y0, self.board_y1}
        for v in geo.vias:
            for dx in (-geo.via_radius_mil, geo.via_radius_mil,
                       -geo.antipad_radius_mil, geo.antipad_radius_mil):
                x_lines.add(round(v.x_mil + dx, 6))
            for dy in (-geo.via_radius_mil, geo.via_radius_mil,
                       -geo.antipad_radius_mil, geo.antipad_radius_mil):
                y_lines.add(round(v.y_mil + dy, 6))
            x_lines.add(round(v.x_mil, 6))
            y_lines.add(round(v.y_mil, 6))

        mesh.AddLine("x", sorted(x_lines))
        mesh.AddLine("y", sorted(y_lines))
        mesh.SmoothMeshLines("x", res)
        mesh.SmoothMeshLines("y", res)
        mesh.SmoothMeshLines("z", res)
        return mesh

    # ------------------------------------------------------------------
    def run_and_extract(self, sim_root: Path, run: bool = True):
        from CSXCAD import ContinuousStructure
        from openEMS import openEMS

        geo = self.geo
        N = geo.n_ports
        freq = np.linspace(F_MIN_HZ, F_MAX_HZ, N_FREQ)
        S = np.zeros((N_FREQ, N, N), dtype=complex)
        sim_root.mkdir(parents=True, exist_ok=True)

        for p_exc in range(N):
            if self.verbose:
                print(f"    [excite port {p_exc+1}/{N}]")
            sim_dir = sim_root / f"excite_{p_exc:02d}"

            FDTD = openEMS(NrTS=MAX_TIMESTEPS, EndCriteria=END_CRITERIA)
            fc = (F_MAX_HZ - F_MIN_HZ) / 2.0
            f0 = (F_MAX_HZ + F_MIN_HZ) / 2.0
            FDTD.SetGaussExcite(f0, fc)
            # PML in z (propagation/return direction), MUR on the lateral air
            # walls. With the air margins added, neither touches the structure.
            FDTD.SetBoundaryCond(["MUR", "MUR", "MUR", "MUR", "PML_8", "PML_8"])

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
            for p_meas in range(N):
                ports[p_meas].CalcPort(str(sim_dir), freq, ref_impedance=50)
            a_inc = ports[p_exc].uf_inc
            for p_meas in range(N):
                S[:, p_meas, p_exc] = ports[p_meas].uf_ref / a_inc

        return freq, S


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", type=str, default="sim_pkg_0017")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--cells", type=int, default=CELLS_PER_WAVELENGTH)
    args = ap.parse_args()

    print("=" * 78)
    print(f"Stage 04 v2: build array model  ({args.sim})")
    print("=" * 78)

    geo = from_sim_folder(args.sim)
    print(geo.summary())
    print()
    builder = ArrayModelBuilder(geo, cells_per_wavelength=args.cells)
    sim_root = _THIS_DIR / "runs" / f"04_{args.sim}"

    if args.dry_run:
        builder.run_and_extract(sim_root, run=False)
        print("\nDry run complete. In AppCSXCAD, confirm:")
        print("  - air gap visible above/below the board (board not flush to edge)")
        print("  - ports sit in the antipad annulus (via wall -> plane edge),")
        print("    NOT as boxes centred on the vias")
    elif args.run:
        print("\nStarting full N-port solve (slow).\n")
        freq, S = builder.run_and_extract(sim_root, run=True)
        out = _THIS_DIR / "results" / "04_openems"
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / f"{args.sim}_openems_se.npz", freq=freq, S=S)
        print(f"\nSaved: {out / (args.sim + '_openems_se.npz')}")
        print("Proceed to stage 05.")
    else:
        print("\nPass --dry-run or --run.")


if __name__ == "__main__":
    main()