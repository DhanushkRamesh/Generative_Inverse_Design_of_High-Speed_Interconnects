"""
04_build_array_model.py
================================================================================
Stage 04 of the openEMS validation pipeline.

PURPOSE
    Turn a GeometryDescription (from stage 03) into a full-wave OpenEMS model
    of the complete via array, following the TUHH dataset conventions exactly,
    and run a 2N-port S-parameter simulation whose result stage 05 compares
    against the CONMLS ground truth.

TUHH CONVENTIONS (from Universal-Diff-SI-Array.pdf, verified)
    Units          : MIL. 1 mil = 25.4 um = 25.4e-6 m. (Table 4 "Value (mil)")
    Vias           : solid PEC cylinders, radius = VIA_RADIUS.
                     inner_radius = 0, pad_radius_signal = 0, pad_radius_plane = 0
                     -> NO pads, NO hollow vias. Just solid cylinders.
    Connectivity   : "All ground vias connected to all ground planes. All power
                     vias connected to all power planes." (sec 2.3.1)
                     Signal vias pass through every plane with an ANTIPAD hole.
    Array vias     : full-height through-vias (back-drilled stubs are Link-only).
    Planes         : G and P layers are solid metal sheets spanning the array
                     footprint; signal (and non-matching-net) vias clear them
                     with a hole of radius ANTIPAD_RADIUS.
    Ports          : via-end to nearest ground plane, 50 ohm (confirmed by the
                     stage 03b port-reference diagnosis: Sdd21(low) ~ 0 dB).

ANTIPAD IMPLEMENTATION (CSXCAD has no boolean subtraction)
    A clearance hole is made by placing a DIELECTRIC cylinder of radius
    ANTIPAD_RADIUS at the via location, with HIGHER priority than the metal
    plane. Higher-priority primitives override lower-priority ones in the
    overlap region, so the dielectric "punches" a hole in the plane. The via
    cylinder itself has even higher priority, so the sequence at a signal-via
    location through a ground plane is:
        plane metal (priority P_PLANE)
        < antipad dielectric (priority P_ANTIPAD > P_PLANE)
        < via metal          (priority P_VIA     > P_ANTIPAD)
    leaving a metal via surrounded by a dielectric annulus inside a metal plane.

PRIORITY LADDER (higher wins)
    dielectric block      : 0
    metal planes (G/P)    : 10
    antipad dielectric    : 20
    via metal             : 30
    ports                 : 40

USAGE
    # dry run: build the model, write the CSXCAD XML for inspection, no solve
    python 04_build_array_model.py --sim sim_pkg_0017 --dry-run

    # full run: 16-port sim (excite each port in turn), extract touchstone
    python 04_build_array_model.py --sim sim_pkg_0017 --run

    # then view the geometry in AppCSXCAD (from --dry-run output):
    #   AppCSXCAD runs/04_sim_pkg_0017/model.xml

DEPENDS ON
    03_parse_geometry.py (imported for GeometryDescription + from_sim_folder)
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Import stage 03 (filename starts with a digit, so load by path)
# ----------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "stage03", _THIS_DIR / "parse_geometry.py"
)
stage03 = importlib.util.module_from_spec(_spec)
sys.modules["stage03"] = stage03
_spec.loader.exec_module(stage03)

GeometryDescription = stage03.GeometryDescription
from_sim_folder = stage03.from_sim_folder

# ----------------------------------------------------------------------------
# Units and constants
# ----------------------------------------------------------------------------
MIL_TO_M = 25.4e-6                 # 1 mil in metres
C0 = 299792458.0                   # speed of light, m/s

# Priority ladder (higher overrides lower in CSXCAD)
PRIO_DIELECTRIC = 0
PRIO_PLANE = 10
PRIO_ANTIPAD = 20
PRIO_VIA = 30
PRIO_PORT = 40

# Simulation defaults
F_MAX_HZ = 100e9
F_MIN_HZ = 0.25e9
N_FREQ = 401
CELLS_PER_WAVELENGTH = 20          # mesh resolution at f_max in dielectric
END_CRITERIA = 1e-4                # FDTD energy decay stop (-40 dB)
MAX_TIMESTEPS = 60000              # hard cap so a run can never hang forever


# ============================================================================
# Model builder
# ============================================================================
class ArrayModelBuilder:
    """Builds a CSXCAD model of the full via array from a GeometryDescription.

    Because OpenEMS lumped-port excitation is fixed at construction time,
    run_and_extract() rebuilds the model per excited port; build_geometry() is
    the shared core it calls each time.
    """

    def __init__(self, geo: GeometryDescription,
                 f_max_hz: float = F_MAX_HZ,
                 cells_per_wavelength: int = CELLS_PER_WAVELENGTH,
                 margin_mil: float | None = None,
                 verbose: bool = True):
        self.geo = geo
        self.f_max_hz = f_max_hz
        self.cells_per_wavelength = cells_per_wavelength
        self.verbose = verbose

        xs = [v.x_mil for v in geo.vias]
        ys = [v.y_mil for v in geo.vias]
        self.x_min, self.x_max = min(xs), max(xs)
        self.y_min, self.y_max = min(ys), max(ys)

        self.margin_mil = margin_mil if margin_mil is not None else geo.pitch_mil
        self.foot_x0 = self.x_min - self.margin_mil
        self.foot_x1 = self.x_max + self.margin_mil
        self.foot_y0 = self.y_min - self.margin_mil
        self.foot_y1 = self.y_max + self.margin_mil

        self.z_top = 0.0
        self.z_bot = geo.total_thickness_mil

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
        print(f"  ArrayModelBuilder for {g.sim_id}")
        print(f"    footprint (mil): x[{self.foot_x0:.1f},{self.foot_x1:.1f}] "
              f"y[{self.foot_y0:.1f},{self.foot_y1:.1f}]")
        print(f"    stack height    : {self.z_bot:.2f} mil "
              f"({self.z_bot*MIL_TO_M*1e6:.1f} um)")
        print(f"    metal layers    : {len(self.metal_layers)} "
              f"(G={len(self.ground_layers)} S={len(self.signal_layers)} "
              f"P={len(self.power_layers)})")
        print(f"    via radius      : {g.via_radius_mil:.3f} mil")
        print(f"    antipad radius  : {g.antipad_radius_mil:.3f} mil")
        print(f"    mesh resolution : {self.resolution_mil:.3f} mil "
              f"({self.cells_per_wavelength} cells/lambda at "
              f"{self.f_max_hz/1e9:.0f} GHz)")
        print(f"    n_ports         : {g.n_ports}")

    # ------------------------------------------------------------------
    # Geometry construction
    # ------------------------------------------------------------------
    def build_geometry(self, FDTD, CSX, excited_port):
        """Populate CSX with the full array. If excited_port is not None, that
        single-ended port index (0-based) is excited; all ports are always
        created (unexcited ones act as 50 ohm loads).

        Returns the list of port objects, indexed by single-ended port number.
        """
        geo = self.geo

        # ---- Materials -------------------------------------------------
        eps0 = 8.8541878128e-12
        eps_r = geo.permittivity
        tan_d = geo.loss_tangent
        kappa = 2.0 * np.pi * self.f_max_hz * eps0 * eps_r * tan_d
        diel = CSX.AddMaterial("dielectric", epsilon=eps_r, kappa=kappa)
        pec = CSX.AddMetal("pec")

        # ---- Dielectric block (fills the whole volume) -----------------
        diel.AddBox(
            [self.foot_x0, self.foot_y0, self.z_top],
            [self.foot_x1, self.foot_y1, self.z_bot],
            priority=PRIO_DIELECTRIC,
        )

        # ---- Metal planes (G and P) ------------------------------------
        for layer in self.metal_layers:
            if layer.token in ("G", "P"):
                pec.AddBox(
                    [self.foot_x0, self.foot_y0, layer.z_top_mil],
                    [self.foot_x1, self.foot_y1, layer.z_bot_mil],
                    priority=PRIO_PLANE,
                )
            # Signal layers (S) carry no plane (pad_radius_signal = 0).

        # ---- Antipads (dielectric cylinders punching holes in planes) --
        #   signal via : clears ALL planes (G and P)
        #   ground via : connects G, clears P  -> antipad in P planes
        #   power via  : connects P, clears G  -> antipad in G planes
        antipad_r = geo.antipad_radius_mil
        for via in geo.vias:
            if via.net == "S":
                planes_to_clear = self.ground_layers + self.power_layers
            elif via.net == "G":
                planes_to_clear = self.power_layers
            elif via.net == "P":
                planes_to_clear = self.ground_layers
            else:
                planes_to_clear = []
            for layer in planes_to_clear:
                diel.AddCylinder(
                    [via.x_mil, via.y_mil, layer.z_top_mil],
                    [via.x_mil, via.y_mil, layer.z_bot_mil],
                    antipad_r,
                    priority=PRIO_ANTIPAD,
                )

        # ---- Vias (solid PEC cylinders, full height) -------------------
        via_r = geo.via_radius_mil
        for via in geo.vias:
            pec.AddCylinder(
                [via.x_mil, via.y_mil, self.z_top],
                [via.x_mil, via.y_mil, self.z_bot],
                via_r,
                priority=PRIO_VIA,
            )

        # ---- Ports -----------------------------------------------------
        ports = self._add_ports(FDTD, CSX, pec, excited_port)

        # ---- Mesh ------------------------------------------------------
        self._build_mesh(CSX)

        return ports

    def _add_ports(self, FDTD, CSX, pec, excited_port):
        """Add 2N lumped ports: for each signal via, one at the top end and one
        at the bottom end, each referenced to the nearest ground plane through
        the first/last dielectric spacer. Excitation along z, R = 50 ohm.

        Port numbering matches geo.se_port_map (se_index 0..N-1); openEMS port
        numbers are 1-based.
        """
        geo = self.geo
        R = 50.0

        diel_layers = [l for l in geo.layers if l.kind == "dielectric"]
        first_diel = diel_layers[0]           # just below top ground plane
        last_diel = diel_layers[-1]           # just above bottom ground plane

        ports = [None] * geo.n_ports
        sgn_to_via = {v.sgn_index: v for v in geo.vias if v.net == "S"}

        for pm in geo.se_port_map:
            se_idx = pm["se_index_0based"]
            sgn = pm["sgn"]
            half = pm["half"]                 # 'pup' (top) or 'plw' (bottom)
            via = sgn_to_via[sgn]

            if half == "pup":
                z0, z1 = first_diel.z_top_mil, first_diel.z_bot_mil
            else:
                z0, z1 = last_diel.z_top_mil, last_diel.z_bot_mil

            r = geo.via_radius_mil
            start = [via.x_mil - r, via.y_mil - r, z0]
            stop = [via.x_mil + r, via.y_mil + r, z1]

            exc = 1.0 if (excited_port is not None and se_idx == excited_port) else 0.0
            port = FDTD.AddLumpedPort(
                se_idx + 1, R, start, stop, "z",
                excite=exc, priority=PRIO_PORT,
            )
            ports[se_idx] = port

        return ports

    def _build_mesh(self, CSX):
        """Rectilinear mesh: base resolution everywhere, refined at via walls,
        antipad edges, and every metal-layer z boundary.
        """
        mesh = CSX.GetGrid()
        mesh.SetDeltaUnit(MIL_TO_M)

        geo = self.geo
        res = self.resolution_mil

        # z lines: every layer boundary, densified in thick dielectric gaps
        z_lines = set()
        for layer in geo.layers:
            z_lines.add(round(layer.z_top_mil, 6))
            z_lines.add(round(layer.z_bot_mil, 6))
        z_sorted = sorted(z_lines)
        dense_z = list(z_sorted)
        for a, b in zip(z_sorted, z_sorted[1:]):
            gap = b - a
            if gap > res:
                n = int(np.ceil(gap / res))
                dense_z.extend(np.linspace(a, b, n + 1).tolist())
        mesh.AddLine("z", sorted(set(round(z, 6) for z in dense_z)))

        # x / y lines: via walls, antipad edges, centres, footprint
        x_lines = {self.foot_x0, self.foot_x1}
        y_lines = {self.foot_y0, self.foot_y1}
        for v in geo.vias:
            for dx in (-geo.via_radius_mil, geo.via_radius_mil):
                x_lines.add(round(v.x_mil + dx, 6))
            for dy in (-geo.via_radius_mil, geo.via_radius_mil):
                y_lines.add(round(v.y_mil + dy, 6))
            for dx in (-geo.antipad_radius_mil, geo.antipad_radius_mil):
                x_lines.add(round(v.x_mil + dx, 6))
            for dy in (-geo.antipad_radius_mil, geo.antipad_radius_mil):
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
    # Full S-parameter run
    # ------------------------------------------------------------------
    def run_and_extract(self, sim_root: Path, run: bool = True):
        """Run the N-port S-parameter simulation by exciting each port in turn.

        Returns (freq_hz, S) with S shape (F, N, N) complex, single-ended.
        """
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
            FDTD.SetBoundaryCond(["MUR"] * 6)

            CSX = ContinuousStructure()
            FDTD.SetCSX(CSX)

            ports = self.build_geometry(FDTD, CSX, excited_port=p_exc)

            if not run:
                xml = sim_root / "model.xml"
                CSX.Write2XML(str(xml))
                if self.verbose:
                    print(f"    [dry-run] wrote {xml}")
                    print(f"    Inspect with: AppCSXCAD {xml}")
                return None, None

            FDTD.Run(str(sim_dir), cleanup=True, verbose=0)

            for p_meas in range(N):
                ports[p_meas].CalcPort(str(sim_dir), freq, ref_impedance=50)
            a_inc = ports[p_exc].uf_inc
            for p_meas in range(N):
                b_ref = ports[p_meas].uf_ref
                S[:, p_meas, p_exc] = b_ref / a_inc

        return freq, S


# ============================================================================
# CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", type=str, default="sim_pkg_0017")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the model + write CSXCAD XML, no solve")
    ap.add_argument("--run", action="store_true",
                    help="run the full N-port FDTD (slow)")
    ap.add_argument("--cells", type=int, default=CELLS_PER_WAVELENGTH,
                    help="mesh cells per wavelength at f_max")
    args = ap.parse_args()

    print("=" * 78)
    print(f"Stage 04: build array model  ({args.sim})")
    print("=" * 78)

    geo = from_sim_folder(args.sim)
    print(geo.summary())
    print()

    builder = ArrayModelBuilder(geo, cells_per_wavelength=args.cells)
    sim_root = _THIS_DIR / "runs" / f"04_{args.sim}"

    if args.dry_run:
        builder.run_and_extract(sim_root, run=False)
        print("\nDry run complete. Open the XML in AppCSXCAD to inspect the "
              "geometry\nbefore committing to a full solve.")
    elif args.run:
        print("\nStarting full N-port solve. This is slow (N excitations x "
              "FDTD).\n")
        freq, S = builder.run_and_extract(sim_root, run=True)
        out = _THIS_DIR / "results" / "04_openems"
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / f"{args.sim}_openems_se.npz", freq=freq, S=S)
        print(f"\nSaved single-ended S-matrix: "
              f"{out / (args.sim + '_openems_se.npz')}")
        print("Proceed to stage 05 for the three-way comparison.")
    else:
        print("\nNothing to do. Pass --dry-run (build+inspect) or --run (solve).")


if __name__ == "__main__":
    main()