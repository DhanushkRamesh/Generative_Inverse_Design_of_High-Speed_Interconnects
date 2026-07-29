"""
04_build_array_model_v5.py
================================================================================
Stage 04 of the openEMS validation pipeline.

CONTEXT / PORT HISTORY (read before editing)
    openEMS v0.0.36 (the installed version) has ONLY AddLumpedPort,
    AddWaveGuidePort, AddRectWaveGuidePort, AddMSLPort. There is NO
    AddCoaxialPort / CurvePort in this build, so the native-coax approach
    cannot run here. This version therefore uses the stable radial LUMPED port
    that already worked (v3), and fixes v3's two real weaknesses:

    v3 weakness 1 - assumed R = 50 ohm at the port. The radial gap between a
        signal via (r_i) and the ground-plane antipad edge (r_o) is a short
        coaxial line whose characteristic impedance is NOT 50 ohm. Assuming 50
        mis-references every port.
    v3 weakness 2 - a single DC scalar (cal = 1/|S21_dc|) was multiplied over
        the whole S-matrix to force S21(DC)=0 dB. That partly fixes the through
        path (S21) but CANNOT fix reflection (S11), because S11 depends on the
        impedance mismatch itself, not a scale. Hence v3 gave S21=2.9 dB (good)
        but S11=8.5 dB (off), and "we multiplied by a fudge factor" is not
        thesis-defensible.

    v5 fix - reference each lumped port to the ACTUAL coaxial impedance of the
        via-in-antipad, then renormalize the whole matrix to 50 ohm with
        scikit-rf (frequency-correct, corrects S11 AND S21 by real physics, no
        scalar):
            Z_coax = (60 / sqrt(eps_r)) * ln(antipad_radius / via_radius)
        For sim_0017: ~64 ohm. AddLumpedPort(R=Z_coax); then renormalize
        Z_coax -> 50 with rf.Network.renormalize. The saved npz is 50-ohm
        referenced, so stages 05-08 consume it unchanged, with NO calibration
        step anywhere.

    HONEST LIMITATION (state this in the thesis): a lumped port is a single R,
    not a full transmission line, so it cannot capture the frequency dispersion
    of the launch perfectly. Using the computed coaxial Z + renormalization is
    strictly more defensible than a DC scalar, and is the best available with
    this openEMS build. If a cleaner match is required, upgrading openEMS to a
    version with CoaxialPort is the path (documented in the handoff report).

INFINITE BOARD (kept from the working v3 experiment)
    Dielectric + all G/P planes extend to the full x/y domain and run into the
    lateral PML_8, reproducing CONMLS's infinite-plane assumption and absorbing
    parallel-plate modes at the edges (this corrected the Sdd21 HF slope).
    Boundaries: PML_8 on all six faces.

TUHH conventions unchanged (MIL units, solid PEC vias, antipad priority
override, ground/power connectivity). Priority ladder: dielectric 0 < plane 10
< antipad 20 < via 30 < port 40.

USAGE
    python 04_build_array_model_v5.py --sim sim_pkg_0017 --dry-run
    python 04_build_array_model_v5.py --sim sim_pkg_0017 --run
    # output npz is 50-ohm referenced; run your existing stage 05 unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
-
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
    raise FileNotFoundError("stage 03 not found")


stage03 = _load_stage03()
GeometryDescription = stage03.GeometryDescription
from_sim_folder = stage03.from_sim_folder

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
CELLS_PER_WAVELENGTH = 15
END_CRITERIA = 1e-4
MAX_TIMESTEPS = 60000

AIR_MARGIN_Z_MIL = 60.0


def coaxial_impedance(eps_r: float, r_outer: float, r_inner: float) -> float:
    """Characteristic impedance of the via-in-antipad coaxial gap (ohm)."""
    return (60.0 / np.sqrt(eps_r)) * np.log(r_outer / r_inner)


# ============================================================================
class ArrayModelBuilder:
    def __init__(self, geo: GeometryDescription,
                 cells_per_wavelength: int = CELLS_PER_WAVELENGTH,
                 verbose: bool = True):
        self.geo = geo
        self.f_max_hz = F_MAX_HZ
        self.cells_per_wavelength = cells_per_wavelength
        self.verbose = verbose

        xs = [v.x_mil for v in geo.vias]
        ys = [v.y_mil for v in geo.vias]
        pitch = geo.pitch_mil

        # Infinite planes: extend domain well past the vias; board fills it and
        # runs into the lateral PML.
        self.dom_x0, self.dom_x1 = min(xs) - 2 * pitch, max(xs) + 2 * pitch
        self.dom_y0, self.dom_y1 = min(ys) - 2 * pitch, max(ys) + 2 * pitch
        self.board_z0, self.board_z1 = 0.0, geo.total_thickness_mil
        self.dom_z0 = self.board_z0 - AIR_MARGIN_Z_MIL
        self.dom_z1 = self.board_z1 + AIR_MARGIN_Z_MIL

        self.metal_layers = [l for l in geo.layers if l.kind == "metal"]
        self.ground_layers = [l for l in geo.layers if l.token == "G"]
        self.power_layers = [l for l in geo.layers if l.token == "P"]

        lam_min_m = C0 / (self.f_max_hz * np.sqrt(geo.permittivity))
        self.resolution_mil = (lam_min_m / MIL_TO_M) / self.cells_per_wavelength

        # The physical port reference impedance (computed, not assumed 50).
        self.port_Z = coaxial_impedance(geo.permittivity,
                                        geo.antipad_radius_mil,
                                        geo.via_radius_mil)
        if self.verbose:
            self._print_summary()

    def _print_summary(self):
        g = self.geo
        print(f"  ArrayModelBuilder v5 (radial lumped + Z-renorm) for {g.sim_id}")
        print(f"    domain (mil): x[{self.dom_x0:.1f},{self.dom_x1:.1f}] "
              f"y[{self.dom_y0:.1f},{self.dom_y1:.1f}] "
              f"z[{self.dom_z0:.1f},{self.dom_z1:.1f}]")
        print(f"    infinite planes: dielectric + G/P fill x/y into lateral PML")
        print(f"    port impedance : {self.port_Z:.1f} ohm "
              f"(computed coaxial Z of via-antipad; renormalized to 50 on output)")
        print(f"    mesh res       : {self.resolution_mil:.3f} mil "
              f"({self.cells_per_wavelength} cells/lambda @ 100 GHz)")
        print(f"    boundaries     : PML_8 all six faces")
        print(f"    n_ports        : {g.n_ports}")

    # ------------------------------------------------------------------
    def build_geometry(self, FDTD, CSX, excited_port):
        geo = self.geo
        eps0 = 8.8541878128e-12
        kappa = (2.0 * np.pi * self.f_max_hz * eps0
                 * geo.permittivity * geo.loss_tangent)
        diel = CSX.AddMaterial("dielectric", epsilon=geo.permittivity,
                               kappa=kappa)
        pec = CSX.AddMetal("pec")

        # Infinite board: dielectric fills whole x/y domain
        diel.AddBox([self.dom_x0, self.dom_y0, self.board_z0],
                    [self.dom_x1, self.dom_y1, self.board_z1],
                    priority=PRIO_DIELECTRIC)
        # Infinite planes: G/P fill whole x/y domain, into the PML
        for layer in self.metal_layers:
            if layer.token in ("G", "P"):
                pec.AddBox([self.dom_x0, self.dom_y0, layer.z_top_mil],
                           [self.dom_x1, self.dom_y1, layer.z_bot_mil],
                           priority=PRIO_PLANE)

        # Antipads (priority override)
        antipad_r = geo.antipad_radius_mil
        for via in geo.vias:
            clear = (self.ground_layers + self.power_layers if via.net == "S"
                     else self.power_layers if via.net == "G"
                     else self.ground_layers if via.net == "P" else [])
            for layer in clear:
                diel.AddCylinder([via.x_mil, via.y_mil, layer.z_top_mil],
                                 [via.x_mil, via.y_mil, layer.z_bot_mil],
                                 antipad_r, priority=PRIO_ANTIPAD)

        # Vias: solid PEC full height
        for via in geo.vias:
            pec.AddCylinder([via.x_mil, via.y_mil, self.board_z0],
                            [via.x_mil, via.y_mil, self.board_z1],
                            geo.via_radius_mil, priority=PRIO_VIA)

        ports = self._add_ports(FDTD, CSX, excited_port)
        self._build_mesh(CSX)
        return ports

    def _add_ports(self, FDTD, CSX, excited_port):
        """Radial lumped port across the antipad annulus, referenced to the
        computed coaxial impedance self.port_Z (NOT 50). Renormalized to 50 in
        run_and_extract.
        """
        geo = self.geo
        via_r = geo.via_radius_mil
        antipad_r = geo.antipad_radius_mil
        top_g = self.ground_layers[0]
        bot_g = self.ground_layers[-1]
        thin = via_r / 2.0

        ports = [None] * geo.n_ports
        sgn_to_via = {v.sgn_index: v for v in geo.vias if v.net == "S"}

        for pm in geo.se_port_map:
            se_idx = pm["se_index_0based"]
            via = sgn_to_via[pm["sgn"]]
            half = pm["half"]
            zlo, zhi = ((top_g.z_top_mil, top_g.z_bot_mil) if half == "pup"
                        else (bot_g.z_top_mil, bot_g.z_bot_mil))

            # radial: via wall (+x) out to antipad edge, thin in y, plane z-slab
            start = [via.x_mil + via_r, via.y_mil - thin, zlo]
            stop = [via.x_mil + antipad_r, via.y_mil + thin, zhi]

            exc = 1.0 if (excited_port is not None
                          and se_idx == excited_port) else 0.0
            port = FDTD.AddLumpedPort(se_idx + 1, self.port_Z,
                                      start, stop, "x",
                                      excite=exc, priority=PRIO_PORT)
            ports[se_idx] = port
        return ports

    def _build_mesh(self, CSX):
        mesh = CSX.GetGrid()
        mesh.SetDeltaUnit(MIL_TO_M)
        geo = self.geo
        res = self.resolution_mil

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
        for sgn, z0 in ((-1, self.board_z0), (+1, self.board_z1)):
            acc, step = 0.0, res
            while acc + step < AIR_MARGIN_Z_MIL:
                acc += step
                dense.append(round(z0 + sgn * acc, 6))
                step *= 1.8
        dense += [self.dom_z0, self.dom_z1]
        mesh.AddLine("z", sorted(set(round(z, 6) for z in dense)))

        x_lines = {self.dom_x0, self.dom_x1}
        y_lines = {self.dom_y0, self.dom_y1}
        r_i, r_o = geo.via_radius_mil, geo.antipad_radius_mil
        for v in geo.vias:
            for d in (-r_o, -r_i, 0.0, r_i, r_o):
                x_lines.add(round(v.x_mil + d, 6))
                y_lines.add(round(v.y_mil + d, 6))
        mesh.AddLine("x", sorted(x_lines))
        mesh.AddLine("y", sorted(y_lines))
        mesh.SmoothMeshLines("x", res)
        mesh.SmoothMeshLines("y", res)
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
                print(f"    [excite port {p_exc + 1}/{N}]")
            sim_dir = sim_root / f"excite_{p_exc:02d}"

            FDTD = openEMS(NrTS=MAX_TIMESTEPS, EndCriteria=END_CRITERIA)
            FDTD.SetGaussExcite((F_MAX_HZ + F_MIN_HZ) / 2,
                                (F_MAX_HZ - F_MIN_HZ) / 2)
            FDTD.SetBoundaryCond(["PML_8"] * 6)

            CSX = ContinuousStructure()
            FDTD.SetCSX(CSX)
            ports = self.build_geometry(FDTD, CSX, excited_port=p_exc)

            if not run:
                xml = sim_root / "model.xml"
                CSX.Write2XML(str(xml))
                if self.verbose:
                    print(f"    [dry-run] wrote {xml}")
                return None, None

            FDTD.Run(str(sim_dir), cleanup=True, verbose=0)

            # Wave separation at the port's OWN impedance (self.port_Z), not 50
            for m in range(N):
                ports[m].CalcPort(str(sim_dir), freq,
                                  ref_impedance=self.port_Z)
            a_inc = ports[p_exc].uf_inc
            for m in range(N):
                S[:, m, p_exc] = ports[m].uf_ref / a_inc

        # Renormalize the whole matrix from port_Z to 50 ohm (physics-correct,
        # frequency-consistent, fixes S11 AND S21 -- no DC scalar).
        S50 = self._renormalize(S, self.port_Z, freq)
        return freq, S50

    @staticmethod
    def _renormalize(S, z_from, freq):
        import skrf as rf
        ntwk = rf.Network(frequency=rf.Frequency.from_f(freq, unit="Hz"),
                          s=S, z0=z_from)
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
    args = ap.parse_args()

    print("=" * 78)
    print(f"Stage 04 v5 (radial lumped + impedance renorm): {args.sim}")
    print("=" * 78)

    geo = from_sim_folder(args.sim)
    print(geo.summary())
    print()
    builder = ArrayModelBuilder(geo, cells_per_wavelength=args.cells)
    sim_root = _THIS_DIR / "runs" / f"04_{args.sim}"

    if args.dry_run:
        builder.run_and_extract(sim_root, run=False)
        print("\nDry run complete. Ports referenced to "
              f"{builder.port_Z:.1f} ohm, renormalized to 50 on output.")
    elif args.run:
        print(f"\nFull 16-port solve. Ports at {builder.port_Z:.1f} ohm "
              f"-> renormalized to 50 ohm (no DC scalar).\n")
        freq, S = builder.run_and_extract(sim_root, run=True)
        out = _THIS_DIR / "results" / "04_openems"
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / f"{args.sim}_coax_openems_se.npz", freq=freq, S=S)
        print(f"\nSaved (50-ohm referenced, no scalar): "
              f"{out / (args.sim + '_coax_openems_se.npz')}")
        print("Run your existing stage 05 unchanged.")
    else:
        print("\nPass --dry-run or --run.")


if __name__ == "__main__":
    main()