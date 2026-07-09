"""
build_array_model.py
================================================================================
Stage 04 of the openEMS validation pipeline.

PURPOSE
    Turn a GeometryDescription (from stage 03) into a full-wave OpenEMS model
    of the complete via array, following the TUHH dataset conventions exactly.
    Runs the 16-port FDTD simulation.
"""

from __future__ import annotations
import argparse
import importlib.util
import sys
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------------
# Import stage 03
# ----------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("stage03", _THIS_DIR / "parse_geometry.py")
stage03 = importlib.util.module_from_spec(_spec)
sys.modules["stage03"] = stage03
_spec.loader.exec_module(stage03)

GeometryDescription = stage03.GeometryDescription
from_sim_folder = stage03.from_sim_folder

# ----------------------------------------------------------------------------
# Constants & Priorities
# ----------------------------------------------------------------------------
MIL_TO_M = 25.4e-6
C0 = 299792458.0

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
CELLS_PER_WAVELENGTH = 20          
END_CRITERIA = 1e-4                
MAX_TIMESTEPS = 60000              

# ============================================================================
# Model builder
# ============================================================================
class ArrayModelBuilder:
    def __init__(self, geo: GeometryDescription, f_max_hz: float = F_MAX_HZ,
                 cells_per_wavelength: int = CELLS_PER_WAVELENGTH, margin_mil: float | None = None):
        self.geo = geo
        self.f_max_hz = f_max_hz
        self.cells_per_wavelength = cells_per_wavelength
        
        xs = [v.x_um / 25.4 for v in geo.vias]  # Convert parsed um to mil for CSXCAD
        ys = [v.y_um / 25.4 for v in geo.vias]
        self.x_min, self.x_max = min(xs), max(xs)
        self.y_min, self.y_max = min(ys), max(ys)
        
        self.margin_mil = margin_mil if margin_mil is not None else (geo.pitch_um / 25.4)
        self.foot_x0, self.foot_x1 = self.x_min - self.margin_mil, self.x_max + self.margin_mil
        self.foot_y0, self.foot_y1 = self.y_min - self.margin_mil, self.y_max + self.margin_mil
        
        self.z_top = 0.0
        self.z_bot = geo.total_thickness_um / 25.4
        
        self.metal_layers = [l for l in geo.layers if l.kind == "metal"]
        self.ground_layers = [l for l in geo.layers if l.token == "G"]
        self.power_layers = [l for l in geo.layers if l.token == "P"]
        
        lam_min_m = C0 / (self.f_max_hz * np.sqrt(geo.permittivity))
        self.resolution_mil = (lam_min_m / MIL_TO_M) / self.cells_per_wavelength

    def build_geometry(self, FDTD, CSX, excited_port):
        geo = self.geo
        
        # Materials
        kappa = 2.0 * np.pi * self.f_max_hz * 8.8541878128e-12 * geo.permittivity * geo.loss_tangent
        diel = CSX.AddMaterial("dielectric", epsilon=geo.permittivity, kappa=kappa)
        pec = CSX.AddMetal("pec")
        
        # Dielectric block
        diel.AddBox([self.foot_x0, self.foot_y0, self.z_top], [self.foot_x1, self.foot_y1, self.z_bot], priority=PRIO_DIELECTRIC)
        
        # Metal planes
        for layer in self.metal_layers:
            if layer.token in ("G", "P"):
                pec.AddBox([self.foot_x0, self.foot_y0, layer.z_top_um/25.4], [self.foot_x1, self.foot_y1, layer.z_bot_um/25.4], priority=PRIO_PLANE)
        
        # Antipads (Clearance holes)
        antipad_r = geo.antipad_radius_um / 25.4
        for via in geo.vias:
            planes_to_clear = []
            if via.net == "S": planes_to_clear = self.ground_layers + self.power_layers
            elif via.net == "G": planes_to_clear = self.power_layers
            elif via.net == "P": planes_to_clear = self.ground_layers
            
            for layer in planes_to_clear:
                diel.AddCylinder([via.x_um/25.4, via.y_um/25.4, layer.z_top_um/25.4], [via.x_um/25.4, via.y_um/25.4, layer.z_bot_um/25.4], antipad_r, priority=PRIO_ANTIPAD)
        
        # Via Cylinders
        via_r = geo.via_radius_um / 25.4
        for via in geo.vias:
            pec.AddCylinder([via.x_um/25.4, via.y_um/25.4, self.z_top], [via.x_um/25.4, via.y_um/25.4, self.z_bot], via_r, priority=PRIO_VIA)
        
        ports = self._add_ports(FDTD, CSX, pec, excited_port)
        self._build_mesh(CSX)
        return ports

    def _add_ports(self, FDTD, CSX, pec, excited_port):
        geo = self.geo
        diel_layers = [l for l in geo.layers if l.kind == "dielectric"]
        first_diel, last_diel = diel_layers[0], diel_layers[-1]
        
        ports = [None] * geo.n_ports
        sgn_to_via = {v.sgn_index: v for v in geo.vias if v.net == "S"}
        
        for pm in geo.se_port_map:
            se_idx = pm["se_index_0based"]
            via = sgn_to_via[pm["sgn"]]
            
            z0, z1 = (first_diel.z_top_um/25.4, first_diel.z_bot_um/25.4) if pm["half"] == "pup" else (last_diel.z_top_um/25.4, last_diel.z_bot_um/25.4)
            r = geo.via_radius_um / 25.4
            
            exc = 1.0 if (excited_port is not None and se_idx == excited_port) else 0.0
            port = FDTD.AddLumpedPort(se_idx + 1, 50.0, [via.x_um/25.4 - r, via.y_um/25.4 - r, z0], [via.x_um/25.4 + r, via.y_um/25.4 + r, z1], "z", excite=exc, priority=PRIO_PORT)
            ports[se_idx] = port
            
        return ports

    def _build_mesh(self, CSX):
        mesh = CSX.GetGrid()
        mesh.SetDeltaUnit(MIL_TO_M)
        res = self.resolution_mil
        
        z_lines = set(round(l.z_top_um/25.4, 6) for l in self.geo.layers) | set(round(l.z_bot_um/25.4, 6) for l in self.geo.layers)
        
        # Densify thick dielectric gaps to satisfy FDTD limits
        z_sorted = sorted(z_lines)
        dense_z = list(z_sorted)
        for a, b in zip(z_sorted, z_sorted[1:]):
            gap = b - a
            if gap > res:
                n = int(np.ceil(gap / res))
                dense_z.extend(np.linspace(a, b, n + 1).tolist())
        mesh.AddLine("z", sorted(set(round(z, 6) for z in dense_z)))
        
        x_lines, y_lines = {self.foot_x0, self.foot_x1}, {self.foot_y0, self.foot_y1}
        for v in self.geo.vias:
            x_m, y_m = v.x_um/25.4, v.y_um/25.4
            r_v, r_a = self.geo.via_radius_um/25.4, self.geo.antipad_radius_um/25.4
            x_lines.update([x_m - r_v, x_m + r_v, x_m - r_a, x_m + r_a, x_m])
            y_lines.update([y_m - r_v, y_m + r_v, y_m - r_a, y_m + r_a, y_m])
            
        mesh.AddLine("x", sorted(x_lines))
        mesh.AddLine("y", sorted(y_lines))
        
        for axis in ["x", "y", "z"]: mesh.SmoothMeshLines(axis, res)

    def run_and_extract(self, sim_root: Path, run: bool = True):
        from CSXCAD import ContinuousStructure
        from openEMS import openEMS
        
        sim_root.mkdir(parents=True, exist_ok=True)
        N = self.geo.n_ports
        freq = np.linspace(F_MIN_HZ, F_MAX_HZ, N_FREQ)
        S = np.zeros((N_FREQ, N, N), dtype=complex)
        
        if not run:
            FDTD = openEMS(NrTS=MAX_TIMESTEPS, EndCriteria=END_CRITERIA)
            CSX = ContinuousStructure()
            FDTD.SetCSX(CSX)
            self.build_geometry(FDTD, CSX, excited_port=None)
            xml = sim_root / "model.xml"
            CSX.Write2XML(str(xml))
            print(f"    [dry-run] wrote {xml}")
            print(f"    Inspect with: AppCSXCAD {xml}")
            return None, None

        # -----------------------------------------------------------
        # THE ACTUAL SOLVER LOOP (Excites each of the 16 ports)
        # -----------------------------------------------------------
        for p_exc in range(N):
            print(f"    [Excite port {p_exc+1}/{N}]")
            sim_dir = sim_root / f"excite_{p_exc:02d}"
            FDTD = openEMS(NrTS=MAX_TIMESTEPS, EndCriteria=END_CRITERIA)
            fc = (F_MAX_HZ - F_MIN_HZ) / 2.0
            f0 = (F_MAX_HZ + F_MIN_HZ) / 2.0
            FDTD.SetGaussExcite(f0, fc)
            FDTD.SetBoundaryCond(["MUR"] * 6)
            
            CSX = ContinuousStructure()
            FDTD.SetCSX(CSX)
            ports = self.build_geometry(FDTD, CSX, excited_port=p_exc)
            
            # Fire the C++ FDTD engine
            FDTD.Run(str(sim_dir), cleanup=True, verbose=0)
            
            # Extract port measurements
            for p_meas in range(N):
                ports[p_meas].CalcPort(str(sim_dir), freq, ref_impedance=50.0)
            a_inc = ports[p_exc].uf_inc
            for p_meas in range(N):
                b_ref = ports[p_meas].uf_ref
                S[:, p_meas, p_exc] = b_ref / a_inc
                
        return freq, S

# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=str, default="sim_pkg_0017")
    parser.add_argument("--dry-run", action="store_true", help="build the model + write CSXCAD XML, no solve")
    parser.add_argument("--run", action="store_true", help="run the full N-port FDTD (slow)")
    args = parser.parse_args()

    geo = from_sim_folder(args.sim)
    builder = ArrayModelBuilder(geo)
    sim_root = _THIS_DIR / "runs" / f"04_{args.sim}"
    
    if args.dry_run:
        builder.run_and_extract(sim_root, run=False)
    elif args.run:
        print(f"\nStarting full {geo.n_ports}-port solve. This will take a while...\n")
        freq, S = builder.run_and_extract(sim_root, run=True)
        
        out = _THIS_DIR / "results" / "04_openems"
        out.mkdir(parents=True, exist_ok=True)
        save_path = out / f"{args.sim}_openems_se.npz"
        np.savez(save_path, freq=freq, S=S)
        
        print(f"\nSaved single-ended S-matrix: {save_path}")
        print("Proceed to stage 05 for the three-way comparison.")
    else:
        print("Please specify --dry-run or --run")

if __name__ == "__main__":
    main()