"""
build_array_model.py
Refined FDTD engine wrapper for openEMS.
Fixed: Z-boundary padding, mesh smoothing, port stability, and directory creation.
"""

from __future__ import annotations
import argparse
import importlib.util
import sys
from pathlib import Path
import numpy as np

# Load stage 03 utility
_THIS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("stage03", _THIS_DIR / "parse_geometry.py")
stage03 = importlib.util.module_from_spec(_spec)
sys.modules["stage03"] = stage03
_spec.loader.exec_module(stage03)
GeometryDescription = stage03.GeometryDescription
from_sim_folder = stage03.from_sim_folder

# Constants
MIL_TO_M = 25.4e-6
C0 = 299792458.0
PRIO_DIELECTRIC, PRIO_PLANE, PRIO_ANTIPAD, PRIO_VIA, PRIO_PORT = 0, 10, 20, 30, 40

class ArrayModelBuilder:
    def __init__(self, geo: GeometryDescription):
        self.geo = geo
        self.x_min = min(v.x_um for v in geo.vias)
        self.x_max = max(v.x_um for v in geo.vias)
        self.y_min = min(v.y_um for v in geo.vias)
        self.y_max = max(v.y_um for v in geo.vias)
        
        # Add padding to footprint
        margin = geo.pitch_um * 2
        self.foot_x0, self.foot_x1 = self.x_min - margin, self.x_max + margin
        self.foot_y0, self.foot_y1 = self.y_min - margin, self.y_max + margin
        self.z_top, self.z_bot = 0.0, geo.total_thickness_um

    def build_geometry(self, FDTD, CSX, excited_port):
        geo = self.geo
        diel = CSX.AddMaterial("dielectric", epsilon=geo.permittivity)
        pec = CSX.AddMetal("pec")
        
        # Base substrate
        diel.AddBox([self.foot_x0, self.foot_y0, self.z_top], [self.foot_x1, self.foot_y1, self.z_bot], priority=PRIO_DIELECTRIC)
        
        # Planes
        for layer in geo.layers:
            if layer.token in ("G", "P"):
                pec.AddBox([self.foot_x0, self.foot_y0, layer.z_top_um], [self.foot_x1, self.foot_y1, layer.z_bot_um], priority=PRIO_PLANE)
        
        # Vias and Antipads
        for via in geo.vias:
            # Add antipads in planes
            for layer in geo.layers:
                if layer.token in ("G", "P"):
                    # Check if layer needs antipad based on via net
                    if (via.net == "S" and layer.token in ("G", "P")) or \
                       (via.net == "G" and layer.token == "P") or \
                       (via.net == "P" and layer.token == "G"):
                        diel.AddCylinder([via.x_um, via.y_um, layer.z_top_um], [via.x_um, via.y_um, layer.z_bot_um], geo.antipad_radius_um, priority=PRIO_ANTIPAD)
            
            # Via cylinder
            pec.AddCylinder([via.x_um, via.y_um, self.z_top], [via.x_um, via.y_um, self.z_bot], geo.via_radius_um, priority=PRIO_VIA)

        # Ports
        ports = [None] * geo.n_ports
        for pm in geo.se_port_map:
            via = next(v for v in geo.vias if v.sgn_index == pm["sgn"])
            z0, z1 = (geo.layers[0].z_top_um, geo.layers[0].z_bot_um) if pm["half"] == "pup" else (geo.layers[-1].z_top_um, geo.layers[-1].z_bot_um)
            exc = 1.0 if (excited_port is not None and pm["se_index_0based"] == excited_port) else 0.0
            ports[pm["se_index_0based"]] = FDTD.AddLumpedPort(pm["se_index_0based"] + 1, 50.0, [via.x_um, via.y_um, z0], [via.x_um, via.y_um, z1], "z", excite=exc, priority=PRIO_PORT)
        return ports

    def _build_mesh(self, CSX):
        mesh = CSX.GetGrid()
        mesh.SetDeltaUnit(MIL_TO_M)
        
        # Padding for PML
        pad = 500.0 # um
        
        # Z-Mesh
        z_lines = {l.z_top_um for l in self.geo.layers} | {l.z_bot_um for l in self.geo.layers}
        z_lines.update([self.z_top - pad, self.z_bot + pad])
        mesh.AddLine("z", sorted(list(z_lines)))
        
        # X/Y-Mesh
        x_lines = {self.foot_x0, self.foot_x1, self.foot_x0 - pad, self.foot_x1 + pad}
        y_lines = {self.foot_y0, self.foot_y1, self.foot_y0 - pad, self.foot_y1 + pad}
        for v in self.geo.vias:
            x_lines.update([v.x_um - self.geo.via_radius_um, v.x_um + self.geo.via_radius_um])
            y_lines.update([v.y_um - self.geo.via_radius_um, v.y_um + self.geo.via_radius_um])
        
        mesh.AddLine("x", sorted(list(x_lines)))
        mesh.AddLine("y", sorted(list(y_lines)))
        mesh.SmoothMeshLines("all", 100.0) # Ensure no giant cells

    def run_solve(self, sim_root: Path):
        from openEMS import openEMS
        from CSXCAD import ContinuousStructure
        
        for p in range(self.geo.n_ports):
            sim_dir = sim_root / f"p{p}"
            # FIX: Create the directory before openEMS tries to write to it
            sim_dir.mkdir(parents=True, exist_ok=True)
            
            FDTD = openEMS(NrTS=80000, EndCriteria=1e-4)
            FDTD.SetGaussExcite(50e9, 50e9)
            FDTD.SetBoundaryCond(["PML_8"]*6)
            CSX = ContinuousStructure()
            FDTD.SetCSX(CSX)
            ports = self.build_geometry(FDTD, CSX, p)
            self._build_mesh(CSX)
            FDTD.Run(str(sim_dir), cleanup=True)

if __name__ == "__main__":
    geo = from_sim_folder("sim_pkg_0017")
    builder = ArrayModelBuilder(geo)
    builder.run_solve(Path("./runs"))