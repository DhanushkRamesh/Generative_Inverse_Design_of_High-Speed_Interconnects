# -*- coding: utf-8 -*-
"""
msl_notch_sample_tutorial.py
================================================================================
Stage 02 of the openEMS validation pipeline.

PURPOSE
    Run the OFFICIAL openEMS Microstrip Notch Filter tutorial as a known-good
    verification of the full simulation chain (mesh, MSL ports, FDTD run,
    S-parameter extraction) before any project-specific geometry is built.

    The simulation setup below (geometry, mesh, ports, excitation) is taken
    VERBATIM from the official tutorial shipped with openEMS:
        python/Tutorials/MSL_NotchFilter.py
        (c) 2016-2023 Thorsten Liebig, GPL - see the openEMS repository.
        https://docs.openems.de/python/openEMS/Tutorials/MSL_NotchFilter.html
    Local adaptations, clearly marked with "ADAPTED:", are limited to:
      - headless matplotlib backend + saving the figure instead of showing it
      - simulation directory under openEMS_Sim/runs/
      - saving S-parameters to CSV under openEMS_Sim/results/
      - an explicit GATE check at the end (deep S21 notch in the 2-4 GHz band,
        which is the documented behaviour of this quarter-wave stub filter:
        a 12 mm open stub on eps_r = 3.66 substrate notches near ~3 GHz).

GATE
    S21 exhibits a notch deeper than -20 dB somewhere in 2-4 GHz, while the
    low-frequency S21 (< 1 GHz) stays above -3 dB (line transmits). If this
    passes, the toolchain reproduces a published reference result and stage
    03+ can be trusted to fail only for project-specific reasons.

RUNTIME
    A few minutes on a typical machine (the official tutorial as-is).

USAGE
    cd ~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/openEMS_Sim
    python msl_notch_sample_tutorial.py
"""

### Import Libraries
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # ADAPTED: headless backend
import matplotlib.pyplot as plt

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import *   # provides C0


### Setup the simulation
# ADAPTED: run/results directories inside openEMS_Sim instead of tempdir
THIS_DIR = Path(__file__).resolve().parent
Sim_Path = str(THIS_DIR / "runs" / "msl_notch")
RESULTS_DIR = THIS_DIR / "results" / "msl_notch"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

post_proc_only = False

unit = 1e-6  # specify everything in um
MSL_length = 50000
MSL_width = 600
substrate_thickness = 254
substrate_epr = 3.66
stub_length = 12e3
f_max = 7e9

### Setup FDTD parameters & excitation function
FDTD = openEMS()
FDTD.SetGaussExcite(f_max / 2, f_max / 2)
FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'MUR', 'MUR', 'PEC', 'MUR'])

### Setup Geometry & Mesh
CSX = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)

resolution = C0 / (f_max * np.sqrt(substrate_epr)) / unit / 50  # lambda/50
third_mesh = np.array([2 * resolution / 3, -resolution / 3]) / 4

## Do manual meshing
mesh.AddLine('x', 0)
mesh.AddLine('x', MSL_width / 2 + third_mesh)
mesh.AddLine('x', -MSL_width / 2 - third_mesh)
mesh.SmoothMeshLines('x', resolution / 4)

mesh.AddLine('x', [-MSL_length, MSL_length])
mesh.SmoothMeshLines('x', resolution)

mesh.AddLine('y', 0)
mesh.AddLine('y', MSL_width / 2 + third_mesh)
mesh.AddLine('y', -MSL_width / 2 - third_mesh)
mesh.SmoothMeshLines('y', resolution / 4)

mesh.AddLine('y', [-15 * MSL_width, 15 * MSL_width + stub_length])
mesh.AddLine('y', (MSL_width / 2 + stub_length) + third_mesh)
mesh.SmoothMeshLines('y', resolution)

mesh.AddLine('z', np.linspace(0, substrate_thickness, 5))
mesh.AddLine('z', 3000)
mesh.SmoothMeshLines('z', resolution)

## Add the substrate
substrate = CSX.AddMaterial('RO4350B', epsilon=substrate_epr)
start = [-MSL_length, -15 * MSL_width, 0]
stop = [+MSL_length, +15 * MSL_width + stub_length, substrate_thickness]
substrate.AddBox(start, stop)

## MSL port setup
port = [None, None]
pec = CSX.AddMetal('PEC')
portstart = [-MSL_length, -MSL_width / 2, substrate_thickness]
portstop = [0, MSL_width / 2, 0]
port[0] = FDTD.AddMSLPort(1, pec, portstart, portstop, 'x', 'z', excite=-1,
                           FeedShift=10 * resolution,
                           MeasPlaneShift=MSL_length / 3, priority=10)

portstart = [MSL_length, -MSL_width / 2, substrate_thickness]
portstop = [0, MSL_width / 2, 0]
port[1] = FDTD.AddMSLPort(2, pec, portstart, portstop, 'x', 'z',
                           MeasPlaneShift=MSL_length / 3, priority=10)

## Filter-Stub Definition
start = [-MSL_width / 2, MSL_width / 2, substrate_thickness]
stop = [MSL_width / 2, MSL_width / 2 + stub_length, substrate_thickness]
pec.AddBox(start, stop, priority=10)

### Run the simulation
if not post_proc_only:
    print(f"Running official MSL notch tutorial (sim dir: {Sim_Path})")
    print("This takes a few minutes ...")
    os.makedirs(Sim_Path, exist_ok=True)
    FDTD.Run(Sim_Path, cleanup=True)

### Post-processing
f = np.linspace(1e6, f_max, 1601)
for p in port:
    p.CalcPort(Sim_Path, f, ref_impedance=50)

s11 = port[0].uf_ref / port[0].uf_inc
s21 = port[1].uf_ref / port[0].uf_inc

s11_db = 20.0 * np.log10(np.abs(s11) + 1e-15)
s21_db = 20.0 * np.log10(np.abs(s21) + 1e-15)

## Plot s-parameter  (ADAPTED: save instead of show)
fig, axis = plt.subplots(num="S-Parameters", tight_layout=True)
axis.plot(f / 1e9, s11_db, 'k-', linewidth=2, label='$S_{11}$')
axis.plot(f / 1e9, s21_db, 'r--', linewidth=2, label='$S_{21}$')
axis.grid()
axis.set_xmargin(0)
axis.set_xlabel('Frequency (GHz)')
axis.set_ylabel('S-Parameter (dB)')
axis.set_title('Official MSL notch tutorial - verification run')
axis.legend()
fig_path = RESULTS_DIR / "msl_notch_s_parameters.png"
fig.savefig(fig_path, dpi=150)
plt.close(fig)
print(f"Saved figure: {fig_path}")

# ADAPTED: save raw S-parameters for the record
csv_path = RESULTS_DIR / "msl_notch_s_parameters.csv"
np.savetxt(csv_path,
           np.column_stack([f, s11_db, s21_db]),
           delimiter=",", header="freq_Hz,S11_dB,S21_dB", comments="")
print(f"Saved CSV:    {csv_path}")

# ----------------------------------------------------------------------------
# ADAPTED: GATE check
# The 12 mm open quarter-wave stub on eps_r 3.66 produces a deep S21 notch
# in the low-GHz range (documented tutorial result). Verify:
#   (a) a notch deeper than -20 dB exists within 2-4 GHz
#   (b) transmission below 1 GHz is healthy (S21 > -3 dB)
# ----------------------------------------------------------------------------
band = (f >= 2e9) & (f <= 4e9)
lowf = f <= 1e9
notch_db = float(s21_db[band].min())
notch_f = float(f[band][int(np.argmin(s21_db[band]))] / 1e9)
lowf_s21 = float(s21_db[lowf].max())

print("\n" + "=" * 70)
print("Stage 02 GATE")
print("=" * 70)
print(f"  S21 notch in 2-4 GHz : {notch_db:7.1f} dB at {notch_f:.2f} GHz "
      f"(require < -20 dB)")
print(f"  S21 below 1 GHz (max): {lowf_s21:7.1f} dB (require > -3 dB)")

if notch_db < -20.0 and lowf_s21 > -3.0:
    print("\nGATE PASSED - the toolchain reproduces the published reference "
          "result.\nProceed to stage 03 (dataset geometry parser).")
else:
    print("\nGATE FAILED - the tutorial did not reproduce its documented "
          "behaviour.\nDo NOT proceed. Likely causes: incomplete openEMS "
          "build, PML/engine\nissues, or an interrupted run. Inspect the "
          "saved figure and the solver\nlog in the sim directory.")
    raise SystemExit(1)