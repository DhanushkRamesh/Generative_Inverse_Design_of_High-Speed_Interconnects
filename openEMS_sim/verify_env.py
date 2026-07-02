"""
01_verify_environment.py
================================================================================
Stage 01 of the openEMS validation pipeline.

PURPOSE
    Verify that the full toolchain needed for stages 02-08 works on this
    machine BEFORE any project-specific geometry is attempted:

      1. Imports and versions: CSXCAD, openEMS, numpy, matplotlib, scikit-rf.
      2. A minimal end-to-end FDTD smoke simulation: a small PEC dipole fed by
         a lumped port inside absorbing boundaries. This exercises geometry
         creation, meshing, excitation, the solver binary, and port
         post-processing (voltage/current -> S11) in under a minute.
      3. scikit-rf mixed-mode capability check: builds a synthetic 4-port
         network and runs se2gmm(p=2), which is the exact call stages 05-07
         rely on to convert OpenEMS single-ended results to the Sdd/Sdc/Scc
         quadrants used by the thesis models.

GATE (all must pass)
      [PASS] all imports
      [PASS] smoke sim runs and produces a finite S11
      [PASS] se2gmm(p=2) executes and returns a 4x4 mixed-mode matrix

USAGE
    cd ~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/openEMS_Sim
    python 01_verify_environment.py
"""

import os
import sys
import tempfile
import traceback

RESULTS = []  # (name, passed, detail)


def record(name, passed, detail=""):
    RESULTS.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


# ----------------------------------------------------------------------------
# 1) Imports and versions
# ----------------------------------------------------------------------------
print("=" * 78)
print("Stage 01: environment verification")
print("=" * 78)
print("\n--- 1. Imports ---")

try:
    import numpy as np
    record("numpy import", True, f"v{np.__version__}")
except Exception as e:
    record("numpy import", False, str(e))
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe
    record("matplotlib import", True, f"v{matplotlib.__version__}")
except Exception as e:
    record("matplotlib import", False, str(e))

try:
    import CSXCAD
    ver = getattr(CSXCAD, "__version__", "unknown")
    record("CSXCAD import", True, f"v{ver}")
except Exception as e:
    record("CSXCAD import", False, str(e))
    print("\nCSXCAD failed to import - openEMS Python bindings are not"
          " installed correctly. Fix before proceeding (see"
          " https://docs.openems.de/install).")
    sys.exit(1)

try:
    import openEMS
    ver = getattr(openEMS, "__version__", "unknown")
    record("openEMS import", True, f"v{ver}")
except Exception as e:
    record("openEMS import", False, str(e))
    sys.exit(1)

try:
    import skrf
    record("scikit-rf import", True, f"v{skrf.__version__}")
    HAVE_SKRF = True
except Exception as e:
    record("scikit-rf import", False,
           f"{e}  -> pip install scikit-rf (required from stage 05 onward)")
    HAVE_SKRF = False


# ----------------------------------------------------------------------------
# 2) Minimal FDTD smoke simulation
#    A ~14 mm PEC strip dipole along z with a 1 mm lumped-port feed gap,
#    in free space with MUR boundaries. Gaussian excitation around 10 GHz
#    (lambda ~ 30 mm, so the structure resonates near half-wave within the
#    excited band). Small mesh -> runs in seconds; the point is end-to-end
#    plumbing, not physical accuracy.
# ----------------------------------------------------------------------------
print("\n--- 2. FDTD smoke simulation (small dipole + lumped port) ---")

try:
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS as openEMS_solver

    sim_path = os.path.join(tempfile.gettempdir(), "openems_smoke_test")

    # Cap the timestep count so the smoke test can never run away.
    FDTD = openEMS_solver(NrTS=20000, EndCriteria=1e-4)
    f0 = 10e9      # center 10 GHz
    fc = 8e9       # 20 dB corner bandwidth
    FDTD.SetGaussExcite(f0, fc)
    FDTD.SetBoundaryCond(["MUR"] * 6)

    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    unit = 1e-3                    # mm
    mesh.SetDeltaUnit(unit)

    # Simulation volume: 40 mm cube around the dipole
    box = 20.0
    # Coarse background mesh ~1 mm plus explicit lines at structure edges
    mesh.AddLine("x", np.linspace(-box, box, 41))
    mesh.AddLine("y", np.linspace(-box, box, 41))
    mesh.AddLine("z", np.linspace(-box, box, 41))

    # Dipole: two PEC arms along z, 0.5 mm wide, total tip-to-tip 14 mm,
    # with a 1 mm feed gap at the centre (from z=-0.5 to z=+0.5).
    arm_w = 0.25   # half-width in x and y -> 0.5 mm square cross-section
    pec = CSX.AddMetal("dipole")
    pec.AddBox([-arm_w, -arm_w, 0.5],  [arm_w, arm_w, 7.0])    # upper arm
    pec.AddBox([-arm_w, -arm_w, -7.0], [arm_w, arm_w, -0.5])   # lower arm
    # Ensure mesh lines exactly at metal edges and the feed gap
    for zline in (-7.0, -0.5, 0.5, 7.0):
        mesh.AddLine("z", zline)
    for xy in (-arm_w, arm_w):
        mesh.AddLine("x", xy)
        mesh.AddLine("y", xy)

    # Lumped port across the feed gap, excitation along z, 50 ohm
    port = FDTD.AddLumpedPort(1, 50.0,
                               [-arm_w, -arm_w, -0.5],
                               [arm_w,  arm_w,  0.5],
                               "z", excite=1.0, priority=10)

    print(f"  Running solver (sim dir: {sim_path}) ...")
    FDTD.Run(sim_path, cleanup=True, verbose=0)

    # Post-process: port voltages/currents -> S11 over the excited band
    freq = np.linspace(2e9, 18e9, 161)
    port.CalcPort(sim_path, freq, ref_impedance=50)
    s11 = port.uf_ref / port.uf_inc
    s11_db = 20.0 * np.log10(np.abs(s11) + 1e-15)

    finite = np.all(np.isfinite(s11_db))
    dip_db = float(s11_db.min())
    dip_f = float(freq[int(np.argmin(s11_db))] / 1e9)
    record("smoke sim ran + S11 computed", bool(finite),
           f"min |S11| = {dip_db:.1f} dB at {dip_f:.1f} GHz "
           f"(a dip somewhere in-band indicates the port and solver work; "
           f"exact value is not the point)")
    if not finite:
        raise RuntimeError("S11 contains non-finite values")

except Exception as e:
    record("smoke sim ran + S11 computed", False, str(e))
    traceback.print_exc(limit=4)


# ----------------------------------------------------------------------------
# 3) scikit-rf mixed-mode conversion check (the exact call used later)
# ----------------------------------------------------------------------------
print("\n--- 3. scikit-rf mixed-mode (se2gmm) check ---")

if HAVE_SKRF:
    try:
        import skrf as rf
        rng = np.random.default_rng(0)
        F = 11
        frequency = rf.Frequency(1, 10, F, unit="GHz")
        # Build a small random passive-ish reciprocal 4-port for the API check
        A = rng.normal(size=(F, 4, 4)) + 1j * rng.normal(size=(F, 4, 4))
        S = 0.5 * (A + A.transpose(0, 2, 1)) * 0.1   # symmetric, small
        ntwk = rf.Network(frequency=frequency, s=S, z0=50)
        # The stage-05+ convention: ports renumbered so pairs are (0,1), (2,3),
        # then generalized mixed-mode with p=2 differential pairs.
        mm = ntwk.copy()
        mm.se2gmm(p=2)
        ok = mm.s.shape == (F, 4, 4)
        record("se2gmm(p=2) executes", ok, f"mixed-mode S shape {mm.s.shape}")
    except Exception as e:
        record("se2gmm(p=2) executes", False, str(e))
        traceback.print_exc(limit=3)
else:
    record("se2gmm(p=2) executes", False, "scikit-rf not installed")


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("Stage 01 summary")
print("=" * 78)
n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
for name, ok, detail in RESULTS:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
if n_fail == 0:
    print("\nGATE PASSED")
else:
    print(f"\nGATE FAILED - {n_fail} check(s) failed. Fix before proceeding.")
    sys.exit(1)