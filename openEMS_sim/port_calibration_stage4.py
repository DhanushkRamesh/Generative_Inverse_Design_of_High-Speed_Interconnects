"""
04b_port_calibration.py
================================================================================
Stage 04b of the openEMS validation pipeline (port-geometry calibration).

PURPOSE
    Stage 05 revealed that the array ports reflect ~5 dB at DC and the through
    path is ~12 dB down, instead of the near-0 dB through-response CONMLS shows.
    Root cause: the lumped-port box is centred on the via, so the PEC via runs
    straight through the port and shorts it.

    Rather than debug this on the expensive 16-port array, this script builds a
    TRIVIAL single signal-via structure (one via between a top and bottom ground
    plane, with an antipad clearance) and tries several PORT GEOMETRIES to find
    the one that gives a clean, low-reflection through-response. Each trial is a
    2-port sim that runs in seconds.

    Once a port geometry passes here, the SAME definition is transplanted into
    stage 04's _add_ports().

WHAT A GOOD PORT LOOKS LIKE
    For a short through-via referenced to ground, at low frequency:
      |S21| ~ 0 dB      (signal passes straight through the via)
      |S11| deep        (well matched)
    A shorted/broken port shows |S11| ~ -5 dB (mostly reflecting) and |S21|
    well below 0 dB -- exactly the array symptom.

PORT GEOMETRY CANDIDATES
    The via is a PEC cylinder along z through the whole stack. A port connects
    one via END to the adjacent ground plane. Candidates:

    (A) z-gap port, via stops short of the plane:
        The via runs from just below the top plane to just above the bottom
        plane, leaving a small z gap at each end. A z-directed lumped port fills
        that gap between the via end-cap and the plane. Via does NOT pass through
        the port -> no short. This matches a via-end wave port referenced to the
        adjacent plane.

    (B) radial port across the antipad annulus:
        A lumped element from the via wall radially outward to the ground-plane
        inner edge (the antipad rim), in the plane's z-slab. Excites the coaxial
        via-to-plane mode.

    This script implements (A) as the primary candidate (cleanest, matches the
    CONMLS via-end port) and reports its S-parameters. (B) is provided as an
    alternative if (A) underperforms.

USAGE
    python 04b_port_calibration.py --geometry A
    python 04b_port_calibration.py --geometry B
    # inspect the printed |S11|/|S21|; the winner is transplanted to stage 04.

This uses representative sim_pkg_0017 dimensions (via radius, antipad, pitch,
TMET, TDIEL, eps_r) so the calibration is at the right scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# sim_pkg_0017 representative dimensions (mil)
VIA_RADIUS = 3.294504
ANTIPAD_RADIUS = 20.534246
TMET = 2.172022
TDIEL = 18.249897
EPS_R = 2.9235348
TAN_D = 0.013408833
PITCH = 57.932522

MIL_TO_M = 25.4e-6
C0 = 299792458.0

F_MIN = 0.25e9
F_MAX = 100e9
N_FREQ = 401

PRIO_DIEL = 0
PRIO_PLANE = 10
PRIO_ANTIPAD = 20
PRIO_VIA = 30
PRIO_PORT = 40

_THIS_DIR = Path(__file__).resolve().parent


def build_single_via(FDTD, CSX, geometry: str, excite_port: int,
                     port_gap_mil: float):
    """Build a 1-signal-via, 2-port test structure.

    Stack (small, enough to be a clean through-via test):
        top ground plane  (z: 0 .. TMET)
        dielectric        (z: TMET .. TMET+TDIEL)
        bottom ground plane (z: TMET+TDIEL .. 2*TMET+TDIEL)

    A signal via runs along z through the stack (with a z-gap at each end for
    geometry A), clearing both ground planes with an antipad.

    Ports:
        port 1 = top end of via to top ground plane
        port 2 = bottom end of via to bottom ground plane
    """
    # z layout
    z0 = 0.0
    z_topplane_bot = TMET
    z_botplane_top = TMET + TDIEL
    z_bot = 2 * TMET + TDIEL

    # footprint: a few pitches around the single via at origin
    half = 1.5 * PITCH
    fx0, fx1 = -half, half
    fy0, fy1 = -half, half

    eps0 = 8.8541878128e-12
    kappa = 2 * np.pi * F_MAX * eps0 * EPS_R * TAN_D
    diel = CSX.AddMaterial("dielectric", epsilon=EPS_R, kappa=kappa)
    pec = CSX.AddMetal("pec")

    # dielectric block
    diel.AddBox([fx0, fy0, z0], [fx1, fy1, z_bot], priority=PRIO_DIEL)

    # top + bottom ground planes
    pec.AddBox([fx0, fy0, z0], [fx1, fy1, z_topplane_bot], priority=PRIO_PLANE)
    pec.AddBox([fx0, fy0, z_botplane_top], [fx1, fy1, z_bot], priority=PRIO_PLANE)

    # antipad clearance through both planes (dielectric cylinder, higher prio)
    diel.AddCylinder([0, 0, z0], [0, 0, z_topplane_bot], ANTIPAD_RADIUS,
                     priority=PRIO_ANTIPAD)
    diel.AddCylinder([0, 0, z_botplane_top], [0, 0, z_bot], ANTIPAD_RADIUS,
                     priority=PRIO_ANTIPAD)

    ports = [None, None]

    if geometry == "A":
        # via stops short of each plane surface by port_gap_mil; z-directed
        # lumped ports fill the gaps.
        via_top = z0 + port_gap_mil          # via starts below top surface
        via_bot = z_bot - port_gap_mil       # via ends above bottom surface
        pec.AddCylinder([0, 0, via_top], [0, 0, via_bot], VIA_RADIUS,
                        priority=PRIO_VIA)

        # port 1: from top plane surface (z0... but plane occupies 0..TMET, so
        # reference at z_topplane_bot? Use gap between via_top and the top
        # plane's inner surface at z_topplane_bot). We put the port in the gap
        # [z0, via_top] connecting the top plane to the via top cap, z-directed.
        # A lumped port spanning a small x-y footprint at the via location.
        r = VIA_RADIUS
        p1 = FDTD.AddLumpedPort(1, 50.0,
                                [-r, -r, z0], [r, r, via_top],
                                "z", excite=(1.0 if excite_port == 0 else 0.0),
                                priority=PRIO_PORT)
        p2 = FDTD.AddLumpedPort(2, 50.0,
                                [-r, -r, via_bot], [r, r, z_bot],
                                "z", excite=(1.0 if excite_port == 1 else 0.0),
                                priority=PRIO_PORT)
        ports = [p1, p2]

    elif geometry == "B":
        # via full-height; radial ports across the antipad annulus at each end.
        pec.AddCylinder([0, 0, z0], [0, 0, z_bot], VIA_RADIUS, priority=PRIO_VIA)
        # radial port: from via wall (x=VIA_RADIUS) to antipad edge
        # (x=ANTIPAD_RADIUS), thin in y, within the top plane's z-slab.
        thin = VIA_RADIUS
        p1 = FDTD.AddLumpedPort(1, 50.0,
                                [VIA_RADIUS, -thin, z0],
                                [ANTIPAD_RADIUS, thin, z_topplane_bot],
                                "x", excite=(1.0 if excite_port == 0 else 0.0),
                                priority=PRIO_PORT)
        p2 = FDTD.AddLumpedPort(2, 50.0,
                                [VIA_RADIUS, -thin, z_botplane_top],
                                [ANTIPAD_RADIUS, thin, z_bot],
                                "x", excite=(1.0 if excite_port == 1 else 0.0),
                                priority=PRIO_PORT)
        ports = [p1, p2]
    else:
        raise ValueError(f"unknown geometry {geometry!r}")

    # ---- mesh ----
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(MIL_TO_M)
    res = (C0 / (F_MAX * np.sqrt(EPS_R)) / MIL_TO_M) / 20.0

    zlines = {z0, z_topplane_bot, z_botplane_top, z_bot}
    if geometry == "A":
        zlines |= {port_gap_mil, z_bot - port_gap_mil}
    zl = sorted(zlines)
    dense = list(zl)
    for a, b in zip(zl, zl[1:]):
        if b - a > res:
            n = int(np.ceil((b - a) / res))
            dense += np.linspace(a, b, n + 1).tolist()
    mesh.AddLine("z", sorted(set(round(z, 6) for z in dense)))

    xl = {fx0, fx1, -VIA_RADIUS, VIA_RADIUS, -ANTIPAD_RADIUS, ANTIPAD_RADIUS, 0.0}
    yl = set(xl)
    mesh.AddLine("x", sorted(xl))
    mesh.AddLine("y", sorted(yl))
    mesh.SmoothMeshLines("x", res)
    mesh.SmoothMeshLines("y", res)
    mesh.SmoothMeshLines("z", res)

    return ports


def run_trial(geometry: str, port_gap_mil: float, run: bool):
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS

    freq = np.linspace(F_MIN, F_MAX, N_FREQ)
    S = np.zeros((N_FREQ, 2, 2), dtype=complex)
    sim_root = _THIS_DIR / "runs" / f"04b_portcal_{geometry}"
    sim_root.mkdir(parents=True, exist_ok=True)

    for p_exc in range(2):
        FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)
        FDTD.SetGaussExcite((F_MAX + F_MIN) / 2, (F_MAX - F_MIN) / 2)
        FDTD.SetBoundaryCond(["MUR"] * 6)
        CSX = ContinuousStructure()
        FDTD.SetCSX(CSX)
        ports = build_single_via(FDTD, CSX, geometry, p_exc, port_gap_mil)

        if not run:
            xml = sim_root / "model.xml"
            CSX.Write2XML(str(xml))
            print(f"  [dry-run] wrote {xml}")
            return None, None

        sim_dir = sim_root / f"excite_{p_exc}"
        FDTD.Run(str(sim_dir), cleanup=True, verbose=0)
        for pm in range(2):
            ports[pm].CalcPort(str(sim_dir), freq, ref_impedance=50)
        a_inc = ports[p_exc].uf_inc
        for pm in range(2):
            S[:, pm, p_exc] = ports[pm].uf_ref / a_inc

    return freq, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=["A", "B"], default="A")
    ap.add_argument("--gap", type=float, default=None,
                    help="port gap in mil for geometry A (default: TMET)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gap = args.gap if args.gap is not None else TMET
    print("=" * 70)
    print(f"Stage 04b: port calibration  (geometry {args.geometry}, "
          f"gap={gap:.3f} mil)")
    print("=" * 70)
    print(f"  via_radius={VIA_RADIUS:.3f}  antipad={ANTIPAD_RADIUS:.3f}  "
          f"TMET={TMET:.3f}  TDIEL={TDIEL:.3f}  eps_r={EPS_R:.3f} (mil)")

    if args.dry_run:
        run_trial(args.geometry, gap, run=False)
        print("\nDry run: inspect model.xml in AppCSXCAD, confirm the port sits "
              "in the\nGAP between the via end and the plane, NOT overlapping "
              "the via body.")
        return

    freq, S = run_trial(args.geometry, gap, run=True)
    db = lambda x: 20 * np.log10(np.abs(x) + 1e-12)
    print("\n  Results (a clean through-via port -> |S21|~0 dB, |S11| deep):")
    print(f"    {'f[GHz]':>8s}  {'|S11|dB':>8s}  {'|S21|dB':>8s}")
    for k in [0, 5, int(N_FREQ*0.14), int(N_FREQ*0.28), int(N_FREQ*0.56)]:
        print(f"    {freq[k]/1e9:8.2f}  {db(S[k,0,0]):8.2f}  {db(S[k,1,0]):8.2f}")

    s11_lo = db(S[0, 0, 0]); s21_lo = db(S[0, 1, 0])
    print("\n  VERDICT:")
    if s21_lo > -1.0 and s11_lo < -10.0:
        print(f"    PASS -- |S21|(low)={s21_lo:.2f} dB ~ 0, "
              f"|S11|(low)={s11_lo:.2f} dB deep.")
        print(f"    Port geometry {args.geometry} is clean. Transplant it into "
              f"stage 04 _add_ports().")
    else:
        print(f"    NOT CLEAN -- |S21|(low)={s21_lo:.2f} dB, "
              f"|S11|(low)={s11_lo:.2f} dB.")
        print(f"    Try the other geometry, or adjust --gap.")

    out = _THIS_DIR / "results" / "04b_portcal"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"portcal_{args.geometry}.npz", freq=freq, S=S)
    print(f"\n  saved {out / ('portcal_' + args.geometry + '.npz')}")


if __name__ == "__main__":
    main()