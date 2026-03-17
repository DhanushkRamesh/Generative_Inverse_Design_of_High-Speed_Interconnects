import os, numpy as np
import matplotlib.pyplot as plt
from openEMS import openEMS
from CSXCAD import ContinuousStructure
from openEMS.ports import MSLPort

def run_msl_test():
    Sim_Path = os.path.abspath('Sim_Results_MSL')
    if not os.path.exists(Sim_Path):
        os.makedirs(Sim_Path)

    FDTD = openEMS(EndCriteria=1e-4)
    f_max = 10e9
    FDTD.SetGaussExcite(f0=f_max/2, fc=f_max/2) 
    
    # FIX 1: Use Perfectly Matched Layers (PML) to absorb energy and stop reflections
    FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PEC', 'PML_8'])
    
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(1e-3) # mm
    
    unit = 1e-3 # mm
    sub_h = 1.6
    sub_er = 4.4
    trace_w = 3.0
    line_l = 50.0
    cu_thick = 0.035 # FIX 2: 35 microns real copper thickness
    
    # Add Air Padding (10mm) to keep ports away from the walls
    pad = 10.0 
    
    # Define Geometry
    substrate = CSX.AddMaterial('FR4', epsilon=sub_er)
    substrate.AddBox(priority=0, start=[-10, -line_l/2, 0], stop=[10, line_l/2, sub_h])
    
    gnd = CSX.AddMetal('GND')
    gnd.AddBox(priority=1, start=[-10, -line_l/2, 0], stop=[10, line_l/2, 0])
    
    trace = CSX.AddMetal('Trace')
    # Use real thickness so openEMS doesn't ignore the primitive!
    trace.AddBox(priority=2, start=[-trace_w/2, -line_l/2, sub_h], stop=[trace_w/2, line_l/2, sub_h + cu_thick])
    
    # MESHING
    # Include the padding in the mesh limits
    xs = [-10-pad, -10, -trace_w/2, 0, trace_w/2, 10, 10+pad]
    ys = [-line_l/2-pad, -line_l/2, 0, line_l/2, line_l/2+pad]
    zs = [0, sub_h, sub_h + cu_thick, sub_h + pad] 
    
    mesh.AddLine('x', xs)
    mesh.AddLine('y', ys)
    mesh.AddLine('z', zs)
    
    max_res = 3e8 / f_max / sub_er**0.5 / 10 / unit 
    mesh.SmoothMeshLines('x', max_res, ratio=1.4)
    mesh.SmoothMeshLines('y', max_res, ratio=1.4)
    mesh.SmoothMeshLines('z', max_res/2, ratio=1.4)
    
    port_len = 4.0 
    
    port1 = MSLPort(CSX, port_nr=1, 
                metal_prop=trace, ground_prop=gnd, 
                start=[-trace_w/2, -line_l/2, sub_h], 
                stop=[trace_w/2, -line_l/2 + port_len, 0], 
                prop_dir='y', exc_dir='z',
                excite=1)
                
    port2 = MSLPort(CSX, port_nr=2, 
                metal_prop=trace, ground_prop=gnd, 
                start=[-trace_w/2, line_l/2, sub_h], 
                stop=[trace_w/2, line_l/2 - port_len, 0], 
                prop_dir='y', exc_dir='z')
    
    print("Starting FDTD Solver. Watch the Energy drop!")
    FDTD.Run(Sim_Path, cleanup=True)
    
    f = np.linspace(1e9, f_max, 501)
    port1.CalcPort(Sim_Path, f)
    port2.CalcPort(Sim_Path, f)
    
    s11 = port1.uf_ref / port1.uf_inc
    s21 = port2.uf_ref / port1.uf_inc
    
    plt.figure(figsize=(10, 5))
    plt.plot(f/1e9, 20*np.log10(np.abs(s11)), label='S11 (Return Loss)')
    plt.plot(f/1e9, 20*np.log10(np.abs(s21)), label='S21 (Insertion Loss)')
    plt.title('Microstrip Simulation Results (0-10 GHz)')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Magnitude (dB)')
    plt.ylim([-50, 5])
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_msl_test()