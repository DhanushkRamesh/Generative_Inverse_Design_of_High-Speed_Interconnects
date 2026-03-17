import os
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0

def simulate_sim_pkg_0032():
    # INITIALIZE WORKSPACE
    sim_path = os.path.join(os.getcwd(), "openEMS_Validation")
    os.makedirs(sim_path, exist_ok=True)
    
    CSX = ContinuousStructure()
    FDTD = openEMS(EndCriteria=1e-4) # Stops when energy drops to -40dB
    
    # CRITICAL PYTHON FIX: You must link the geometry to the solver
    FDTD.SetCSX(CSX)

    # 2. PARAMETERS (From CSV for sim_pkg_0032)
    mil2m = 25.4e-6 # Converting TUHH mils to meters
    f_max = 100e9   # 100 GHz from your .pt file
    
    via_radius = 11.03 * mil2m
    antipad_radius = 23.08 * mil2m
    pitch = 60.34 * mil2m
    t_met = 4.09 * mil2m
    t_diel = 10.74 * mil2m
    
    # 3. MATERIALS (Using kappa for conductivity)
    copper = CSX.AddMaterial('Copper', kappa=51277479.75)
    fr4 = CSX.AddMaterial('FR4', epsilon=4.65)
    air = CSX.AddMaterial('Air', epsilon=1.0)

    # 4. BUILD THE STACKUP (12-layer logic)
    stackup = ['G', 'S', 'G', 'S', 'G', 'P', 'P', 'G', 'S', 'G', 'S', 'G']
    total_z = (12 * t_met) + (11 * t_diel)
    
    fr4.AddBox([0, 0, 0], [5 * pitch, 4 * pitch, total_z], priority=0)
    
    current_z = 0
    for layer_type in stackup:
        if layer_type in ['G', 'P']: 
            copper.AddBox([0, 0, current_z], [5 * pitch, 4 * pitch, current_z + t_met], priority=10)
        current_z += (t_met + t_diel)

    # 5. BUILD THE VIA ARRAY
    s1_x, s1_y = 2 * pitch, 2 * pitch
    s2_x, s2_y = 2 * pitch, 1 * pitch
    
    air.AddCylinder([s1_x, s1_y, 0], [s1_x, s1_y, total_z], antipad_radius, priority=20)
    air.AddCylinder([s2_x, s2_y, 0], [s2_x, s2_y, total_z], antipad_radius, priority=20)
    
    copper.AddCylinder([s1_x, s1_y, 0], [s1_x, s1_y, total_z], via_radius, priority=30)
    copper.AddCylinder([s2_x, s2_y, 0], [s2_x, s2_y, total_z], via_radius, priority=30)

    # 6. EXCITATION & PORTS
    # Center frequency (f0) = 50 GHz, Bandwidth (fc) = 50 GHz -> Sweeps 0 to 100 GHz
    f0 = f_max / 2.0
    fc = f_max / 2.0
    FDTD.SetGaussExcite(f0, fc)
    
    # Add a Lumped Port (port_nr=1, R=50, start, stop, dir='z', excite=1.0, priority=50)
    # Inject signal on S1 between Z=0 and Z=t_met (bottom ground plane)
    port1 = FDTD.AddLumpedPort(1, 50, [s1_x, s1_y, 0], [s1_x, s1_y, t_met], 'z', 1.0, priority=50)

    # 7. THE FDTD MESH
    lambda_min = C0 / (f_max * np.sqrt(4.65))
    max_res = lambda_min / 15
    
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(1) # Base unit is meters
    
    x_lines = [0, s1_x - via_radius, s1_x, s1_x + via_radius, 5 * pitch]
    y_lines = [0, s2_y - via_radius, s2_y, s2_y + via_radius, s1_y - via_radius, s1_y, s1_y + via_radius, 4 * pitch]
    z_lines = [0, t_met, total_z]
    
    mesh.AddLine('x', x_lines)
    mesh.AddLine('y', y_lines)
    mesh.AddLine('z', z_lines)
    
    # Calculate the remaining mesh lines automatically
    mesh.SmoothMeshLines('all', max_res, 1.4)

    # 8. BOUNDARY CONDITIONS
    FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8'])

    # 9. RUN THE SOLVER
    xml_file = 'sim_pkg_0032.xml'
    xml_path = os.path.join(sim_path, xml_file)
    CSX.Write2XML(xml_path)
    
    print(f"Geometry successfully generated at: {xml_path}")
    print("Starting FDTD Solver. This will take some time...")
    
    # Run openEMS
    FDTD.Run(sim_path, cleanup=True)
    print("Simulation Complete!")

if __name__ == "__main__":
    simulate_sim_pkg_0032()