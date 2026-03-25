import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import CSXCAD
from openEMS import openEMS
from openEMS.ports import LumpedPort

def parse_via_array(filepath):
    """Dynamically reads the [ARRAY] block from via_array.txt"""
    grid = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    in_array = False
    for line in lines:
        line = line.strip()
        if line == '[ARRAY]':
            in_array = True
            continue
        if in_array and line:
            row = line.split()
            if row:
                grid.append(row)
    return grid

def generate_tuhh_geometry(params, array_grid):
    """Builds the 3D FDTD simulation based on TUHH parameters and via grid."""
    print(f"\n🚀 Initializing Shielded Array Simulation for: {params['SIMULATION']}")
    
    Sim_Path = os.path.abspath('Sim_Results_TUHH')
    os.makedirs(Sim_Path, exist_ok=True)

    FDTD = openEMS(EndCriteria=1e-4)
    f_max = 40e9 
    FDTD.SetGaussExcite(f0=f_max/2, fc=f_max/2)
    FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8'])
    
    CSX = CSXCAD.ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    unit = 1e-6 
    mesh.SetDeltaUnit(unit) 
    
    # 1. Extract Geometry Parameters
    layers = int(params['LAYER_AMOUNT'])
    t_diel = params['TDIEL']  
    t_met = params['TMET']    
    via_rad = params['VIA_RADIUS']
    antipad_rad = params['ANTIPAD_RADIUS']
    pitch = params['PITCH']
    eps_r = params['PERMITTIVITY']
    
    cols = len(array_grid[0])
    rows = len(array_grid)
    board_w = cols * pitch + 4 * pitch
    board_l = rows * pitch + 4 * pitch
    
    diel_mat = CSX.AddMaterial('Dielectric', epsilon=eps_r)
    copper = CSX.AddMetal('Copper')
    
    # 2. Map out all via coordinates
    center_x = (cols - 1) / 2.0
    center_y = (rows - 1) / 2.0
    via_locations = []
    s1_pos = None

    for r in range(rows):
        for c in range(cols):
            via_name = array_grid[r][c]
            px = (c - center_x) * pitch
            py = (center_y - r) * pitch 
            via_locations.append({'name': via_name, 'x': px, 'y': py})
            if via_name == 'S1': 
                s1_pos = (px, py)

    if not s1_pos:
        raise ValueError("Could not find 'S1' in the via_array.txt grid!")

    # 3. Build the Stackup
    current_z = 0.0
    z_mesh_lines = [0]
    
    for i in range(layers):
        # Dielectric
        diel_mat.AddBox(priority=1, start=[-board_w/2, -board_l/2, current_z], stop=[board_w/2, board_l/2, current_z + t_diel])
        plane_z_start = current_z + t_diel
        plane_z_stop = plane_z_start + t_met
        
        # Copper Ground Plane
        copper.AddBox(priority=2, start=[-board_w/2, -board_l/2, plane_z_start], stop=[board_w/2, board_l/2, plane_z_stop])
        
        # Antipads ONLY for 'S' and 'P'
        for v in via_locations:
            if v['name'].startswith('S') or v['name'].startswith('P'):
                diel_mat.AddCylinder(priority=3, start=[v['x'], v['y'], plane_z_start - 0.1], stop=[v['x'], v['y'], plane_z_stop + 0.1], radius=antipad_rad)
                             
        current_z = plane_z_stop
        z_mesh_lines.extend([plane_z_start, plane_z_stop])

    total_height = current_z
    z_mesh_lines.append(total_height)
    
    # 4. Drop Vias & Meshing
    x_mesh_lines = []
    y_mesh_lines = []
    for v in via_locations:
        copper.AddCylinder(priority=4, start=[v['x'], v['y'], 0], stop=[v['x'], v['y'], total_height], radius=via_rad)
        x_mesh_lines.extend([v['x']-antipad_rad, v['x']-via_rad, v['x'], v['x']+via_rad, v['x']+antipad_rad])
        y_mesh_lines.extend([v['y']-antipad_rad, v['y']-via_rad, v['y'], v['y']+via_rad, v['y']+antipad_rad])

    pad = 30.0 
    mesh.AddLine('x', [-board_w/2] + x_mesh_lines + [board_w/2])
    mesh.AddLine('y', [-board_l/2] + y_mesh_lines + [board_l/2])
    mesh.AddLine('z', [-pad] + z_mesh_lines + [total_height+pad])
    
    mesh.SmoothMeshLines('x', 3.0, ratio=1.3)
    mesh.SmoothMeshLines('y', 3.0, ratio=1.3)
    mesh.SmoothMeshLines('z', 3.0, ratio=1.3)

    # 5. Add Ports to S1
    port1 = LumpedPort(CSX, priority=10, port_nr=1,
                       start=[s1_pos[0] - antipad_rad, s1_pos[1] - via_rad/2, total_height], 
                       stop=[s1_pos[0] - via_rad, s1_pos[1] + via_rad/2, total_height], 
                       dir='x', exc_dir='x', R=50, excite=1)

    port9 = LumpedPort(CSX, priority=10, port_nr=9,
                       start=[s1_pos[0] - antipad_rad, s1_pos[1] - via_rad/2, t_diel], 
                       stop=[s1_pos[0] - via_rad, s1_pos[1] + via_rad/2, t_diel], 
                       dir='x', exc_dir='x', R=50)

    # 6. Run Engine
    print(f"FDTD Setup Complete. Starting Solver (this will take a while)...")
    FDTD.Run(Sim_Path, cleanup=True)
    
    # 7. Post-Processing S-Parameters
    f = np.linspace(1e9, f_max, 501)
    port1.CalcPort(Sim_Path, f)
    port9.CalcPort(Sim_Path, f)
    
    s11 = port1.uf_ref / port1.uf_inc
    s91 = port9.uf_ref / port1.uf_inc
    
    plt.figure(figsize=(10, 5))
    plt.plot(f/1e9, 20*np.log10(np.abs(s11)), label='openEMS S11 (Return Loss)')
    plt.plot(f/1e9, 20*np.log10(np.abs(s91)), label='openEMS S91 (Insertion Loss)')
    plt.title(f'20-Via Shielded Array: {params["SIMULATION"]}')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Magnitude (dB)')
    plt.ylim([-60, 5])
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    target_sim = 'sim_pkg_3146'
    
    # Safely find the project root directory regardless of where you run the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    base_dir = os.path.join(project_root, "data", "raw", "Universal-Diff-SI-Array")
    csv_path = os.path.join(base_dir, "parameter.csv")
    array_txt_path = os.path.join(base_dir, "variation", target_sim, "via_array.txt")
    
    print(f"Loading parameters for {target_sim} from CSV...")
    df = pd.read_csv(csv_path)
    
    sim_row = df[df['SIMULATION'] == target_sim]
    if sim_row.empty:
        raise ValueError(f"Simulation {target_sim} not found in CSV!")
    
    test_params = sim_row.iloc[0].to_dict()
    
    print(f"Parsing 2D via grid from {array_txt_path}...")
    array_grid = parse_via_array(array_txt_path)
    
    generate_tuhh_geometry(test_params, array_grid)