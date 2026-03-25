import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import CSXCAD
from openEMS import openEMS
from openEMS.ports import LumpedPort

def parse_via_array(filepath):
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
            if row: grid.append(row)
    return grid

def run_simulation(params, array_grid, port_config="narrow"):
    sim_id = params['SIMULATION']
    print(f"\n🧪 Testing Configuration: {port_config.upper()}")
    
    # Setup unique path for each test run
    Sim_Path = os.path.abspath(f'Sim_Study_{port_config}')
    os.makedirs(Sim_Path, exist_ok=True)

    FDTD = openEMS(EndCriteria=1e-4)
    f_max = 60e9 # 60GHz is enough to see the primary wiggles
    FDTD.SetGaussExcite(f0=f_max/2, fc=f_max/2)
    FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8'])
    
    CSX = CSXCAD.ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    unit = 1e-6 
    mesh.SetDeltaUnit(unit) 
    
    # 1. Geometry Constants
    layers = int(params['LAYER_AMOUNT'])
    t_diel, t_met = params['TDIEL'], params['TMET']    
    via_rad, antipad_rad = params['VIA_RADIUS'], params['ANTIPAD_RADIUS']
    pitch, eps_r = params['PITCH'], params['PERMITTIVITY']
    cols, rows = len(array_grid[0]), len(array_grid)
    board_w, board_l = cols * pitch + 2 * pitch, rows * pitch + 2 * pitch
    
    diel_mat = CSX.AddMaterial('Dielectric', epsilon=eps_r)
    copper = CSX.AddMetal('Copper')
    
    # 2. Coordinates
    center_x, center_y = (cols - 1) / 2.0, (rows - 1) / 2.0
    via_locations = []
    s1_pos, s2_pos = None, None
    for r in range(rows):
        for c in range(cols):
            v_name = array_grid[r][c]
            px, py = (c - center_x) * pitch, (center_y - r) * pitch 
            via_locations.append({'name': v_name, 'x': px, 'y': py})
            if v_name == 'S1': s1_pos = (px, py)
            if v_name == 'S2': s2_pos = (px, py)

    # 3. Stackup & Vias
    current_z = 0.0
    z_mesh_lines = [0]
    port_z_top = t_diel + t_met
    port_z_bottom = (t_diel + t_met) * (layers - 2) # Dynamically find second-to-last S layer
    
    for i in range(layers):
        diel_mat.AddBox(priority=1, start=[-board_w/2, -board_l/2, current_z], stop=[board_w/2, board_l/2, current_z + t_diel])
        p_start, p_stop = current_z + t_diel, current_z + t_diel + t_met
        copper.AddBox(priority=2, start=[-board_w/2, -board_l/2, p_start], stop=[board_w/2, board_l/2, p_stop])
        for v in via_locations:
            if v['name'].startswith('S') or v['name'].startswith('P'):
                diel_mat.AddCylinder(priority=3, start=[v['x'], v['y'], p_start-0.5], stop=[v['x'], v['y'], p_stop+0.5], radius=antipad_rad)
        current_z = p_stop
        z_mesh_lines.extend([p_start, p_stop])

    total_height = current_z
    x_mesh_lines, y_mesh_lines = [], []
    for v in via_locations:
        copper.AddCylinder(priority=4, start=[v['x'], v['y'], 0], stop=[v['x'], v['y'], total_height], radius=via_rad)
        x_mesh_lines.extend([v['x']-antipad_rad, v['x'], v['x']+antipad_rad])
        y_mesh_lines.extend([v['y']-antipad_rad, v['y'], v['y']+antipad_rad])

    mesh.AddLine('x', [-board_w/2, board_w/2] + x_mesh_lines)
    mesh.AddLine('y', [-board_l/2, board_l/2] + y_mesh_lines)
    mesh.AddLine('z', [-20, total_height+20] + z_mesh_lines)
    mesh.SmoothMeshLines('x', 5.0, ratio=1.3); mesh.SmoothMeshLines('y', 5.0, ratio=1.3); mesh.SmoothMeshLines('z', 3.0, ratio=1.3)

    # 4. Sensitivity Logic: Port Width
    # We change how much of the via face the port 'sees'
    if port_config == "narrow":
        w = via_rad * 0.5
    elif port_config == "wide":
        w = via_rad * 1.5
    else: # "sheet"
        w = (antipad_rad - via_rad) * 2.0 

    # Ports
    p1 = LumpedPort(CSX, port_nr=1, start=[s1_pos[0]-antipad_rad, s1_pos[1]-w/2, port_z_top], stop=[s1_pos[0]-via_rad, s1_pos[1]+w/2, port_z_top], dir='x', exc_dir='x', R=50, excite=1)
    p2 = LumpedPort(CSX, port_nr=2, start=[s2_pos[0]+antipad_rad, s2_pos[1]-w/2, port_z_top], stop=[s2_pos[0]+via_rad, s2_pos[1]+w/2, port_z_top], dir='x', exc_dir='x', R=50, excite=-1)
    p9 = LumpedPort(CSX, port_nr=9, start=[s1_pos[0]-antipad_rad, s1_pos[1]-w/2, port_z_bottom], stop=[s1_pos[0]-via_rad, s1_pos[1]+w/2, port_z_bottom], dir='x', exc_dir='x', R=50)
    p10 = LumpedPort(CSX, port_nr=10, start=[s2_pos[0]+antipad_rad, s2_pos[1]-w/2, port_z_bottom], stop=[s2_pos[0]+via_rad, s2_pos[1]+w/2, port_z_bottom], dir='x', exc_dir='x', R=50)

    # 5. Run & Post-Process
    FDTD.Run(Sim_Path, cleanup=True)
    f = np.linspace(1e9, f_max, 301)
    for p in [p1, p2, p9, p10]: p.CalcPort(Sim_Path, f)
    sdd11 = (p1.uf_ref - p2.uf_ref) / (p1.uf_inc - p2.uf_inc)
    sdd21 = (p9.uf_ref - p10.uf_ref) / (p1.uf_inc - p2.uf_inc)
    
    return f, sdd11, sdd21

if __name__ == "__main__":
    target_sim = 'sim_pkg_3146'
    
    # Path Setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    base_dir = os.path.join(project_root, "data", "raw", "Universal-Diff-SI-Array")
    
    df = pd.read_csv(os.path.join(base_dir, "parameter.csv"))
    test_params = df[df['SIMULATION'] == target_sim].iloc[0].to_dict()
    array_grid = parse_via_array(os.path.join(base_dir, "variation", target_sim, "via_array.txt"))

    # Execute Study
    configs = ["narrow", "wide", "sheet"]
    results = {}
    for c in configs:
        f, s11, s21 = run_simulation(test_params, array_grid, port_config=c)
        results[c] = (s11, s21)

    # Plot Comparison
    plt.figure(figsize=(12, 8))
    for c, (s11, s21) in results.items():
        plt.plot(f/1e9, 20*np.log10(np.abs(s11)), label=f'Oracle Sdd11 ({c})', linewidth=2)
    
    plt.title(f"Sensitivity Analysis: Finding the Wiggles for {target_sim}")
    plt.xlabel('Freq (GHz)'); plt.ylabel('Mag (dB)'); plt.ylim([-60, 5])
    plt.legend(); plt.grid(True); plt.show()