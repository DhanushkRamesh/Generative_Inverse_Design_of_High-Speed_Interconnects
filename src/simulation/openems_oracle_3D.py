import os
import CSXCAD

def generate_tuhh_geometry(params, output_xml="tuhh_geometry.xml"):
    """
    Translates TUHH Dataset parameters into a 3D CSXCAD model.
    """
    print(f"Building 3D Model for: {params['SIMULATION']}")
    
    # Initialize 3D Space & Scale
    # We set unit to 1e-6 because these are package-level micrometers (um)
    CSX = CSXCAD.ContinuousStructure()
    unit = 1e-6 
    
    # Extract Geometry Parameters
    layers = int(params['LAYER_AMOUNT'])
    t_diel = params['TDIEL']  
    t_met = params['TMET']    
    via_rad = params['VIA_RADIUS']
    antipad_rad = params['ANTIPAD_RADIUS']
    pitch = params['PITCH']
    eps_r = params['PERMITTIVITY']
    
    # Board Size (making it large enough to hold the central differential pair)
    board_w = pitch * 4
    board_l = pitch * 4
    
    # Materials
    diel_mat = CSX.AddMaterial('Dielectric', epsilon=eps_r)
    copper = CSX.AddMetal('Copper')
    
    # Build the Stackup (Looping through the layers)
    current_z = 0.0
    
    for i in range(layers):
        # A. Draw Dielectric Layer (Priority 1)
        diel_mat.AddBox(priority=1, 
                        start=[-board_w/2, -board_l/2, current_z], 
                        stop=[board_w/2, board_l/2, current_z + t_diel])
        
        # B. Draw Copper Ground Plane above the dielectric (Priority 2)
        # (We skip the top ground plane if it's an exposed microstrip layer, 
        # but for stripline/vias, TUHH usually has metal between every layer)
        plane_z_start = current_z + t_diel
        plane_z_stop = plane_z_start + t_met
        copper.AddBox(priority=2, 
                      start=[-board_w/2, -board_l/2, plane_z_start], 
                      stop=[board_w/2, board_l/2, plane_z_stop])
        
        # C. Drill Antipads through the Copper (Priority 3 - Overwrites copper with dielectric)
        # Left Antipad
        diel_mat.AddCylinder(priority=3, 
                             start=[-pitch/2, 0, plane_z_start - 0.1], 
                             stop=[-pitch/2, 0, plane_z_stop + 0.1], 
                             radius=antipad_rad)
        # Right Antipad
        diel_mat.AddCylinder(priority=3, 
                             start=[pitch/2, 0, plane_z_start - 0.1], 
                             stop=[pitch/2, 0, plane_z_stop + 0.1], 
                             radius=antipad_rad)
                             
        # Move up to the next layer
        current_z = plane_z_stop

    # Drop the Via Barrels completely through the board (Priority 4)
    total_height = current_z
    
    # Left Via
    copper.AddCylinder(priority=4, 
                       start=[-pitch/2, 0, 0], stop=[-pitch/2, 0, total_height], 
                       radius=via_rad)
    # Right Via
    copper.AddCylinder(priority=4, 
                       start=[pitch/2, 0, 0], stop=[pitch/2, 0, total_height], 
                       radius=via_rad)

    # Save for AppCSXCAD Visualization
    CSX.Write2XML(output_xml)
    print(f"Success! 3D model saved to {output_xml}")

if __name__ == "__main__":
    # Parameters for SIM_ID 3146
    test_params = {
        'SIMULATION': 'sim_pkg_3146',
        'LAYER_AMOUNT': 8.0,
        'TDIEL': 20.909839,
        'TMET': 2.788379,
        'VIA_RADIUS': 7.753086,
        'ANTIPAD_RADIUS': 21.684783,
        'PITCH': 46.725307,
        'PERMITTIVITY': 3.502265
    }
    
    generate_tuhh_geometry(test_params)