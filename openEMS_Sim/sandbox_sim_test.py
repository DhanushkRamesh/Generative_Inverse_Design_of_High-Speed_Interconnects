import os
import CSXCAD
from openEMS import openEMS
import openEMS.ports as ports

def run_simple_engine_test():
    Sim_Path = os.path.abspath('Sim_Results_Engine_Test')
    os.makedirs(Sim_Path, exist_ok=True)

    # Initialize FDTD Engine
    # EndCriteria=1e-3 means the simulation stops when the energy decays to 0.1%
    FDTD = openEMS(EndCriteria=1e-3)
    
    # Excite with a Gaussian pulse centered at 1 GHz, with a 1 GHz bandwidth
    FDTD.SetGaussExcite(f0=1e9, fc=1e9) 

    # Initialize 3D Space and Boundaries
    CSX = CSXCAD.ContinuousStructure()
    FDTD.SetCSX(CSX)
    # Boundary Conditions: 0 = Perfectly Electric Conducting (PEC) walls
    FDTD.SetBoundaryCond([0, 0, 0, 0, 0, 0]) 

    # Create a Basic Mesh (Grid)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(1e-3) # Dimensions in mm
    # Adding a very coarse 3x3x3 grid just so the engine has math to compute
    mesh.AddLine('x', [-10, 0, 10])
    mesh.AddLine('y', [-10, 0, 10])
    mesh.AddLine('z', [-10, 0, 10])

    # Add a Excitation Port
    # A Lumped Port acts like an ideal voltage source + 50 ohm resistor
    port = ports.LumpedPort(CSX, priority=10, port_nr=1,
                            start=[-5, 0, 0], stop=[5, 0, 0],
                            dir='x', exc_dir='x', R=50, excite=True)

    # Run the Simulation!
    xml_file = os.path.join(Sim_Path, 'engine_test.xml')
    CSX.Write2XML(xml_file)
    
    print(" Firing up the openEMS FDTD Engine...")
    # This actually runs the C++ solver!
    FDTD.Run(Sim_Path, cleanup=True)
    
    print(" openEMS Engine computed the FDTD math successfully!")

if __name__ == "__main__":
    run_simple_engine_test()