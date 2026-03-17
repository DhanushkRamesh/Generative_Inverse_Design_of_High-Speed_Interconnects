import os
import CSXCAD
import openEMS

print("This is a sandbox for testing openEMS simulations and CSXCAD geometry creation. You can run this file to quickly iterate on your simulation setup before integrating it into the full dataset pipeline.")

#initialize the 3D space
CSX = CSXCAD.ContinuousStructure()
#define the material
metal = CSX.AddMetal('copper')

#draw a simple via structure for testing
metal.AddBox(priority=1, start=[0, 0, 0], stop=[10, 10, 10])

#save 3D geometry to xml file
xml_path = "sandbox_initial_test.xml"
CSX.Write2XML(xml_path)
print(f"3D geometry saved to {xml_path}")
print("run sandbox_initial_test.xml through openEMS to verify the setup and visualize the structure.")