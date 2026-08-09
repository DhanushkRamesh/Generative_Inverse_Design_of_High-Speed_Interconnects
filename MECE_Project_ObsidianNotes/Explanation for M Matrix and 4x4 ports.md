### **1. Clarification of M matrix used and where it was obtained from**

The M matrix used and source where it was obtained: So while searching google scholar to get papers related to mixed mode s-parameter derivations, I found the paper "[A. Ferrero and M. Pirola, "Generalized mixed-mode S-parameters," in _IEEE Transactions on Microwave Theory and Techniques_](https://ieeexplore.ieee.org/abstract/document/1573844)[, vol. 54, no. 1, pp. 458-463, Jan. 2006, doi: 10.1109/TMTT.2005.860497.](https://ieeexplore.ieee.org/abstract/document/1573844) " as the top search - and their work provides a transformation matrix specifically designed for asymmetrical ports. However, the official TUHH dataset documentation (for both Array and Link dataset) explicitly mentions that "Two signal vias are always placed adjacent to each other to enable differential signaling through post-processing" and in Link dataset it was also mentioned "In this dataset always two identical arrays are placed with a distance of [LENGTH] between each other". Also, the dataset utilizes uniform $50 \Omega$ reference impedances across all ports. Since they have identical physical dimensions I used the matrix defined in the paper " [D. Bockelman and W. R. Einsenstadt, “Combined differential and common-mode scattering parameters: Theory and simulation,” IEEE Trans. Microw. Theory Tech., vol. 43, no. 7, pp. 1530–1539, Jul. 1995](https://ieeexplore.ieee.org/document/392911)" - I found this paper was referred to in Ferrero et al. and I used the same math in the code. 

In Bockelman et al. they define the mixed-mode normalized power waves as scalar equations,

eg. differential and common mode waves at port 1 are,

![[m matrix logic.png]]

so assembling these port equations into a linear algebra system, the orthogonal transformation matrix is obtained. ​  

$$M = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & -1 & 0 & 0 \\ 0 & 0 & 1 & -1 \\ 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \end{bmatrix}$$
Row 1 - D1 (Differential Mode1), Row 2 - D2 (Differential Mode 2), Row 3 -C1 (Common Mode 1), Row 4 -C2 (Common Mode 2)

The foundational paper (Bockelman et al.) is where I obtained the M matrix.

The code logic of the mixed-mode conversion function is available in the following link in git --> [Generative_Inverse_Design_of_High-Speed_Interconnects/src/utils/physics_utils.py](https://github.com/DhanushkRamesh/Generative_Inverse_Design_of_High-Speed_Interconnects/blob/feat/data-pipeline/src/utils/physics_utils.py)

### 2. Confirmation on selecting the 4 * 4 ports as those ports that have the greatest linkages between them (i.e. least attenuation)

Yes, I confirm that the 4 * 4 ports selected are the ones with the greatest linkages and least attenuation. This is guaranteed by the find_ports.py script that I wrote to test the exact physical behavior. The script injects a signal to port 1 and profiles the insertion loss (S21) across every other port in the simulation. The script specifically looks for the path that stays better than -5dB at near -DC freq. therefore, it computationally filters the path with least attenuation to isolate from the attenuated crosstalk paths. 

I ran this script across multiple variations in the dataset (8-port, 16-port, 80-port) and the it was validated from the script that the port with the good linkage perfectly aligns with the standard Near-End/Far-End block ordering in every single case - meaning, the strongest receiver for port 1 is always exactly at N/2 + 1. Because of this alignment, the main parser uses (N/2) math to  dynamically slice the matrix. This guarantees that we are always extracting the code channels with the strongest linkage for our 4 x 4 matrix tensor. 

The code to find the ports can be referred from --> [Generative_Inverse_Design_of_High-Speed_Interconnects/src/data/find_ports.py](https://github.com/DhanushkRamesh/Generative_Inverse_Design_of_High-Speed_Interconnects/blob/feat/data-pipeline/src/data/find_ports.py)


