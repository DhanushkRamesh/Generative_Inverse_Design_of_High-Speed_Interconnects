#             **Data Analysis and Parsing for TUHH SI/PI Database**

## **1 Origin of Dataset**

I will be using the SI/PI database developed by Institut für Theoretische Elektrotechnik at the Hamburg University of Technology (TUHH). My model here specifically relies on the Universal Differential SI datasets - " Universal-Diff-SI-Array" and " Universal-Diff-SI-Link" datasets. These datasets provides simulation based data for high-speed differential interconnects on PCBs'.

 - **Differential Array Dataset**: This dataset focuses on single via array configurations where the no. of vias in x and y directions are varied. The array consists of signal, ground, and power vias, with two signal vias always placed adjacently to enable differential signaling.
 
 - **Differential Link Dataset**: In this dataset, two identical vias are placed at a variable length from each other and are connected using multiconductor transmission lines (MTL). This introduces trace-specific parameters such as line width. 

The Electromagnetic (EM) simulations for these structures were performed using a physics-based simulation tool. The resulting S-Parameters capture the wideband frequency spectrum from 0.25 GHz  to 100 GHz, discretized into 400 linearly spaced frequency points.

## **2 Exploratory Data Analysis**

An initial exploratory data analysis was performed on the raw parameter.csv file for both Array dataset and Link dataset using python (Pandas/NumPy). This phase was used to assess data cleanliness, identify feature distributions, and also to check if there are any bottlenecks in the data that can affect the Deep Learning network. 

### **2.1 Dataset Integrity and Feature Space (Universal-Diff-SI-Array)**

The Array dataset showed a highly structured dataset consisting of a total of 1916 independent simulations. The parameter file contained 16 columns where two columns were identifiers (SIM_ID, SIMULATION) and rest of the 14 columns were continuous physical features ranging from geometric dimensions to material properties.
***************************

**DataFrame shape**: (1916, 16) 
**DataFrame columns**: ['SIM_ID', 'PERMITTIVITY', 'CONDUCTIVITY', 'LOSSTANGENT', 'TDIEL', 'TMET', 'LAYER_AMOUNT', 'VIAS_X_AMOUNT', 'VIAS_Y_AMOUNT', 'VIA_RADIUS', 'SIMULATION', 'ANTIPAD_RADIUS', 'PITCH', 'SIGNAL_AMOUNT', 'GROUND_AMOUNT', 'POWER_AMOUNT'] 
******************************
### **2.2 Dataset Integrity and Feature Space (Universal-Diff-SI-Link)**

The Link dataset was analyzed using the similar python notebook to check for the data integrity and topologies and it also had a highly structured dataset consisting of 1073 indivudual simulations. The parameter file for Link Dataset consisted of 18 columns in total with 2 identifiers (SIM_ID, SIMULATION), and link dataset had additional two geometric features with a total of 16 columns with continuous physical features of geometric dimensions to material properties.
*******************
**DataFrame shape**: (1073, 18)
**DataFrame columns**: ['SIM_ID', 'PERMITTIVITY', 'CONDUCTIVITY', 'LOSTANGENT', 'TDIEL', 'TMET', 'LAYER_AMOUNT', 'VIAS_X_AMOUNT', 'VIAS_Y_AMOUNT', 'VIA_RADIUS', 'LENGTH', 'SL_WIDTH', 'SIMULATION', 'ANTIPAD_RADIUS', 'PITCH', 'SIGNAL_AMOUNT', 'GROUND_AMOUNT', 'POWER_AMOUNT']
********************************************

When Checked for data integrity using Python (pandas and NumPy) confirmed 0 duplicated and 0 missing values (NaN) in the dataset. This indicates that TUHH successfully generated a fully disjoint and complete parameter space. A minor analogical mistake was found in the Link dataset where the column for "Losstangent" was misspelled as "Lostangent", this was identified during Initial EDA and was corrected during the data parsing. 

### **2.3 Scaling Disparity**

During the initial EDA, there was found to be an extreme variance in the feature magnitudes. There was a severe imbalance across the physical parameters as below,
***********************
- **Micro-Scale Features**: The dielectric Losstangent has values as low as 0.000155 (i.e 1.55 x 10^-4)
- **Macro Scale Features**: Trace Length in the Link dataset expanded to 20,000 mils. and Conductivity had values as high as 57986261.822743 (5.79 x 10^7)
*********************************
Feeding this raw data to the neural network may destroy the optimizer and the model may crash as there will be a gradient explosion or vanishing gradients when kept the raw data with high variance. So A Logarithmic Scaling for the material properties and Z-Score Normalization for the feature set were carried during the parsing. 

### **2.4 Variance in Number of Ports**

I wrote a Python script to find the number of ports and its respective simulations and found that there were different number of ports during the simulations and it varied from 4 ports to 80 ports (in Link dataset). This was one of the major finding during the EDA and if this data is fed to the tensor in neural network it will crash resulting in dimensionality mismatches. So to input to have a fixed dimension structure of the data fed to the tensor - all the simulations were sliced to a 4 port (4x4 core matrix with differential return loss and differential Insertion Loss taken as core). This core matrix represents the primary differential transmitter pair and corresponding receiver pair.
***********************
Found 1916 simulation folders in variations.
--------------------------------------------------
Port count summary:
4 ports: 4 simulations
8 ports: 84 simulations
12 ports: 173 simulations
16 ports: 258 simulations
20 ports: 274 simulations
24 ports: 292 simulations
28 ports: 260 simulations
32 ports: 175 simulations
36 ports: 146 simulations
40 ports: 125 simulations
44 ports: 43 simulations
48 ports: 50 simulations
52 ports: 20 simulations
56 ports: 12 simulations
--------------------------------------------------
Analyzing raw ports for dataset: Universal-Diff-SI-Link
Found 1073 simulation folders in variations.
--------------------------------------------------
Port count summary:
4 ports: 1 simulations
8 ports: 34 simulations
12 ports: 100 simulations
16 ports: 144 simulations
20 ports: 187 simulations
24 ports: 166 simulations
28 ports: 120 simulations
32 ports: 93 simulations
40 ports: 70 simulations
48 ports: 59 simulations
56 ports: 39 simulations
64 ports: 41 simulations
72 ports: 7 simulations
80 ports: 12 simulations
-------------------------------------------
*****************************

- **Finding the right ports for slicing the 4x4 matrix**: To ensure that the correct ports were extracted across all topologies, a verification script (https://github.com/DhanushkRamesh/Generative_Inverse_Design_of_High-Speed_Interconnects/blob/feat/data-pipeline/src/data/find_ports.py) was developed using scikit-rf library. This script operates on the principle of Insertion Loss (S21) Profiling. 
- For the given N-port simulation, the script isolates the signal injected port 1 and plots the magnitude of the signal received at every other port in the system. The physical through-path can be computationally isolated because the copper trace connecting the Tx and Rx experiences significantly less attenuation than the cross talk paths. So, the script applies a logic where any signal path with an attenuation better than -5 dB at near DC frequencies is classified as the primary through-path.
- By executing this search - the script validated that for any topology if a transmitter is located at port 1, its corresponding receiver is always located at port (N/2) + 1.  This is supported by the "Near End / Far End Block Ordering Convention" a standard that is used by commercial EM solvers such as Ansys HFSS [1]  [2]. In High speed digital interconnect simulations the EM solvers group all input ports on the Near End plane - from port 1 to port N/2, and all the output ports on the far end from port (N/2) + 1, to port N.
- This logic was used in the parsing script where it is programmed to dynamically calculate the halfway point and slice the core tensor using the exact indices. By implementing this the entire dataset  was guaranteed to feed the same physical signal paths (Differential Return Loss and Differential Insertion Loss).
![[Pasted image 20260324230854.png|584]]
![[Pasted image 20260324230936.png]]
![[Pasted image 20260324230954.png]]

### **2.5 Topology Complexity**

As seem from the number of ports - it was also evident that the AI architecture would rely on global structural parameters (eg. NUM_PORTS, and LENGTH) to intelligently constraint the physics of varying interconnect densities. This was evident from the initial EDA where the via arrays varied form minimal configurations up to a highly dense 7x7 grids containing up to 28 signal vias and 32 ground vias. 
## **3 Data Pipeline (Parsing the raw data to tensor ready data)**

Source: https://github.com/DhanushkRamesh/Generative_Inverse_Design_of_High-Speed_Interconnects/blob/feat/data-pipeline/src/data/parse_touchstone_array.py , https://github.com/DhanushkRamesh/Generative_Inverse_Design_of_High-Speed_Interconnects/blob/feat/data-pipeline/src/data/parse_touchstone_link.py

Even though the raw touchstone files and parameter.csv files provide clean and accurate data, the raw files still needs mathematical conditioning before they can be ingested to the Deep Learning Model. I wrote a Python script to develop a parsing pipeline to ingest the data, enforce electromagnetic constraints (passivity, mixed - mode conversion), to have Python-ready tensors.

### **3.1 Feature Normalization and Logarithmic Scaling**

The extreme variance identified during the initial EDA are addressed by applying selective logarithmic scaling to the material properties. The parameters with high variance like Conductivity and Losstangent are transformed using a base-10 logarithm (np.log10) in order to compress the variables into a linear subspace. This is done to prevent gradient explosion during back propagation in the neural network. Following this, the entire features undergo Z-Score Normalization, forcing a mean of 0 and standard deviation of 1 across all inputs. 

### **3.2 Frequency Interpolation and DC Extrapolation**

According to the TUHH dataset documentation, the raw s-parameter simulations provide 400 linearly spaced frequency points from 250 MHz to 100 GHz. However, a steady state baseline at 0Hz is very critical as the neural network and the Time-Domain Reflectometry (TDR) relies on this. 

The scikit-rf library is utilized to resolve this in the parser to interpolate and extrapolate the data onto a standardized axis of 0 to 100 GHz with 401 linearly spaced datapoints each with 0.25 GHz steps. The script computationally estimates the DC operation point by using the fill_value = "extrapolate" argument. This prepares the data for any causal transient analysis in future and also standardize the tensor dimensions across the entire dataset.

### **3.3 Mixed-Mode Conversion (Single-Ended to Differential)**

The raw extracted 4x4 matrices represents the localized pin-to-ground wave relationship (Tx+, Tx-, Rx+, Rx-) being single-ended. The AI must be trained in teh differential domain because the high-speed systems utilize differential signaling to reject common-mode noise. 

I used the standard orthogonal matrix multiplication to transform the single-ended matrix (s_se) to mixed-mode matrix (s_mm) 

$$S_{mm} = M \cdot S_{se} \cdot M^{-1}$$


Where M is defined as, 
$$M = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & -1 & 0 & 0 \\ 0 & 0 & 1 & -1 \\ 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \end{bmatrix}$$

The most critical metrics are optimized by the neural network by this transformation which isolates the differential return loss ($S_{dd11}$) and differential insertion loss ($S_{dd21}$). [3]

### **3.4 Eigenvalue based passivity check**

A fundamental thermodynamic requirement for passive interconnects - such as vias, and copper traces - is that they cannot generate energy. An S-Parameter matrix is passive in the frequency domain if its matrix norm does not exceed 1 i.e. the dissipation matrix Q must be positive semi-definite [4]. 

This is proven using the relationship between incident power waves (a) and reflected power waves (b), where b = S.a [5]

For the network to be strictly passive, the reflected power ($b^H b$) must be less than or equal to the incident power ($a^H a$) 
			$$b^H b \le a^H a$$
	Substituting $b = S \cdot a$ into the inequality yields,
		    $$(S a)^H (S a) \le a^H a$$
		    $$a^H S^H S a \le a^H I a$$
		    $$a^H (I - S^H S) a \ge 0$$
So this mathematical proof demonstrates that the energy dissipation matrix $Q(\omega) = I - S^H S$, must be a **positive semi-definite** matrix [3]

- Software Implementation: The parsing pipeline enforces this constraint algorithmically though eigen value solver. The custom check_passivity function operates in the s-parameter matrix iteratively across all frequency points. At each freq. steps the algorithm dynamically constructs the identity matrix (I) matching the port dimensions, and compute the dissipation matrix $Q = I - S^H S$ . The script utilizes NumPy's "linalg.eigvalsh" function becasue Q is mathematically guaranteed to be Hermitian matrix (equal to its own conjugate transpose). The algorithm tracks the absolute minimum eigenvalue ($\lambda_{min}$)across the entire freq. sweep. So, the eigenvalue must be semi-positive i.e, it should always be zero or positive. 
### **4 Tensor Finalization and Traceability**

After all the physical transformations are complete, the complex s-parameters are separated into real and imaginary PyTorch tensors (Y_real and Y_imag).  The final structured dataset, along with the normalized inputs, metadata (original sim_id, reverse scalling, creation_date, mean, and standard deviations) is serialized into a .pt file and made ready for the Neural Network.

### **4.1 Verifying Parsed Data**

The parsed data with the sliced and mixed-mode 4x4 matrix is compared with the raw simulation data and below are the results - there was a perfect match between the raw and processed data confirming that the datasets are parsed successfully related to the raw simulations with guaranteeing the values and physics. 

![[Pasted image 20260324231446.png]]
![[Pasted image 20260324231512.png]]

References:

[1] https://ansyshelp.ansys.com/public/Views/Secured/Electronics/v242/en/Subsystems/Circuit/Content/Circuit/TouchstoneDataFormat.htm
[2]  [Touchstone® File Format Specification, Version 2.1, Ratified by the IBIS Open Forum January 26, 2024.](https://ibis.org/touchstone_ver2.0/touchstone_ver2_0.pdf)
[3]  [D. E. Bockelman and W. R. Eisenstadt, "Combined differential and common-mode scattering parameters: theory and simulation," in IEEE Transactions on Microwave Theory and Techniques, vol. 43, no. 7, pp. 1530-1539, July 1995, doi: 10.1109/22.392911](https://ieeexplore.ieee.org/document/392911).
[4]  [P. Triverio, S. Grivet-Talocia, M. S. Nakhla, F. G. Canavero and R. Achar, "Stability, Causality, and Passivity in Electrical Interconnect Models," in _IEEE Transactions on Advanced Packaging_, vol. 30, no. 4, pp. 795-808, Nov. 2007, doi: 10.1109/TADVP.2007.901567.](https://ieeexplore.ieee.org/document/4358038)
[5] Microwave Engineering (4th ed.), Pozar, D. M. (2011), John Wiley & Sons.
