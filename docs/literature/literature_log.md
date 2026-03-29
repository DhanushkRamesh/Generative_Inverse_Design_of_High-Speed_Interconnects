**Literature Search Log**

**Student:** Dhanush Kumar Ramesh

**Supervisor:** Dr. Marissa Condon

**Project:** Yield-Aware Inverse Design of High-Speed Interconnects: A Physics-Constrained Generative Approach


| Date       | Database             | Search Terms                                                                                                 | Results | Useful Hits                                            |     |     |
| :--------- | :------------------- | :----------------------------------------------------------------------------------------------------------- | :------ | :----------------------------------------------------- | --- | --- |
| 30-11-2025 | IEEE Xplore          | "integrated circuit modelling" AND "signal Integrity applications" AND "Neural Networks"                     | 27      | 1 (Akinwande et al.)                                   |     |     |
| 03-12-2025 | IEEE Xplore          | "high-density interconnects" AND "machine learning"                                                          | 43      | 1 (Sreekumar & Gupta)                                  |     |     |
| 07-01-2026 | Google Scholar       | "physics-informed" AND "interconnects"                                                                       | 1600    | 3 (Garbuglia et al.) (J. Fan et al) (T. -L. Wu et al.) |     |     |
| 12-01-2026 | Research Gate        | "PCB manufacturing" AND "Impedance" AND "Variation"                                                          | 540     | 1 (Abdelghani Renbi et al.)                            |     |     |
| 14-01-2026 | IEEE Xplore          | "signal and power integrity" AND "channel modeling"                                                          | 2       | 1 (Juhitha Konduru et al.)                             |     |     |
| 21-01-2026 | IEEE Xplore          | "SI/PI Database" AND "Machine Learning"                                                                      | 7       | 1 (Morten Schierholz et al.)                           |     |     |
| 05-02-2026 | IEEE Xplore          | "inverse design" AND "Neural Network" AND "channel"                                                          | 24      | 1 (Hanzhi Ma et al.)                                   |     |     |
| 07-02-2026 | arxiv                | "inverse problems" AND "Deep Learning"                                                                       | 8       | 1 (Jaweria Amjad et al.)                               |     |     |
| 11-02-2026 | arxiv                | Deep Learning AND "regularization techniques" AND "generalization gap" OR "classification margin"            | 41      | 2 (Judy Hoffman et al.), (Maya Janvier et al.)         |     |     |
| 14-02-2026 | arxiv                | "Conditional Variational Autoencoder" AND "Tandem Network" AND "Inverse Design" AND "physical realizability" | 1       | 1(Yuxiao Li et al.)                                    |     |     |
| 16-02-2026 | NeurIPS Proceedings  | "Generative Models" AND " Conditional Models"                                                                | 1       | 1(Sohn et al.)                                         |     |     |
| 03-03-2026 | IEEE Xplore          | "Pole-Residue" AND "Neural Networks" OR "EM sensitivity"                                                     | 25      | 1 (Feng et al.)                                        |     |     |
| 03-03-2026 | IEEE Xplore          | "Vector Fitting" OR "Rational Approximation"                                                                 | 16      | 1 (Gustavsen et al.)                                   |     |     |
| 03-03-2026 | Wiley Online Library | "openEMS"                                                                                                    | 1       | 1 (Leibig et al.)                                      |     |     |
| 16-03-2026 | IEEE Xplore          | "Mixed-Mode S-Parameters" AND "Differential S-Parameters"                                                    | 5       | 1 (Bockelman et al.)                                   |     |     |
| 09-03-2026 | ibis.org             | "Touchstone File Format Specification"                                                                       | 1       | Touchstone® File Format Specification,version 2.0      |     |     |

## 2. Log of Articles Reviewed

### 1. Reference paper for building the proposal
**Citation:**
O. Akinwande, S. Erdogan, R. Kumar and M. Swaminathan, "Surrogate Modeling With Complex-Valued Neural Nets for Signal Integrity Applications," in IEEE Transactions on Microwave Theory and Techniques, vol. 72, no. 1, pp. 478-489, Jan. 2024, doi: 10.1109/TMTT.2023.3319835. 
**Key Findings:**
* Proposes the complex-valued Neural Networks method to handle the phase information in machine learning  which is the core idea to develop our forward and inverse model. 
**Relevance**
* This is the primary reference for our forward model.

### 2. High-Density Interconnects
**Citation:**
D. Sreekumar and S. Gupta, "Efficient Synthesis and Simulation of High-Density Interconnects Using Machine Learning," 2025 IEEE 29th Workshop on Signal and Power Integrity (SPI),Gaeta, Italy, 2025, pp. 1-4, doi: 10.1109/SPI64682.2025.11014451. 

**Key Findings:**
*Discussed the sysnthesis of interconnects using machine learning which can be very usefull for data generation.
**Relevance**
it offers insights on simulation strategy for data generation of high-density designs.

**Key Findings:**
### 3. Physics-Informed Modeling
**Citation:**
F. Garbuglia, T. Reuschel, C. Schuster, D. Deschrijver, T. Dhaene and D. Spina, "Modeling Electrically Long Interconnects Using Physics-Informed Delayed Gaussian Processes," in IEEE Transactions on Electromagnetic Compatibility, vol. 65, no. 6, pp. 1715-1723, Dec. 2023, doi: 10.1109/TEMC.2023.3317917. 

**Key Findings:**
This paper used Guassian process which gives insights on physivs informed loss-functions that we refer for model design to make it accurate not just for the exact values but also for the range of values while predicting the output params.

**Relevance:**
* It is useful for considering how the moedel need to be modified and designed on combining physics informed loss and complex valued neural network.

### 4. SI Fundamentals
**Citation:**
J. Fan, X. Ye, J. Kim, B. Archambeault and A. Orlandi, "Signal Integrity Design for HighSpeed Digital Circuits: Progress and Directions," in IEEE Transactions on Electromagnetic Compatibility, vol. 52, no. 2, pp. 392-400, May 2010, doi: 10.1109/TEMC.2010.2045381.

**Relevance** 
* This paper is used to find the problem statement that we are solving which gives foundational knoledge on the non-ideal effects (skin-effects and surface roughness)

### 5. PCB Technology Overview
**Citation:**
T. -L. Wu, F. Buesink and F. Canavero, "Overview of Signal Integrity and EMC Design Technologies on PCB: Fundamentals and Latest Progress," in IEEE Transactions on Electromagnetic Compatibility, vol. 55, no. 4, pp. 624-638, Aug. 2013, doi: 10.1109/TEMC.2013.2257796. 

**Relevance**
* This paper gives the context for stochastic manufacturing variations during the manufacture of PCB. This will help us with the project to consider the schotastic variation while predicting the s-params and during the output.

### 6. Manufacturing variations
**Citation**
Renbi, A.; Carlson, J.; Delsing, J. Impact of pcb manufacturing process variations on trace impedance. In Proceedings of theInternational Symposium on Microelectronics, Long Beach, CA, USA, 9–13 October 2011; International Microelectronics As-sembly and Packaging Society: Pittsburgh, PA, USA, 2008; pp. 000891–000895. 

**Key Findings**
The author demonstrates that the use of 1-D convolution neural networks are more efficient and effective than the standard dense layers to capture the high-frequncy ripples in the S-Parameters. 

**Strength**
1_D CNN architecture captures the harmonics in the freqency data i.e. long range dependencies

**Weakness**
In this paper the model addresses only the forward path and no Inverse model is implemented

**Relevance**
This paper provide a reference for improving the architecture for the forward model where the 1-D CNN can be adapted to build the tandem loop to detect the small signal deviations caused by manufaturing defects.

### 7. Convolution Nets for Forward Modeling (used as reference for supporting the state-of-the-art forward simulation)
**Citation**
J. Konduru, O. Mikulchenko, L. Y. Foo and J. E. Schutt-Ainé, "Signal Integrity Analysis and Design Optimization using Neural Networks," _2024 IEEE 74th Electronic Components and Technology Conference (ECTC)_, Denver, CO, USA, 2024, pp. 924-928, doi: 10.1109/ECTC51529.2024.00150.

**Key Findings**
The author demonstrates that the use of 1-D convolution neural networks are more efficient and effective than the standard dense layers to capture the high-frequncy ripples in the S-Parameters. 

**Strength**
1_D CNN architecture captures the harmonics in the freqency data i.e. long range dependencies

**Weakness**
In this paper the model addresses only the forward path and no Inverse model is implemented

**Relevance**
This paper provide a reference for improving the architecture for the forward model where the 1-D CNN can be adapted to build the tandem loop to detect the small signal deviations caused by manufaturing defects.

### 8. Database for the model
**Citation**
M. Schierholz _et al_., "SI/PI-Database of PCB-Based Interconnects for Machine Learning Applications," in _IEEE Access_, vol. 9, pp. 34423-34432, 2021, doi: 10.1109/ACCESS.2021.3061788.

**Key Findings**
This paper provides the biggest dataset for SI/PI data of PCB-Based Interconnects for Machine Learning Applications

**Relevance**
We use this data to train the model for our machine learning architecture.

### 9. Inverse model Tandem Network
**Citation**
H. Ma et al., "Channel Inverse Design Using Tandem Neural Network," 2022 IEEE 26th Workshop on Signal and Power Integrity (SPI), Siegen, Germany, 2022, pp. 1-3, doi: 10.1109/SPI54345.2022.9874935.

**Key Findings**
The paper explores channel inverse design using Tandem neural network to solve non-uniqueness problem in invese design. They connect the inverse network to the pre-trained forward model to converge the geometrics from the s-parameters

**Strength**
Tandem neural net solves the one-to-many mapping problem as there can be multiple geometries that can yield the same s-parameters

**Weakness**
The model just assume the forward model without any optimization wihtout any circuit constraints
**Relevance**
This Tandem Neural Net model cn be adapted as baseline where we can inprovise the architecture further for our model.

### 10. Jacobian Loss Function (reference)
**Citation**
J. Amjad, Z. Lyu, and M. R. D. Rodrigues, "Deep Learning for Inverse Problems: Bounds and Regularizers," in arXiv preprint arXiv:1901.11352, 2019.

**Key Findings**
This paper proves that the stability of the solution is directly linked to the spectral norm of the Jacobian matrix for the inverse problems. The neural network can be precvented from being tricked by regularizing the Jacobian.

**Strength**
This provides the theorem connecting Jacobian size to the generelazation errors

**Relevance**
Even though the core concept is focused on image reconstruction, the Jacobian regularization conceept can be adapted to apply in PCB manufacturing yield.

### 11. Jacobian Loss Function (foundation)
**Citation**
Judy Hoffman et al., “Robust Learning with Jacobian Regularization”, 
https://doi.org/10.48550/arXiv.1908.02729](https://doi.org/10.48550/arXiv.1908.02729)

**Key Findings**
The authors by introducing an efficient approximation algorithm using random projections, they try to resolve the traditional computational bottleneck of Jacobian calculation. This reduces the computational overhead.

**Strength**
This paper proves that Jacobian regularization pushes the decision boundaries outwards to find the flatter and more stable prediction spaces. This is mathematically rigorous and computationally efficient method. 

**Weakness**
This methodology is fundamentally tested and framed in computer vison. We will implement in high-speed interconnects to explore inverse design problems.

**Relevance**
This paper provides the fundamental reference to use Jacobian Yield to focus on yield-aware inverse model. We use Jacobian regularization to defend the cVAE against manufacturing tolerances. 

### 12. Variational Autoencoders (reference and inspiration)
**Citation**
Yuxiao Li, Taeyoon Kim, Allen Zhang, Zengbo Wang, Yongmin Liu,"On-Demand Inverse Design for Narrowband Nanophotonic Structures Based on Generative Model and Tandem Network",https://doi.org/10.48550/arXiv.2507.14761.

**Key Findings**
This paper uses conditional variational autoencoders to filter user input before they connect to the tandem network. The idealized target spectrum is taken by the cVAE and modified into cVAE adjusted target spectrum. This adjusted spectrum reduces the prediction errors and hence solve the inverse problem - one-to-many problem.

**Strength**
It prevents the deterministic inverse network from crashing and solves the issue of hallucination. 

**Weakness**
This framework is not explored much in signal integrity applications. It is applied in optical nano photonics. 

**Relevance**
This papers shows the architectural framework for the generative inverse design used in our project. We take this foundation model and build it to solve one-to-many problem.

### 13. Foundation of Conditional Variational Autoencoders

**Citation**
Kihyuk Sohn, Xinchen Yan, Honglak Lee, "Learning Structured Output Representation using Deep Conditional Generative Models", 2015, NIPS'15: Proceedings of the 29th International Conference on Neural Information Processing Systems - Volume 2,Pages 3483 - 3491

**Key Finding**

This paper addresses the limitation of using standard convolution networks and propose a conditional generative model (cVAE) 

**Strength**
In this literature the authors find the discrepancy in the training and testing pipeline of cVAE and address it by  introducing Gaussian Stochastic Neural Networks (GSNN)

**Relevancy**
This paper is the initial inspiration of using cVAE framework in my project to build the generative model.

### 14. Rational Layer Inspiration
**Citation**
F. Feng, V. -M. -R. Gongal-Reddy, C. Zhang, J. Ma and Q. -J. Zhang, "Parametric Modeling of Microwave Components Using Adjoint Neural Networks and Pole-Residue Transfer Functions With EM Sensitivity Analysis," in IEEE Transactions on Microwave Theory and Techniques, vol. 65, no. 6, pp. 1955-1975, June 2017, doi: 10.1109/TMTT.2017.2650904.

**Key Findings**
This paper used the vector fitting method of finding poles and residues for sensitivity analysis based neuro transfer function that utilizes the EM sensitivity information to increase the model accuracy and also to reduce the amount of data required for training. This model uses pole-residue format - Instead of treating the s-parameters as arbitrary function, it defines a function as a function of frequency using poles and residues.

**Strength**
The model employs a two-stage training process. There is a preliminary training on the extracted poles and residue data followed by model refinement against the final EM response. This helps with data efficiency and accuracy.

**Weakness**
The order (no. of poles) can change as the geometry shifts. This causes mathematical discontinuity  this is one of the challenge in SI modeling. 

**Relevance**
I use poles-residues method in my forward model as well but this paper is only an inspiration where I'm trying build a forward proxy model to incorporate physics in construction. 

### 15. Foundational Math of Poles-Residues methodology

**Citation**
B. Gustavsen and A. Semlyen, "Rational approximation of frequency domain responses by vector fitting," in IEEE Transactions on Power Delivery, vol. 14, no. 3, pp. 1052-1061, July 1999, doi: 10.1109/61.772353.

**Key Finding**
This paper serves as the foundation of using vector fitting methodology. This paper is useful to know the mathematical framework for accurately fitting rational functions to the frequency data. 

**Strength**
The authors provide strict mathematical guidelines to avoid poor conditioning of the linear problem in stage 1

For functions with resonance peaks:  The Starting poles must be complex conjugate pairs distributed linearly over the frequency range. To avoid ill-conditioning, the real part should be set exactly to 1% of the imaginary part.

**Weakness**
This is an iterative process and computationally heavy to find the poles and residues using the iterative method.

**Relevancy**
I will be using poles-residues method (Vector Fitting) but will not be using the iterative workflow, instead, I'm modifying the process by using the poles and residues as trainable weights. 

### 16. openEMS

**Citation**
Liebig, T., Rennings, A., Held, S. and Erni, D. (2013), openEMS – a free and open source equivalent-circuit (EC) FDTD simulation platform supporting cylindrical coordinates suitable for the analysis of traveling wave MRI applications. Int. J. Numer. Model., 26: 680-696. https://doi.org/10.1002/jnm.1875

**Key Finding**
This paper is the foundation of openEMS - an opensource EM solver.

**Relevance**
I will be using openEMS in the active learning loop in my project and I will also use it for validation of the generated results by running simulation - since this is opensource and has python API it will be useful to integrate.

### 17. Mixed-Mode S-Parameters

**Citation**
D. E. Bockelman and W. R. Eisenstadt, "Combined differential and common-mode scattering parameters: theory and simulation," in _IEEE Transactions on Microwave Theory and Techniques_, vol. 43, no. 7, pp. 1530-1539, July 1995, doi: 10.1109/22.392911.

**Key Finding**
This paper was referred to implement the single end to mixed mode parameter conversion of the extracted core matrix. 

**Relevance**
The math form the paper was adapted for the conversion of single end to mixed mode conversion of the extracted core 4x4 matrix that was sliced from the raw dataset S-Parameters.

### 18. Port Extraction Logic

**Citation**
[Touchstone_® _File Format Specification_, Version 2.0, Copyright© 2009 by TechAmerica.]([Microsoft Word - touchstone_ver2_0.doc](https://ibis.org/touchstone_ver2.0/touchstone_ver2_0.pdf)), Ratified by the IBIS Open Forum April 24, 2009.

**Key Finding**
The document used in commercial EM solvers like Ansys to format the ports while design and the touchstone fine after simulation. 

**Relevance**
This document is the official file format specification for the touchstone file (.SnP) used for documenting the S-Parameters. This was referred to find the port logic used for extraction of core 4x4 matrix from the raw touchstone files in the dataset - to identify the core ports with least attenuation.
