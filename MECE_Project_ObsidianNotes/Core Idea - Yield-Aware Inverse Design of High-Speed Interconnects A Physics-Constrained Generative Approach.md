==**Problem Statement**==: High-Speed digital circuits are prone and sensitive to physical defects resulting in degradation of signal integrity as the data rate increases. This is because of non-ideal effects like skin effect and surface roughness.

High-speed digital design requires yield analysis - that need a vast amount of simulations (manufacturing variations/Tolerances) to ensure the circuit won't fail in mass production. Currently there are a few methods used in the industry,
- Commercial Solvers (HFSS/CST): These commercial solvers are of high accuracy but they are very slow for generating 10,000 + simulations which may take months to simulate.
- Standard AI innovations: AI models are being adapted to make the simulation faster but most of them use basic models, and recent innovations of using complex models have unconstrained approximations which may fail while there is a slight variation during the manufacturing.
- There are quite a few established forward model to predict the s-parameters from the given input geometrics but the exploration on a proper inverse design process is less. In this project we try to explore the inverse model to predict the geometrics from the target s-parameters also considering the manufacturing yield ()
***********************

==**Solution**==: 

- We propose a physics embedded neural network where our model predicts the constitutive physical parameters (RLGC) instead of predicting the signal directly. these predicted physical parameters are then fed to the hard-coded Telegrapher's equation layer to maintain physical consistency.
- We plan to use hybrid model inspired by Akinwade et al. (complex weights) and Konduru et al. (S-TCNN) where the complex weights are used due to the skin effect the impedance transformation is a complex phenomenon. Complex-valued neural nets ensures that frequency dependent resistance and inductance are mathematically preserved during training.
- We also try to modify the logic by predicting the physical parameters (RLGC) instead of directly predicting the S-params which kind of adds a physics verification layer to maintain the authenticity and physical constraint. (Where the physics - Telegraphers equation is hard coded) 
- The rough Arch will be- Input (Geometry) --> Complex S TCNN (model) --> Physical projections (RLGC) --> Physics Layer --> Output (s-params)
************

==**Data Generation Plan**== - try to collect as much datasets from open-source data repo if available, also generate datasets by using sobol sequencs as used in Konduru et al. (data generation plays an major part in model training so spend a lot of time for this)

Rough execution plan

Data generation / Data set collection -- Use Sobol sequences if generating data through simulation

==**Architecture**==:

Layer 1: Input and Feature Extraction - We give the geometric parameters as input and adapt S - TCNN 

Layer 2: Complex valued processing - we integrate S-TCNN with CvNN (this is done because the skin effect is complex and this will be governed through this - the complex NN is used capture the phase coupling between resistance and inductance allwoing them to lean frequency ripples)

Layer 3: Predicted RLGC - we predict the RLGC which are projected down to real output channels from the complex features from previous layer. We also apply softplus activationto make sure the physical constraints are followed (RLGC being strictly positive)

Layer 4: Physics layer - Hard-Coded Telegraphers equation where we derive the s parameters.


--> Data Generation -> Model Deployment --> Active Learning Loop --> Validation and Yield Analysis
*****************

==**Inverse model**== - backpropogation from the forward model
******************
