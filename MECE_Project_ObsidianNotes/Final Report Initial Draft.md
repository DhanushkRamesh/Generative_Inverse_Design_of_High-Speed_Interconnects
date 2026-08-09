# Dataset and Preprocessing
## A. TUHH SI/PI Database

The dataset used in this work is obtained from SI/PI Database  published by Hamburg University of Technology (TUHH) [1] , specifically the Universal-Differential-SI-Array collections used. This dataset describes a wide range of printed-circuit-board (PCB) through via structures parameterised for machine-learning applications.

Each structure is characterised by its scattering parameters (S-parameters). Each simulation is computed using the in-house via-modeling tool developed at the institute, and uses perfectly-matched-layer (PML) boundary conditions over 0.025GHz to 100GHz. The eight physical parameters listed in Table are used to describe every via, which together define the design space over which the inverse model operates.


| Parameter             | Symbol | Unit | Scale  |
| --------------------- | ------ | ---- | ------ |
| Via radius            | r_v    | mil  | linear |
| Anti-pad radius       | r_a    | mil  | linear |
| Pitch                 | p      | mil  | linear |
| Dielectric thickness  | t_d    | mil  | linear |
| Metal thickness       | t_m    | mil  | linear |
| Relative Permittivity | ε_r    | -    | linear |
| Conductivity          | σ      | S/m  | log    |
| Loss tangent          | tan δ  | -    | log    |
## B. Differential Pair Extraction

A single simulation in the database will contain multiple signal via arranged as an array. Every differential pair in the simulation is extracted as an individual sample, rather than treating each simulation as one training sample. This method was followed to increase the number of training sets without any additional simulations, improving the data available to train the forward and inverse models. In order to have a uniform representation, each pair is extracted with port ordering of [0,2,1,3] mapped to a four-port ordering of [TX+, TX-, RX+, RX-].
## C. Mixed-Mode conversion

