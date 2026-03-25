Traditional EM solvers are slow and time consuming so Konduru et al, introduce a Neural Network approach to solve this.

In this paper they use Transposed Convolution NN where they intend to learn the correlations within the input data to get the corresponding frequency response using convolution layers. They use 1-d convolution instead of 2-D because they only want to study the special correlation that exists along single frequency axis. They use TCNN because they wanted to find the frequency response from the pattern rather than the pattern from frequency response and TCNN maps low-dimensional input parameters to high- dimensional frequency response. 

They ensure the predicted frequency response from the S-TCNN is physcally consisted by introducting Causality and passivity using Causality Enforcement Layer (CE) and Passivity enforcement Layer(PEL). CEL use the Hilbert transofrm to related the real and imaginary part of the S-parameters and EL ensure singular values to be less than 1. 

Data Generation Strategy:

They use Sobol sequences ( a type of quasi-random sequence), these are designed to spread out as evenly as possible, ensuring the model examples from the corner and center of the design space equally. Due to the data generated using Sobol Sequences, they only needed to generate 393 samples to train the entire network. 

Result: They found that the S-TCNN model was 200x faster than the traditional EM solvers.




