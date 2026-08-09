1. Try to study and investigate which frequency range is important and which can be ignored, assessing the relative importance of regions of the frequency range, see if it could be split into regions of varying importance and if this could inform a process for combining the regions.
E. Zhu, E. Li, Z. Wei, Y. Che, Q. Wang and W. -Y. Yin, "Conjugate Adjoint Gradient-Based Inverse Design Method for Aperiodic Frequency-Selective Surface With Weighted Loss," in _IEEE Transactions on Electromagnetic Compatibility_, vol. 66, no. 1, pp. 131-142, Feb. 2024, doi: 10.1109/TEMC.2023.3312114.

Data Study to investigate the frequency range to give importance for:

sdd11 focus on low freq and sdd21 on high freq.

The EDA revealed some critical insights,
- The sensitivity of Return loss and Insertion Loss are inverted, crossing at around 24.7GHz. $S_{dd11}$ variance peaks at baseband, while $S_{dd21}$ variance peaks at high frequencies.
![[Pasted image 20260628212559.png]]
- Another insight was the deep null distribution, where 77.2% of deep resonant nulls for $S_{dd11}$ cluster in the 0–14 GHz band, while 82.5% of $S_{dd21}$ nulls cluster above 56 GHz. The loss function must specifically target these bands to capture resonances.
![[Pasted image 20260628212707.png]]
- Mode conversion ($S_{dc}$) has a quiet median magnitude (-30.8 dB) at baseband but an extreme variance of 16.11 dB. Changing the geometry causes catastrophic Electromagnetic Compatibility (EMC) swings, demanding high prioritization.
![[Pasted image 20260628212842.png]]

So I developed a dynamic algoritheminc decision tree, where the algorithm assigns the weights based on statistical triggers instead of being hardcoded. 
 - Base weight: Regions receive higher weights (up to 2.0x) if their median signal is strong (> -25 dB) and their geometric variance is significant (> 5 dB).
 - This  algorithm will generate a static (4,4,401) element aware weight tensor, which is used to perform a weighted Mean Squared Error (MSE) reduction during cVAE training, and acts as the physics-constrained score function during inference via Latent Test-Time Optimization (TTO) - method used in inverse design to optimize the model.


To improve the accuracy of the inverse model, I tried Test Time Optimization TTO inspired from  ( LaBash, B., Khushrushahi, S., & Ruehle, F. (2025). Improving Generative Inverse Design of Rectangular Patch Antennas with Test Time Optimization. ArXiv. https://arxiv.org/abs/2505.18188). 

So in the initial tandem architecture with cVAE connected with the frozen forward model. i inspired the Test Time Optimization from LaBash et al, where they successfully applied Test-Time Optimization (TTO) to rectangular patch antennas, I implemented an inference-time gradient descent loop. 

My model (cVAE) generated the geometric guesses and I detached this geometry, enabled gradients, andd optimized the via dimensions directly against the target S-parameters using a piecewise masked MAE loss.

But with this method the model suffered with surrogate exploitation. 
- The TTO captured the return loss successfully within the Nyquist band (0–28 GHz), but the insertion loss degraded in the 30 - 60 GHz band. my study was the gradient descent took steps of size `lr=0.05` across 75 iterations, eventually guessing outside the physical limit of the dataset and discovered a geometry that was unphysical via geometry that perfectly satisfied the $S_{dd11}$ math in the Forward Model, but caused the surrogate to hallucinate the degraded $S_{dd21}$ curve.
![[Pasted image 20260628221651.png]]

Next Step : I'm now working on improvising this model by using Latent Space TTO (Pascal Notin, José Miguel Hernández-Lobato, Yarin Gal, "Improving black-box optimization in VAE latent space using decoder uncertainty", https://doi.org/10.48550/arXiv.2107.00096) and Curriculum TTO (Liu, Z.; Shan, G.; Chen, Z.; Yang, Y. Physics-Guided Neural Surrogate Model with Particle Swarm-Based Multi-Objective Optimization for Quasi-Coaxial TSV Interconnect Design. _Micromachines_ **2025**, _16_, 1134. https://doi.org/10.3390/mi16101134) with passivity.

in Latent Space TTO the idea is to optimize the latent space (Latent vector z) instead of physical dimensions. gradient must backpropagate through the frozen cVAE Decoder to update z. And after this, i have an idea to implement the method inspired from (Liu, Z.; Shan, G.; Chen, Z.; Yang, Y. Physics-Guided Neural Surrogate Model with Particle Swarm-Based Multi-Objective Optimization for Quasi-Coaxial TSV Interconnect Design. Micromachines 2025, 16, 1134. https://doi.org/10.3390/mi16101134).

I use curriculum masking - for the first 45 steps, the optimizer is entirely blinded to frequencies above 28 GHz. It is forced to lock the macroscopic via impedance at the baseband. For the remaining steps, the full Element-Aware tensor is unlocked to tune micro-resonances. And Liu et al, demonstrated that applying passivity regularization prevents unphysical predictions in interconnects. We adapted this concept directly into the TTO gradient loop. At each optimization step, we calculate the Singular Value Decomposition (SVD) of the predicted 4x4 S-matrix. If the maximum singular value exceeds 1.0 (indicating the passive structure is artificially generating energy), an aggressive ReLU penalty rejects the optimization step. This completely eliminates surrogate exploitation. This is the idea I have and I'm writing the code to improvise this in the TTO. 

in parallel I'm still working on the simulation using openEMS to validate the results. 

****************************************
