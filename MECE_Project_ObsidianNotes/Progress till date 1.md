Inverse Design: Baseline and changes made till now

I used the improvised data pipeline to train the forward model which gave an average MAE of 1.9dB and i froze the weights for it and used it in the tandem network for the cvae inverse model.

Inverse Design: The initial design i used was straightforward implementation of the cVAE approach to create the generative model, where the cvae generates a geometry and this is passed immediately to the forward surrogate in the tandem network. 

The baseline architecture consists of three modules in it which are, 

- **The Condition Encoder (`SConditionEncoder`):** Because S-parameters are complex, multi-port frequency responses, a 1D Convolutional Neural Network (CNN) was designed to compress the $4 \times 4$ differential target matrix ($Y_{real}$ and $Y_{imag}$) into a dense, 128-dimensional mathematical feature vector. This vector acts as the "Target Condition."
    
- **The Encoder:** A Multi-Layer Perceptron (MLP) that ingests the physical geometry constraints alongside the Target Condition, mapping them into a probabilistic latent space defined by a mean ($\mu$) and variance ($\sigma$).
    
- **The Decoder:** This is the generative engine. It samples a random vector from the latent space, combines it with the Target Condition, and outputs the final physical dimensions ($X_{local}$).
For this baseline model I restricted the loss function to three main components and did not include yield loss in this run.

**Total Loss = $MSE_{geom}$ + $\beta(KLD)$ + $\lambda(MSE_{physics})$**

1. **Geometry Reconstruction ($MSE_{geom}$):** Ensures the generated geometries do not violate physical, printable bounds (e.g., negative via radii).
    
2. **Kullback-Leibler Divergence ($KLD$):** Forces the latent space to remain organized as a continuous Gaussian distribution, allowing for smooth sampling of novel designs.
    
3. **Physics S-Parameter Loss ($MSE_{physics}$):** The dominant driving force. It calculates the Mean Squared Error between the S-parameters of the generated geometry and the user's desired target.
The baseline model was trained over 150 epochs and the MSE received in the final epoch was around **0.0108**. The proof that the model sucessfully generated the geometrics was the geometry reconstruction metric stabilizing around 0.03. I thinnk the baseline model performed better than what i expected and tracked the s-parameter profile of the target. I **need to make a proper validation of the geometrics using openEMS to verify it fully. I will try this later.**

Results:
![[Pasted image 20260609225247.png]]

After this baseline model I tried to include the yield loss and check how the model performed. I made a few changes in the loss function and how the model learns. A yield robustness penalty function was introduced to the loop and the primary modification made was in the train_inverse_epoch function, where instead of predicting the direct geometry  and evaluating it just once, the gpu was tasked with double pass evaluation.

1. **The Initial Evaluation:** The cVAE generates a nominal geometry ($X_{gen}$), which the Forward Model evaluates to find the nominal S-parameters ($S_{gen}$).
    
2. **The "Defective" Evaluation (Noise Injection):** The script injects a simulated Gaussian manufacturing defect directly into the generated geometry:
    
    `xl_gen_noisy = xl_gen + torch.randn_like(xl_gen) * 0.05` (representing a 5% standard deviation manufacturing drift).
    
3. **The Penalty Calculation:** The Forward Model evaluates this noisy geometry to find the defective S-parameters ($S_{gen\_noisy}$).
    

The mathematical difference between the perfect and defective S-parameters is calculated as the `robustness_loss`. This penalty forces the optimizer to update the network weights _away_ from fragile geometric solutions, actively pushing the latent space towards robust, manufacturing-safe plateaus.

the addition of yield penalty actually altered the computational profile and the generative behavior of the model. Firstly it was the computational complexity, because of the double pass evaluation the model took hours to train (more than the initial run of the baseline model) and at extreme frequencies (above 50 GHz), the generated S-parameter curves (notably $S_{dd21}$ insertion loss) exhibited a smoothing effect. When tasked with matching a sharp $-60$ dB resonant dip at 85 GHz, the AI instead generated a geometry that produced a shallower, safer dip at an offset frequency. 

![[Pasted image 20260609230456.png]]

Change to baseline (v1) Frequency-weighted resonance loss
I  am now working on optimizing and adding novel aspects to the baseline model and methods to improvise it. 

I tried to modify the loss and use frequency-weighted resonance loss (inspired from [Wu, Wen-Shu & Zou, Yong-Kui & Zhao, Shi-Shun & Yang, Yu-Jun. (2026). High-fidelity surrogate modeling of high-harmonic generation using Fourier neural operator with high-frequency weighted loss. Optics Express. 34. 16325-16341. 10.1364/OE.595796.](
https://www.researchgate.net/publication/403616824_High-fidelity_surrogate_modeling_of_high-harmonic_generation_using_Fourier_neural_operator_with_high-frequency_weighted_loss) ) to heavily penalize the errors at high-frequency where I multiply the S-parameter MSE by the frequency vector. A 2 dB error at 10 GHz is penalized by $1 \times$, but a 2 dB error at 90 GHz is penalized by $9 \times$. This forces the AI to prioritize matching the deep dips rather than taking the easy way out on the low frequencies.

The objective function was modified to dynamically scale the penalty based on the frequency of the error. The weight vector $W(f)$ was defined to scale linearly from $1.0$ at DC ($0$ Hz) to $5.0$ at $100$ GHz.

$$Loss_{physics} = \frac{1}{N} \sum \left( |S_{target}(f) - S_{gen}(f)|^2 \times W(f) \right)$$

This mathematical adjustment artificially inflates the gradient penalty for errors occurring at high frequencies. It forces the optimizer to prioritize the alignment of complex resonant nulls over the relatively simple low-frequency matching.

![[Pasted image 20260609234115.png]]

This performed slightly better than the initial baseline model and it did track the high-frequency curve better but i will still try to improvise it further to get an accurate generative inverse model. 

Normalizing Flows (v2)
And now I am trying a different method (Normalizing Flows) to see how the model performs.This is inspired from J[ia-Qi Yang, YuCheng Xu, Kebin Fan, Jingbo Wu, Caihong Zhang, De-Chuan Zhan, Biao-Bing Jin, and Willie J. Padilla, "Normalizing Flows for Efficient Inverse Design of Thermophotovoltaic Emitters", ACS Photonics 2023 10 (4), 1001-1011, DOI: 10.1021/acsphotonics.2c01803](https://pubs.acs.org/doi/10.1021/acsphotonics.2c01803)

This method uses a series of complex mathematical transformations that allow the AI's latent space to morph from a simple Bell Curve into highly complex, multi-modal shapes, allowing it to perfectly map sharp resonant nulls. Should check how the model performs will keep you updated in this.