
### **The 8-Minute Natural Pace Script**

**Slide 1:  "Hello everyone. My name is Dhanush Kumar Ramesh, and today I will be presenting the status report for my project: _Yield-Aware Inverse Design of High-Speed Interconnects using a Physics-Constrained Generative Approach_. This research is being supervised by Dr. Marissa Condon. Over the next eight minutes, I will walk you through a critical industry problem, the limitations of current AI, and the novel framework I am developing to solve it." `[Pause for slide transition

**[Slide 2:] ** "Let's begin with the core problem. As data rates in the industry push towards 112G and 224G, high-speed interconnects—specifically vertical vias—have become incredibly sensitive to microscopic manufacturing errors.

While we can easily optimize a perfect via in a simulation, reality is much messier. As you can see in the cross-section image, a very common fabrication error is a slight variance in drill depth. If a drill bit goes just 10% too deep, it leaves behind a small copper remnant called a 'via stub'.

Electromagnetically, this stub creates a massive problem. It causes a capacitive and inductive discontinuity, resulting in a severe resonance dip that distorts the signal.

This creates what I call a 'Yield Crisis'. Current AI models might find the mathematically perfect design in a virtual simulation, but they completely ignore the reality of manufacturing tolerances on the factory floor." `[Pause]`

**[Slide 3] ** "When we look at the current state-of-the-art, there is a distinct gap in how we solve this.

Currently, the gold standard is using commercial Electromagnetic solvers like CST or HFSS. While highly accurate, they are computationally far too slow for iterative inverse design.

To speed this up, researchers have shifted to AI surrogate models. While these AI models are fast, they have two major flaws. First, they require computationally heavy post-processing just to ensure their outputs obey basic physics, like causality. Second, they optimize only for nominal environments—meaning they completely ignore manufacturing yield.

The gap I am addressing is the lack of a generative AI model that combines automated yield optimization with a strictly physics-constrained architecture." `[Pause]`

**[Slide 4:] ** "To bridge this gap, I am proposing the **Yield-Tandem Architecture**. My goal here is to use AI as a physics-aware co-designer.

As you can see in the flowchart, my architecture consists of three distinct mathematical innovations: First, a Conditional Variational Autoencoder, or cVAE, which acts as the generator. Second, a differentiable Rational Layer to strictly enforce physical laws. And third, a Jacobian Yield Loss function to guarantee manufacturability.

Let’s look at how each of these pillars works." `[Pause]`

**[Slide 5: ] "The first pillar addresses geometry generation. In inverse design, we face a 'one-to-many' problem. Multiple different physical geometries can actually yield the exact same S-parameter profile.

Standard AI models try to solve this by minimizing Mean Squared Error. The issue is that MSE forces the network to output a single, averaged geometry, which is incredibly risky for manufacturing.

My approach replaces this with a cVAE. Instead of outputting a single point, my model maps the target S-parameters into a continuous statistical distribution, or latent space, parameterized by a mean and a variance. By sampling from this space, the decoder network generates a diverse _family_ of valid physical solutions. This gives the engineer multiple robust options to choose from." `[Pause]`

**[Slide 6: ]** "Once a geometry is generated, we must verify its electrical performance. However, a major issue with standard neural networks is that they often hallucinate frequency responses that are non-causal or non-passive.

Current models fix this by running heavy post-processing scripts. I am solving this natively. My forward model replaces standard output neurons with a differentiable Rational Layer.

Instead of predicting raw data points, my neural network is mathematically constrained to output the coefficients—specifically the poles and residues—of a rational transfer function. Crucially, I treat these poles and residues as trainable weights. During backpropagation, these coefficients are directly optimized and constrained to stable regions.

Because of this, my generated designs are forced to obey Kramers-Kronig causality by construction. The AI simply cannot hallucinate a physically impossible signal." `[Pause]`

**[Slide 7: J] "The final innovation ensures the AI designs for the real world. As you can see in the 3D optimization landscape on the right, standard AI often converges on sharp, fragile peaks. If the manufacturing tolerance shifts even slightly, the design falls off the cliff and the hardware fails.

To prevent this, I introduce a Jacobian Yield Regularization loss function. During training, I calculate the finite-difference approximation of the Jacobian matrix. This penalty actively punishes the network for selecting highly sensitive geometries. Instead, it forces the AI to seek out 'flat plateaus'—designs that maintain strong signal integrity even if the drill depth varies by 10%.

To validate this, I have integrated an Active Learning loop. A Python script automatically queries the open-source openEMS solver to simulate geometries where the AI is uncertain, continuously updating the model's understanding of real-world physics." `[Pause]`

**[Slide 8: ] "In terms of my progress to date, I have finalized the mathematical architecture and acquired the TUHH SI/PI database to serve as my ground truth.

I engineered a data pipeline that mathematically extracts the core 4x4 differential sub-matrix from this data. This standardizes the dataset for PyTorch tensor batching and properly splits the features into global conditions and local targets.

I am also currently exploring the openEMS Python API to fully automate the active learning simulation loop." `[Pause]`

**[Slide 9: ]"Looking at the project timeline, I am currently on schedule. I am wrapping up Phase 2, which is the core architecture implementation.

Through May, my focus will shift to Phase 3: automating the Active Learning loop and testing the Jacobian math. In June, I will move to Phase 4, benchmarking my model against standard AI baselines.

Ultimately, my success criteria is to generate causal via geometries that maintain acceptable S21 margins while easily withstanding a plus-or-minus 10% manufacturing variance." `[Pause]`

**[Slide 10: ]"Here are the primary references that support my methodology. Thank you very much for your time and attention today, and I would be happy to take any questions."