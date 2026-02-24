
Slide 1: The Hook

Image idea: [To add a clean 112G eye diagram on left, a failed eye diagram on right with a PCB drill in the middle depicting a slight tolerance will have a failure in yield. (one of the main focus of our project)]

Imagine building an F1 car that loses balance if the wind changes by 1 mile per hour. That is the reality of high-speed circuit design today.

As we move to 112G data rates, the margins for error shrink drastically. At these speeds, a microscopic manufacturing variation—say, a drill bit going just 10% too deep—can severely degrade the signal integrity of the PCB.

Currently, ensuring yield requires running thousands of slow Monte Carlo simulations in traditional CAD tools. Meanwhile, state-of-the-art AI models try to replace this, but they entirely focus on finding mathematically optimal design prioritizing pure accuracy. These models work perfectly in ideal, zero-variance simulations but may fail in the factory if the manufacturing variances are not taken into account.

While the industry has mastered the forward simulation, robust inverse design remains largely unsolved. 

My project challenges this by asking: "Can we build an AI model that designs for both performance (accuracy) and also high manufacturing yield?"

Slide 2:  Proposed Solution

Image idea: ["Yield-Tandem VAE" Architecture Diagram. Highlight 3 blocks: The VAE (Generator), The Rational Layer (Physics), and the Jacobian Loss (Robustness).]

We propose a novel framework - the yield Tandem-VAE. This architecture moves beyond standard Deep Learning by integrating three specific innovations.

First, to solve the one-to-many problem - where multiple geometries can produce the same signal. We use conditional Variational Autoencoders (cVAE) - unlike standard regression models that output a single average design, we generate a distribution of valid geometries for the engineers to choose from. 

Second, to enforce physics. We introduce a Differentiable Rational Layer, instead of predicting raw data points using computationally heavy complex penalty functions, our network predicts poles and residues. This mathematically guarantees that every generated design obeys Kramers-Kronig causality, preventing the AI from generating non-physical outputs or hallucinating. 

Third, and most importantly, we introduce a Jacobian Yield Loss. By using a finite-difference approximation of the Jacobian matrix during training, we penalize designs that sit on sharp peaks in the loss landscape—where a small manufacturing change leads to failure. Instead, this forces the model to find flat plateaus that are completely stable, even with manufacturing tolerances."

Slide 3: Why it matters

image idea: [A chart comparing "Standard AI Yield (50%)" vs. "Ours (90%)". A bullet point list: "1. Guaranteed Causality. 2. Factory-Ready Designs. 3. 1000x Speedup". A compelling data visualization. Which uses probability distributions to contrast the "risky" nature of standard AI with the "robust" nature of my solution, directly supporting the "Yield is King" message.]

Why does this matter? Because in the hardware industry, Yield is King!

The design that works in simulation and fails in the factory cost millions in re-spins. Furthermore, most of the standard inverse papers ignore yield entirely.  Our work bridges the gap between academic AI and industrial reality.

By optimizing the Jacobian, we guarantee that the designs will be robust to manufacturing tolerances. And by enforcing the rational layer, engineers know the AI is not violating physical laws.

Ultimately, this transforms AI from being a simple calculator into a physics-aware co-designer—accelerating development from weeks of simulation down to mere milliseconds of inference."

