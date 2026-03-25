- Hessian-Aware Yield Mapping (HAYM) instead of just Jacobian 

Via stub resonance is highly non-linear, but the Jacobian (first-order derivative) assume linear drop-off. We upgrade the regularization loss to have a second-order partial derivatives - Hessian Matrix.

- optimize for multi-variable 3D manufacturing corners. Instead of optimizing for one tolerance, optimize for a "Worst-Case Hyper-Volume".
- Eigenvalue passivity check in dataloader - to be executed now (after this verify openEMS and complete phase 1)

