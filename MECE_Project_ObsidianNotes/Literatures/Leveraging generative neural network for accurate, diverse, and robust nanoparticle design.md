## Leveraging generative neural networks for accurate, diverse, and robust nanoparticle design

In this paper they research the use of Variational Autoencoderto generate multiple, diverse, accurate results. They compare the use of VAE with Tandem Network - to address one-to-many mapping problem.

Architecture:

The VAE architecture - Encoder: Takes the paartical design and optical condition to map to low-dimensional latent space.

Decoder: Uses the latent vector and the condition to reconstruct the design parameters. 

They used AdamW optimizer to improve training speed and maintain accuracy.

In the results - VAE was superior as it could explore the latent space to ind several different structures that met the same target spectra.