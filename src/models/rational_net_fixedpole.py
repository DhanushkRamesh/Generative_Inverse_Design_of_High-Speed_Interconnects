import torch
import torch.nn as nn
import torch.nn.functional as F


class RationalNet(nn.Module):
    """
    Physics-Informed Rational Neural Network for S-parameter prediction of high-speed interconnects.
    
    VF-Informed Fixed-Pole Architecture:
      This version separates the rational function coefficient problem into two parts,
      motivated by the observation that for a family of PCB via structures sharing the
      same physical class, cavity resonance frequencies are approximately geometry-invariant
      while coupling strengths vary significantly with via parameters.

      - Poles are FIXED (non-trainable) — extracted from classical Vector Fitting
        (Gustavsen and Semlyen, 1999) on a small number (~10) of representative samples,
        then clustered into a shared spectral basis using K-Means. Since poles determine
        WHERE resonances occur, and this is approximately constant across the dataset,
        fixing them eliminates the non-convex pole-placement problem that caused
        end-to-end training to fail (documented in v1 and v2 experiments).
      
      - Residues are TRAINABLE — predicted by the neural network backbone from geometry
        features. Since residues determine HOW STRONGLY each port combination couples to
        each resonance, and this varies smoothly with geometry, the network solves a
        well-conditioned regression problem that backpropagation handles reliably.

    Causality Guarantee:
      Poles are complex numbers with Re(Pn) < 0, extracted from VF on passive physical data.
      Since they are non-trainable buffers, no gradient can push them into the right half-plane.
      Causality is guaranteed by construction — mathematically, not by soft penalty.
      This satisfies Objective 2 of the Yield-Tandem Architecture in its strongest form.

    Literature Trail:
      - Gustavsen & Semlyen (1999): foundational Vector Fitting algorithm, established that
        good starting poles are critical for convergence and can be reused across similar problems.
      - Feng et al. (2017, IEEE T-MTT): adjoint neural networks with pole-residue transfer
        functions for parametric microwave component modelling.
      - Boullé et al. (2020, arXiv:2004.01902): rational neural networks with superior
        approximation properties for functions with poles and singularities.
      - Liu et al. (2025, MDPI Micromachines): Neuro-TF for TSV interconnects, demonstrated
        that VF priors stabilise broadband learning in neural surrogates.
      - Silva Rezende et al. (2025, Springer e+i): pole-residue surrogate modelling with
        clustering to resolve the pole-ordering problem across parametric sweeps.
    """

    def __init__(self, num_poles_half, num_local_features, num_global_features, 
                 shared_poles_path, num_ports=4, hidden_dim=512):
        super(RationalNet, self).__init__()

        self.num_poles_half = num_poles_half
        self.num_poles = num_poles_half * 2  # Full count including conjugate mirrors
        self.num_ports = num_ports
        self.hidden_dim = hidden_dim
        total_features = num_local_features + num_global_features

        # --------------------------------------------
        # Load and Register Fixed Shared Poles
        # --------------------------------------------
        # Poles are loaded from the VF-extracted shared basis and registered as
        # non-trainable buffers. register_buffer ensures they:
        #   1. Move to the correct device (GPU/MPS) automatically with model.to(device)
        #   2. Are saved/loaded with model state_dict for checkpoint compatibility
        #   3. Are NOT included in model.parameters() — no gradients, no optimizer updates
        #
        # The poles represent the spectral basis for the entire dataset. For any
        # given geometry, the neural network "activates" relevant poles by predicting
        # large residues for them, and "silences" irrelevant poles with near-zero
        # residues. This is analogous to how Fourier series uses fixed frequencies
        # and adjusts amplitudes — the basis doesn't need to match exact resonances
        # to achieve good approximation.
        pole_data = torch.load(shared_poles_path, weights_only=False)
        
        # Verify pole count matches model configuration
        assert pole_data['num_poles_half'] == num_poles_half, \
            f"Pole file has {pole_data['num_poles_half']} poles, model expects {num_poles_half}"
        
        fixed_poles_real = pole_data['poles_real']  # [num_poles_half]
        fixed_poles_imag = pole_data['poles_imag']  # [num_poles_half]
        
        # Verify causality — all real parts must be strictly negative
        assert torch.all(fixed_poles_real < 0), \
            "FATAL: Loaded poles contain non-causal entries (Re >= 0)"
        
        # Build full pole set with conjugate symmetry:
        # Upper half-plane poles: Pn = σn + jωn
        # Lower half-plane mirrors: Pn* = σn - jωn
        # This guarantees real-valued impulse response h(t) = Σ Rn·exp(Pn·t)
        full_poles_real = torch.cat([fixed_poles_real, fixed_poles_real])  # [num_poles]
        full_poles_imag = torch.cat([fixed_poles_imag, -fixed_poles_imag])  # [num_poles]
        full_poles = torch.complex(full_poles_real, full_poles_imag)  # [num_poles]
        
        # Register as buffer — NOT a parameter, NOT trainable
        self.register_buffer('fixed_poles', full_poles)
        
        print(f"  [RationalNet] Loaded {num_poles_half} shared poles from {shared_poles_path}")
        print(f"  [RationalNet] Pole freq range: {fixed_poles_imag.min()/(2*3.14159):.2f} to {fixed_poles_imag.max()/(2*3.14159):.2f} GHz")
        print(f"  [RationalNet] All poles causal (left half-plane): ✓")

        # --------------------------------------------
        # Multi-Scale Fourier Feature Projection
        # --------------------------------------------
        # We project the input into a high-dimensional space using a fixed random Gaussian matrix.
        # Two frequency scales capture both smooth (low-freq) and sharp (high-freq) geometry variations.
        # The low-scale captures gradual parameter trends (e.g. trace length vs insertion loss slope),
        # while the high-scale captures sharp transitions (e.g. via radius near resonance boundaries).
        B_low = torch.randn(total_features, 64) * 1.0
        B_high = torch.randn(total_features, 64) * 10.0
        self.register_buffer('B_matrix', torch.cat([B_low, B_high], dim=-1))  # fixed random Gaussian
        # After sin/cos concatenation: 256-dim Fourier features

        # --------------------------------------------
        # MLP Backbone: shared feature extractor
        # --------------------------------------------
        # Three residual blocks with LayerNorm for stable gradient flow.
        # The mapping from geometry → residues is smoother than geometry → (poles + residues),
        # but still requires sufficient depth to capture how coupling strengths vary
        # nonlinearly with via radius, antipad diameter, pitch, and stackup parameters.
        self.input_layer = nn.Linear(256, hidden_dim)

        # Residual Block 1: initial feature extraction from Fourier projections
        self.residual_block_1 = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)

        # Residual Block 2: deeper abstraction of geometry-to-residue mapping
        self.residual_block_2 = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)

        # Residual Block 3: final refinement before residue prediction heads
        self.residual_block_3 = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layer_norm_3 = nn.LayerNorm(hidden_dim)

        # --------------------------------------------
        # Output Heads: residues and direct term ONLY
        # --------------------------------------------
        # No poles heads — poles are fixed buffers, not predicted.
        # This is the key architectural change from v1/v2.

        # 1. Residues heads: predict coupling strengths for each pole-port combination.
        #    Each pole has a num_ports x num_ports residue matrix encoding how
        #    strongly that resonance couples between each pair of ports.
        #    Real and imaginary parts predicted separately — PyTorch optimisers
        #    handle real-valued gradients better than complex-valued ones.
        self.residues_real = nn.Linear(hidden_dim, num_poles_half * num_ports * num_ports)
        self.residues_imag = nn.Linear(hidden_dim, num_poles_half * num_ports * num_ports)

        # 2. Direct term head (D): real-valued constant feedthrough matrix.
        #    D is strictly real for passive networks (Gustavsen and Semlyen, 1999).
        self.d_term_real = nn.Linear(hidden_dim, num_ports * num_ports)

        # --------------------------------------------
        # Careful Initialisation of Residues
        # --------------------------------------------
        # Residue weights initialised small to prevent large S-matrix values
        # early in training that would trigger aggressive passivity corrections.
        # With fixed poles already at the correct resonance locations, the residues
        # only need to learn the coupling strengths — starting from small values
        # and growing as the network learns the geometry-to-coupling mapping.
        with torch.no_grad():
            nn.init.normal_(self.residues_real.weight, std=0.01)
            nn.init.constant_(self.residues_real.bias, 0.0)
            nn.init.normal_(self.residues_imag.weight, std=0.01)
            nn.init.constant_(self.residues_imag.bias, 0.0)
            nn.init.constant_(self.d_term_real.bias, 0.0)


    def forward(self, x_local, x_global):
        """
        Forward pass: computes residues and direct term from via geometry.
        Poles are fixed — retrieved from self.fixed_poles buffer.

        Args:
            x_local:  [batch_size, num_local_features]  — local geometry targets
            x_global: [batch_size, num_global_features] — global layout constraints
        Returns:
            poles:    [batch_size, num_poles] complex64         — fixed, conjugate symmetric
            residues: [batch_size, num_poles, num_ports, num_ports] complex64
            d_term:   [batch_size, num_ports, num_ports] complex64 — real part only, zero imag
        """
        # Combine local and global features into a single input vector
        x = torch.cat((x_local, x_global), dim=1)

        # --------------------------------------------
        # Fourier Feature Projection
        # --------------------------------------------
        # Linearly scale Z-scores to [0,1] and clamp outliers to prevent Fourier aliasing.
        x_norm = torch.clamp((x + 3.0) / 6.0, min=0.0, max=1.0)
        x_proj = 2.0 * torch.pi * (x_norm @ self.B_matrix)
        x_fourier = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        # --------------------------------------------
        # Deep Residual Backbone
        # --------------------------------------------
        h = self.input_layer(x_fourier)  # [batch_size, hidden_dim]
        h = self.layer_norm_1(self.residual_block_1(h) + h)
        h = self.layer_norm_2(self.residual_block_2(h) + h)
        h = self.layer_norm_3(self.residual_block_3(h) + h)

        batch_size = h.shape[0]

        # ---- Poles (FIXED — no gradient, no learning) ----
        # Expand the shared pole buffer to match batch dimension.
        # Every sample in the batch uses the same pole locations —
        # only the residues (predicted below) vary per sample.
        poles = self.fixed_poles.unsqueeze(0).expand(batch_size, -1)  # [batch_size, num_poles]

        # ---- Residues (TRAINABLE — the network's primary task) ----
        R_real_half = self.residues_real(h).view(batch_size, self.num_poles_half, self.num_ports, self.num_ports)
        R_imag_half = self.residues_imag(h).view(batch_size, self.num_poles_half, self.num_ports, self.num_ports)
        # Mirror for conjugate symmetry — ensures the sum of residue contributions
        # produces a real-valued impulse response when paired with conjugate poles.
        R_real = torch.cat([R_real_half, R_real_half], dim=1)
        R_imag = torch.cat([R_imag_half, -R_imag_half], dim=1)
        residues = torch.complex(R_real, R_imag)  # [batch_size, num_poles, num_ports, num_ports]

        # ---- Direct term ----
        # Real-valued feedthrough matrix — cast to complex64 for consistent arithmetic
        d_term_real = self.d_term_real(h).view(batch_size, self.num_ports, self.num_ports)
        d_term = torch.complex(d_term_real, torch.zeros_like(d_term_real))

        return poles, residues, d_term
    
    def enforce_passivity(self, S_matrix):
        """
        Hard Passivity Enforcement Layer (PEL) using Singular Value Decomposition.
        Guarantees that the network cannot physically generate energy.

        ONLY called during inference (model.eval()). During training, passivity
        is encouraged via a soft penalty in the loss function.
        """
        jitter = torch.eye(4, dtype=S_matrix.dtype, device=S_matrix.device).view(1, 1, 4, 4) * 1e-7
        U, S_vals, Vh = torch.linalg.svd(S_matrix + jitter)
        max_sv = torch.max(S_vals, dim=-1)[0]
        scale = torch.clamp(max_sv, min=1.0 + 1e-6)
        scale = scale.view(S_matrix.shape[0], S_matrix.shape[1], 1, 1)
        return S_matrix / scale

    def predict_frequency_response(self, poles, residues, d_term, frequencies_hz):
        """
        Evaluates the rational transfer function S(s) = sum(Rn / (s - Pn)) + D
        analytically at each frequency point.

        The computation is identical to v1/v2 — the only difference is that poles
        are now fixed buffers rather than network outputs. Autograd still flows
        through the residues and D term for Jacobian yield loss in Phase 3.

        Args:
            poles:          [batch_size, num_poles] complex64 — fixed, from buffer
            residues:       [batch_size, num_poles, num_ports, num_ports] complex64
            d_term:         [batch_size, num_ports, num_ports] complex64
            frequencies_hz: [num_freqs] float — frequency axis in Hz
        Returns:
            s_matrix:       [batch_size, num_freqs, num_ports, num_ports] complex64
        """
        batch_size = poles.shape[0]
        num_freqs = frequencies_hz.shape[0]
        # Scale frequencies down to GHz for numerical stability
        f_norm = frequencies_hz.float().to(poles.device) / 1e9

        # Build Laplace variable s = j*omega
        omega = 2 * torch.pi * f_norm
        s = torch.complex(torch.zeros_like(omega), omega)  # [num_freqs]

        # Expand for broadcasting: [batch, freq, pole]
        s_exp = s.view(1, num_freqs, 1)
        poles_exp = poles.view(batch_size, 1, self.num_poles)

        # Compute denominator (s - Pn) with numerical stabilisation
        denom = s_exp - poles_exp
        denom_abs = torch.abs(denom)
        denom_stable = torch.where(
            denom_abs < 1e-10,
            torch.complex(
                torch.full_like(denom.real, 1e-10),
                torch.zeros_like(denom.imag)
            ),
            denom
        )

        # Compute rational function via einsum
        inv_denom = 1.0 / denom_stable  # [batch_size, num_freqs, num_poles]
        s_matrix = torch.einsum('bfp, bpij -> bfij', inv_denom, residues)

        # Add direct feedthrough term D
        d_expanded = d_term.view(batch_size, 1, self.num_ports, self.num_ports)
        s_matrix = s_matrix + d_expanded

        # Conditional passivity enforcement (eval mode only)
        if not self.training:
            s_matrix = self.enforce_passivity(s_matrix)

        return s_matrix
    
    def verify_physics_constraints(self, poles, residues):
        """
        Diagnostic method to verify all three physical constraints are satisfied.
        Called after training to confirm the model produces physically valid outputs.
        """
        # Causality: all poles must be in the left half of the complex plane.
        # With fixed poles this should ALWAYS pass — it's guaranteed by construction.
        is_causal = torch.all(poles.real < 0).item()
        
        # Conjugate symmetry check
        p_half_1 = poles[:, :self.num_poles_half]
        p_half_2 = poles[:, self.num_poles_half:]
        is_symmetric = torch.allclose(p_half_1, p_half_2.conj(), atol=1e-5)

        # Passivity check via residue eigenvalues
        R_real = residues.real
        R_sym = (R_real + R_real.transpose(-1, -2)) / 2.0
        
        min_eigval = float('inf')
        is_passive = True
        for b in range(R_sym.shape[0]):
            for pole_idx in range(R_sym.shape[1]):
                eigvals = torch.linalg.eigvalsh(R_sym[b, pole_idx])
                min_eigval = min(min_eigval, eigvals.min().item())
                if eigvals.min().item() < -1e-6:
                    is_passive = False
        return {
            "causality_preserved": is_causal,
            "conjugate_symmetry_preserved": is_symmetric,
            "passivity_preserved": is_passive,
            "min_residue_eigenvalue": min_eigval
        }

    def __repr__(self):
        return (
            f"RationalNet("
            f"num_poles={self.num_poles} [FIXED], "
            f"num_ports={self.num_ports}, "
            f"hidden_dim={self.hidden_dim}, "
            f"trainable_params=residues+D+backbone, "
            f"residual_blocks=3)"
        )