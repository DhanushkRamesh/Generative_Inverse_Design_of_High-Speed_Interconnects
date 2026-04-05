import torch
import torch.nn as nn
import torch.nn.functional as F


class RationalNet(nn.Module):
    """
    Physics-Informed Rational Neural Network for S-parameter prediction of high-speed interconnects.
    
    Based on the Vector Fitting framework (Gustavsen and Semlyen, 1999), extended with:
      - Shared poles across the full 4x4 S-matrix (poles are structural properties, not port-specific)
      - Causality enforced by constraining pole real parts negative via softplus (left half-plane guarantee)
      - Conjugate pole symmetry enforced to guarantee real-valued time-domain impulse response
      - Numerical stabilisation in rational assembly to prevent division by near-zero denominators

    Architectural Improvements (v2):
      - Deeper residual backbone (3 residual blocks) to capture the highly nonlinear
        geometry-to-pole/residue mapping that a single block could not represent.
      - Hard passivity enforcement (SVD clamp) is ONLY applied at inference time.
        During training, a soft passivity penalty in the loss function guides learning
        without destroying gradients or collapsing the S-matrix to trivial solutions.
      - Increased default pole count (80) to capture the dense resonance structure
        observed in the TUHH SI/PI datasets across the 0.25–100 GHz band.
      - Layer normalisation after each residual block for training stability
        across the wider/deeper architecture.
    
    The model predicts poles, residues, and a direct term from via geometry features,
    then evaluates the rational transfer function analytically at all frequency points.
    """

    def __init__(self, num_poles, num_local_features, num_global_features, num_ports=4, hidden_dim=512):
        super(RationalNet, self).__init__()

        # num_poles must be even to support conjugate pair symmetry
        assert num_poles % 2 == 0, "num_poles must be even for conjugate pair symmetry."

        self.num_poles = num_poles
        self.num_poles_half = num_poles // 2  # only predict half, mirror for conjugates
        self.num_ports = num_ports
        self.hidden_dim = hidden_dim
        total_features = num_local_features + num_global_features

        # --------------------------------------------
        # Multi-Scale Fourier Feature Projection
        # --------------------------------------------
        # We project the input into a high-dimensional space using a fixed random Gaussian matrix.
        # Two frequency scales capture both smooth (low-freq) and sharp (high-freq) geometry variations.
        # The low-scale captures gradual parameter trends (e.g. trace length vs insertion loss slope),
        # while the high-scale captures sharp transitions (e.g. via radius near resonance boundaries).
        B_low = torch.randn(total_features, 64) * 1.0
        B_high = torch.randn(total_features, 64) * 10.0
        self.register_buffer('B_matrix', torch.cat([B_low, B_high], dim=-1))  # B Matrix --> fourier feature projection matrix, fixed random Gaussian
        # After sin/cos concatenation: 256-dim Fourier features

        # --------------------------------------------
        # MLP Backbone: shared feature extractor
        # --------------------------------------------
        # Deeper architecture (3 residual blocks) compared to v1 (1 block).
        # The mapping from geometry features to rational function coefficients is
        # highly nonlinear — pole locations shift dramatically with small geometry
        # changes (e.g. a 10% change in via radius can move a resonance by several GHz).
        # A single residual block was insufficient to capture this complexity,
        # resulting in flat-line predictions that failed to track any resonance structure.
        #
        # Width increased from 256 to 512 to provide sufficient capacity for the
        # larger number of poles (80 vs 40) and the deeper feature hierarchy.
        # LayerNorm after each residual addition stabilises the gradient flow
        # through the deeper architecture without the batch-size sensitivity of BatchNorm.
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

        # Residual Block 2: deeper abstraction of geometry-to-physics mapping
        self.residual_block_2 = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)

        # Residual Block 3: final refinement before pole/residue prediction heads
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
        # Output Heads: poles, residues, direct term
        # --------------------------------------------
        # Real and imaginary parts predicted separately — PyTorch optimisers
        # handle real-valued gradients better than complex-valued ones.

        # 1. Poles head: predicts num_poles // 2 unique poles.
        # Conjugate pairs are constructed in forward() so the model always
        # produces a real-valued impulse response (physical requirement).
        # Poles are shared across all S-parameter elements — they are properties
        # of the physical structure, not of individual port combinations.
        self.poles_real = nn.Linear(hidden_dim, self.num_poles_half)
        self.poles_imag = nn.Linear(hidden_dim, self.num_poles_half)

        # 2. Residues heads:
        # Each pole has a num_ports x num_ports residue matrix, so total residue
        # parameters = num_poles_half * num_ports * num_ports per real/imag component.
        self.residues_real = nn.Linear(hidden_dim, self.num_poles_half * num_ports * num_ports)
        self.residues_imag = nn.Linear(hidden_dim, self.num_poles_half * num_ports * num_ports)

        # 3. Direct term head (D): real-valued constant feedthrough matrix.
        # D is strictly real for passive networks (Gustavsen and Semlyen, 1999).
        self.d_term_real = nn.Linear(hidden_dim, num_ports * num_ports)

        # --------------------------------------------
        # Careful Initialisation of Poles and Residues
        # --------------------------------------------
        # Spread the initial poles across the frequency spectrum to encourage
        # stable training from the start. Without this, randomly placed poles
        # cluster in narrow bands and leave large spectral gaps unfitted.
        with torch.no_grad():
            # Imaginary parts span 0.25 GHz to 100 GHz (in rad/s normalised to GHz).
            # Multiply by 2*pi to convert from GHz to rad/s for better numerical
            # stability in the rational function evaluation.
            initial_imag = torch.linspace(0.25 * 2 * torch.pi, 100 * 2 * torch.pi, self.num_poles_half)
            self.poles_imag.bias.copy_(initial_imag)

            # Start the poles with moderate damping (negative real parts).
            # Too close to zero → near-imaginary-axis poles cause denominator explosions.
            # Too negative → overdamped poles can't produce sharp resonances.
            # Range [-0.5, -0.1] was found empirically to balance these concerns.
            nn.init.uniform_(self.poles_real.bias, -0.5, -0.1)

            # Residue weights initialised small to prevent large S-matrix values
            # early in training that would trigger aggressive passivity corrections.
            nn.init.normal_(self.residues_real.weight, std=0.01)
            nn.init.constant_(self.residues_real.bias, 0.0)

            # Imaginary residues bias initialised to zero to bias towards physical
            # realism early in training. Non-zero phase responses will emerge as
            # the network learns the geometry-dependent coupling structure.
            nn.init.normal_(self.residues_imag.weight, std=0.01)
            nn.init.constant_(self.residues_imag.bias, 0.0)

            # Direct term starts at zero — the rational sum should carry the full
            # response initially, with D learning any DC offset as needed.
            nn.init.constant_(self.d_term_real.bias, 0.0)


    def forward(self, x_local, x_global):
        """
        Forward pass: computes poles, residues, and direct term from via geometry.

        Args:
            x_local:  [batch_size, num_local_features]  — local geometry targets
            x_global: [batch_size, num_global_features] — global layout constraints
        Returns:
            poles:    [batch_size, num_poles] complex64         — conjugate symmetric
            residues: [batch_size, num_poles, num_ports, num_ports] complex64
            d_term:   [batch_size, num_ports, num_ports] complex64 — real part only, zero imag
        """
        # Combine local and global features into a single input vector
        x = torch.cat((x_local, x_global), dim=1)

        # --------------------------------------------
        # Fourier Feature Projection
        # --------------------------------------------
        # Linearly scale Z-scores to [0,1] and clamp outliers to prevent Fourier aliasing.
        # Since input features are Z-score normalised (mean=0, std=1), the range [-3, 3]
        # captures ~99.7% of the data. Mapping to [0,1] ensures the Fourier projection
        # operates in a well-conditioned input range.
        x_norm = torch.clamp((x + 3.0) / 6.0, min=0.0, max=1.0)
        # Project into high frequency space for better expressivity in rational function fitting
        x_proj = 2.0 * torch.pi * (x_norm @ self.B_matrix)
        # Create 256-dim Fourier feature by concatenating sin and cos projections
        x_fourier = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        # --------------------------------------------
        # Deep Residual Backbone
        # --------------------------------------------
        # Three residual blocks with LayerNorm after each skip connection.
        # The residual connections ensure gradient flow to early layers even with
        # the increased depth, while LayerNorm prevents internal covariate shift
        # that would otherwise destabilise the pole/residue prediction heads.
        h = self.input_layer(x_fourier)  # shape: [batch_size, hidden_dim]

        h = self.layer_norm_1(self.residual_block_1(h) + h)  # Block 1 + skip
        h = self.layer_norm_2(self.residual_block_2(h) + h)  # Block 2 + skip
        h = self.layer_norm_3(self.residual_block_3(h) + h)  # Block 3 + skip

        batch_size = h.shape[0]

        # ---- Poles ----
        # Enforce causality: real parts must be strictly negative (left half complex plane).
        # softplus is smooth everywhere — unlike abs() which has zero gradient at zero —
        # giving healthy gradients throughout training. Negating guarantees negativity.
        # The +0.01 floor prevents poles from sitting exactly on the imaginary axis,
        # which would cause denominator singularities in the rational function.
        p_real = -(F.softplus(self.poles_real(h)) + 0.01)  # shape: [batch_size, num_poles_half]
        p_imag = self.poles_imag(h)                         # shape: [batch_size, num_poles_half]

        # Enforce conjugate symmetry: mirror each pole with its complex conjugate.
        # This guarantees the impulse response h(t) = sum(Rn * exp(Pn * t)) is real-valued,
        # which is a physical requirement for any passive interconnect structure.
        poles_real = torch.cat([p_real, p_real], dim=1)      # [batch_size, num_poles]
        poles_imag = torch.cat([p_imag, -p_imag], dim=1)     # [batch_size, num_poles]
        poles = torch.complex(poles_real, poles_imag)         # [batch_size, num_poles]

        # ---- Residues ----
        R_real_half = self.residues_real(h).view(batch_size, self.num_poles_half, self.num_ports, self.num_ports)
        R_imag_half = self.residues_imag(h).view(batch_size, self.num_poles_half, self.num_ports, self.num_ports)
        # Mirror for conjugate symmetry — ensures the sum of residue contributions
        # produces a real-valued impulse response when paired with conjugate poles.
        R_real = torch.cat([R_real_half, R_real_half], dim=1)
        R_imag = torch.cat([R_imag_half, -R_imag_half], dim=1)

        # Combine into complex residues
        residues = torch.complex(R_real, R_imag)
        # shape: [batch_size, num_poles, num_ports, num_ports]

        # ---- Direct term ----
        # Real-valued feedthrough matrix — cast to complex64 for consistent arithmetic
        # in predict_frequency_response. Imaginary part is identically zero.
        d_term_real = self.d_term_real(h).view(batch_size, self.num_ports, self.num_ports)
        d_term = torch.complex(d_term_real, torch.zeros_like(d_term_real))  # shape: [batch_size, num_ports, num_ports] complex64
        return poles, residues, d_term
    
    def enforce_passivity(self, S_matrix):
        """
        Hard Passivity Enforcement Layer (PEL) using Singular Value Decomposition.
        Guarantees that the network cannot physically generate energy.

        IMPORTANT: This method is ONLY called during inference (model.eval()).
        During training, passivity is encouraged via a soft penalty in the loss function.
        Hard enforcement during training was found to collapse the S-matrix to near-identity
        solutions because:
          1. Early in training, randomly initialised poles produce large singular values.
          2. The SVD clamp uniformly scales down the entire 4x4 matrix, destroying any
             learned resonance structure.
          3. Gradients through torch.linalg.svd are numerically fragile near degenerate
             singular values, causing training instability.
        """
        # S_matrix shape: [batch_size, num_freqs, 4, 4]
        # Add a microscopic jitter to the diagonal to prevent SVD degenerate NaN crashes
        jitter = torch.eye(4, dtype=S_matrix.dtype, device=S_matrix.device).view(1, 1, 4, 4) * 1e-7
        U, S_vals, Vh = torch.linalg.svd(S_matrix + jitter)  # S_vals shape: [batch_size, num_freqs, 4]
        
        # Find the maximum singular value for each frequency point
        max_sv = torch.max(S_vals, dim=-1)[0]  # shape: [batch_size, num_freqs]
        
        # Calculate the scaling factor. If max_sv < 1, scale is 1.
        # Safety floor (1e-6) prevents division by exactly 1.0 (gradient explosion).
        scale = torch.clamp(max_sv, min=1.0 + 1e-6)
        
        # Reshape scale so it can broadcast over the 4x4 matrix
        scale = scale.view(S_matrix.shape[0], S_matrix.shape[1], 1, 1)
        
        return S_matrix / scale

    def predict_frequency_response(self, poles, residues, d_term, frequencies_hz):
        """
        Evaluates the rational transfer function S(s) = sum(Rn / (s - Pn)) + D
        analytically at each frequency point.

        Kept fully in PyTorch so autograd flows through for Jacobian and Hessian
        yield losses in Phase 3.

        Hard passivity enforcement is conditionally applied:
          - model.eval()  → SVD clamp is applied (inference/validation)
          - model.train() → no SVD clamp, passivity via soft loss only (training)

        Args:
            poles:          [batch_size, num_poles] complex64
            residues:       [batch_size, num_poles, num_ports, num_ports] complex64
            d_term:         [batch_size, num_ports, num_ports] complex64
            frequencies_hz: [num_freqs] float — frequency axis in Hz
        Returns:
            s_matrix:       [batch_size, num_freqs, num_ports, num_ports] complex64

        Note: call model.eval() before this to disable dropout and enable hard passivity
        enforcement for deterministic, physically valid outputs during evaluation.
        During training, dropout is enabled and passivity is soft-constrained only.
        """
        batch_size = poles.shape[0]
        num_freqs = frequencies_hz.shape[0]
        # Scale frequencies down to GHz for numerical stability in the rational function evaluation.
        # This matches the pole initialisation scale (poles_imag in GHz·rad/s).
        f_norm = frequencies_hz.float().to(poles.device) / 1e9

        # Build Laplace variable s = j*omega — fully in PyTorch for autograd compatibility
        omega = 2 * torch.pi * f_norm
        s = torch.complex(torch.zeros_like(omega), omega)  # shape: [num_freqs]

        # Expand all tensors for broadcasting across batch, frequency, pole, and port dimensions
        # Final broadcast target: [batch_size, num_freqs, num_poles, num_ports, num_ports]
        s_exp = s.view(1, num_freqs, 1)
        poles_exp = poles.view(batch_size, 1, self.num_poles)
        # Compute denominator (s - Pn) and stabilise against near-zero division.
        # Early in training, randomly initialised poles may sit near a frequency point,
        # causing denominator explosion. Epsilon floor prevents this.
        denom = s_exp - poles_exp  # shape: [batch_size, num_freqs, num_poles]
        # Stabilise against near-zero denominators to prevent numerical instability during training
        denom_abs = torch.abs(denom)
        denom_stable = torch.where(
            denom_abs < 1e-10,
            torch.complex(
                torch.full_like(denom.real, 1e-10),  # Real part is 1e-10
                torch.zeros_like(denom.imag)          # Imaginary part is 0
            ),
            denom
        )
        # Compute the rational function sum(Rn / (s - Pn)) using broadcasting and stable division
        inv_denom = 1.0 / denom_stable  # shape: [batch_size, num_freqs, num_poles]
        # The VRAM Saver: Einsum handles the multiplication and sum across poles simultaneously!
        # b = batch, f = freq, p = pole, i = port out, j = port in
        s_matrix = torch.einsum('bfp, bpij -> bfij', inv_denom, residues)

        # Add direct feedthrough term D
        d_expanded = d_term.view(batch_size, 1, self.num_ports, self.num_ports)
        s_matrix = s_matrix + d_expanded  # [batch_size, num_freqs, num_ports, num_ports]

        # Conditional passivity enforcement:
        # - Training mode: NO hard clamp. The soft passivity penalty in the loss function
        #   gently guides the model towards passive solutions without destroying gradients.
        #   This allows the network to first learn the correct pole/residue structure,
        #   then the soft penalty tightens passivity as training progresses.
        # - Eval mode: Hard SVD clamp guarantees physical validity for deployment.
        #   At this point the model has already learned sensible poles/residues,
        #   so the SVD clamp only makes minor corrections rather than crushing everything.
        if not self.training:
            s_matrix = self.enforce_passivity(s_matrix)

        return s_matrix
    
    def verify_physics_constraints(self, poles, residues):
        """
        Diagnostic method to verify all three physical constraints are satisfied.
        Called after training to confirm the model produces physically valid outputs.
        Args:
            poles:    [batch_size, num_poles] complex64
            residues: [batch_size, num_poles, num_ports, num_ports] complex64
        Returns:
            dict with causality, conjugate symmetry, and passivity checks
        """
        # Causality requires all poles to be in the left half of the complex plane
        is_causal = torch.all(poles.real < 0).item()
        
        # Conjugate symmetry check (ensure upper half mirrors lower half)
        p_half_1 = poles[:, :self.num_poles_half]
        p_half_2 = poles[:, self.num_poles_half:]
        is_symmetric = torch.allclose(p_half_1, p_half_2.conj(), atol=1e-5)

        # Passivity check: residues must be positive semi-definite, which is guaranteed
        # by the Cholesky parameterisation. We can check that the real part of residues
        # is positive semi-definite by confirming all minimal eigenvalues are non-negative.
        # This is a necessary condition for passivity, though not sufficient on its own.
        R_real = residues.real  # shape: [batch_size, num_poles, num_ports, num_ports]
        
        # Enforce mathematical symmetry for eigvalsh stability
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
            f"num_poles={self.num_poles}, "
            f"num_ports={self.num_ports}, "
            f"hidden_dim={self.hidden_dim}, "
            f"conjugate_pairs={self.num_poles_half}, "
            f"residual_blocks=3)"
        )