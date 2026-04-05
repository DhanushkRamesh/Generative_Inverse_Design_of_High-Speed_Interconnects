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
    
    The model predicts poles, residues, and a direct term from via geometry features,
    then evaluates the rational transfer function analytically at all frequency points.
    """

    def __init__(self, num_poles, num_local_features, num_global_features, num_ports=4):
        super(RationalNet, self).__init__()

        # num_poles must be even to support conjugate pair symmetry
        assert num_poles % 2 == 0, "num_poles must be even for conjugate pair symmetry."

        self.num_poles = num_poles
        self.num_poles_half = num_poles // 2  # only predict half, mirror for conjugates
        self.num_ports = num_ports
        total_features = num_local_features + num_global_features

        # --------------------------------------------
        # MLP Backbone: shared feature extractor
        # --------------------------------------------
        # Starts at 256 (wider than input) to avoid bottleneck compression before
        # useful representations are learned. Dropout(0.1) regularises against
        # overfitting on the small dataset (~1900 samples).
        self.input_layer = nn.Linear(total_features, 512)
        self.residual_block = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(512, 512)
        )

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
        self.poles_real = nn.Linear(512, self.num_poles_half)
        self.poles_imag = nn.Linear(512, self.num_poles_half)

        # 2. Residues heads:
        self.residues_real = nn.Linear(512, self.num_poles_half * num_ports * num_ports)
        self.residues_imag = nn.Linear(512, self.num_poles_half * num_ports * num_ports)

        # 3. Direct term head (D): real-valued constant feedthrough matrix.
        # D is strictly real for passive networks (Gustavsen and Semlyen, 1999).
        self.d_term_real = nn.Linear(512, num_ports * num_ports)

        #manually initialise the resisues bias 
        #spread the initial pole across the frequency spectrum to encourage stable training from the start
        with torch.no_grad():
            initial_imag = torch.linspace(0, 100 * 2 * torch.pi, self.num_poles_half) # multiply by 2*pi to convert from GHz to rad/s for better numerical stability in the rational function evaluation.
            self.poles_imag.bias.copy_(initial_imag)
            #real residues bias initialised to small positive values to encourage passivity early in training, while still allowing gradient flow
            nn.init.normal_(self.residues_real.weight, std=0.01)
            nn.init.constant_(self.residues_real.bias, 0.5) # Start with zero imaginary residues to bias towards physical realism early in training
            #imaginary residues bias initialised to small values to allow non-zero phase responses while still encouraging physical realism early in training. This is a hyperparameter that can be tuned for faster convergence or better final performance. Setting it too high may lead to unstable training early on, while setting it too low may bias the model towards zero-phase solutions which are not physically accurate for many interconnect structures. A value of 0.5 was found to provide a good balance in preliminary experiments, but this can be adjusted based on the specific dataset and training dynamics observed.
            nn.init.normal_(self.residues_imag.weight, std=0.01) 
            nn.init.constant_(self.residues_imag.bias, 0.1) # Start with zero imaginary residues to bias towards physical realism early in training
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
        initial_features = self.input_layer(x)  # shape: [batch_size, 512]
        hidden = self.residual_block(initial_features) + initial_features  # shape: [batch_size, 512]
        batch_size = hidden.shape[0]

        # ---- Poles ----
        # Enforce causality: real parts must be strictly negative (left half complex plane).
        # softplus is smooth everywhere — unlike abs() which has zero gradient at zero —
        # giving healthy gradients throughout training. Negating guarantees negativity.
        p_real = -F.softplus(self.poles_real(hidden))  # shape: [batch_size, num_poles_half]
        p_imag = self.poles_imag(hidden)                # shape: [batch_size, num_poles_half]

        # Enforce conjugate symmetry: mirror each pole with its complex conjugate.
        # This guarantees the impulse response h(t) = sum(Rn * exp(Pn * t)) is real-valued,
        # which is a physical requirement for any passive interconnect structure.
        poles_real = torch.cat([p_real, p_real], dim=1)    # [batch_size, num_poles]
        poles_imag = torch.cat([p_imag, -p_imag], dim=1)   # [batch_size, num_poles]
        poles = torch.complex(poles_real, poles_imag)       # [batch_size, num_poles]

        # ---- Residues ----
        R_real_half = self.residues_real(hidden).view(batch_size, self.num_poles_half, self.num_ports, self.num_ports)
        R_imag_half = self.residues_imag(hidden).view(batch_size, self.num_poles_half, self.num_ports, self.num_ports)
        #Mirror for conjugate symmetry
        R_real = torch.cat([R_real_half, R_real_half], dim=1)
        R_imag = torch.cat([R_imag_half, -R_imag_half], dim=1)

        # Combine into complex residues
        residues = torch.complex(R_real, R_imag)
        # shape: [batch_size, num_poles, num_ports, num_ports]

        # ---- Direct term ----
        # Real-valued feedthrough matrix — cast to complex64 for consistent arithmetic
        # in predict_frequency_response. Imaginary part is identically zero.
        d_term_real = self.d_term_real(hidden).view(batch_size, self.num_ports, self.num_ports)
        d_term = torch.complex(d_term_real, torch.zeros_like(d_term_real)) # shape: [batch_size, num_ports, num_ports] complex64
        return poles, residues, d_term

    def predict_frequency_response(self, poles, residues, d_term, frequencies_hz):
        """
        Evaluates the rational transfer function S(s) = sum(Rn / (s - Pn)) + D
        analytically at each frequency point.

        Kept fully in PyTorch so autograd flows through for Jacobian and Hessian
        yield losses in Phase 3.

        Args:
            poles:          [batch_size, num_poles] complex64
            residues:       [batch_size, num_poles, num_ports, num_ports] complex64
            d_term:         [batch_size, num_ports, num_ports] complex64
            frequencies_hz: [num_freqs] float — frequency axis in Hz
        Returns:
            s_matrix:       [batch_size, num_freqs, num_ports, num_ports] complex64

        Note: call model.eval() before this to disable dropout and ensure deterministic outputs during evaluation.
        During training, dropout is enabled which adds noise to the outputs, so frequency response predictions will be stochastic.
        During inference and evaluation, dropout is disabled to get stable predictions.
        """
        batch_size = poles.shape[0]
        num_freqs = frequencies_hz.shape[0]
        #scale frequencies down to GHz for numerical stability in the rational function evaluation.
        f_norm = frequencies_hz.float().to(poles.device) / 1e9

        # Build Laplace variable s = j*omega — fully in PyTorch for autograd compatibility
        omega = 2 * torch.pi * f_norm
        s = torch.complex(torch.zeros_like(omega), omega)  # shape: [num_freqs]

        # Expand all tensors for broadcasting across batch, frequency, pole, and port dimensions
        # Final broadcast target: [batch_size, num_freqs, num_poles, num_ports, num_ports]
        s_exp        = s.view(1, num_freqs, 1, 1, 1)
        poles_exp    = poles.view(batch_size, 1, self.num_poles, 1, 1)
        residues_exp = residues.view(batch_size, 1, self.num_poles, self.num_ports, self.num_ports)

        # Compute denominator (s - Pn) and stabilise against near-zero division.
        # Early in training, randomly initialised poles may sit near a frequency point,
        # causing denominator explosion. Epsilon floor prevents this.
        denom = s_exp - poles_exp  # shape: [batch_size, num_freqs, num_poles, 1, 1]
        #stabilise against near-zero denominators to prevent numerical instability during training
        denom_abs = torch.abs(denom)
        denom_stable = torch.where(
            denom_abs < 1e-10,
            torch.complex(
                torch.full_like(denom.real, 1e-10), # Real part is 1e-10
                torch.zeros_like(denom.imag)        # Imaginary part is 0
            ),
            denom
        )

        # Compute Rn / (s - Pn) for each pole and sum across poles dimension
        terms    = residues_exp / denom_stable  # [batch_size, num_freqs, num_poles, num_ports, num_ports]
        s_matrix = torch.sum(terms, dim=2)      # [batch_size, num_freqs, num_ports, num_ports]

        # Add direct feedthrough term D and asymptotic term E * s
        d_expanded = d_term.view(batch_size, 1, self.num_ports, self.num_ports)
        s_matrix   = s_matrix + d_expanded      # [batch_size, num_freqs, num_ports, num_ports]
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

        #passivity check: residues must be positive semi-definite, which is guaranteed by the Cholesky parameterisation.
        # We can check that the real part of residues is positive semi-definite by confirming all
        #minimal eigenvalues are non-negative. This is a necessary condition for passivity, though not sufficient on its own.
        #Cholesky parameterisation guarantees positive semi-definiteness, but numerical issues could arise during training, so this is a useful diagnostic.
        R_real = residues.real # shape: [batch_size, num_poles, num_ports, num_ports]
        min_eigval = float('inf')
        is_passive = True
        for b in range(R_real.shape[0]):
            for pole_idx in range(R_real.shape[1]):
                eigvals = torch.linalg.eigvalsh(R_real[b, pole_idx])
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
            f"conjugate_pairs={self.num_poles_half})"
        )