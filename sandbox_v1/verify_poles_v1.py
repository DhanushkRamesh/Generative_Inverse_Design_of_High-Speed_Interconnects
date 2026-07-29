import sys
from pathlib import Path
import torch
import numpy as np
import math

def main():
    print("=== VFIT BASIS LIE DETECTOR (LINEAR LEAST SQUARES) ===")
    sandbox_root = Path(__file__).resolve().parent
    
    dataset_path = sandbox_root.parent / "data" / "processed" / "Universal-Diff-SI-Array" / "diff_pair_dataset.pt"
    poles_path = sandbox_root / "data" / "universal_pole_basis.pt"
    
    data = torch.load(dataset_path, weights_only=False)
    poles = torch.load(poles_path, weights_only=False).numpy()
    
    freqs = data["frequencies"].numpy()
    Y_real, Y_imag = data["Y_real"].numpy(), data["Y_imag"].numpy()
    
    # Pick a random physical sample (Sample 0)
    S_target = Y_real[0] + 1j * Y_imag[0]  # Shape: (401, 4, 4)
    
    N = len(poles)
    F = len(freqs)
    w_scale = 2.0 * math.pi * 100e9
    
    s_hat = 1j * (freqs / 100e9)
    p_hat = poles / w_scale
    
    # Build the linear basis matrix Phi
    # S(s) = R_re * [1/(s-p) + 1/(s-p*)] + R_im * [j/(s-p) - j/(s-p*)] + D
    Phi = np.zeros((F, 2 * N + 1), dtype=complex)
    
    for k in range(N):
        term_pos = 1.0 / (s_hat - p_hat[k])
        term_neg = 1.0 / (s_hat - np.conj(p_hat[k]))
        
        Phi[:, k] = term_pos + term_neg            # R_re basis
        Phi[:, N + k] = 1j * (term_pos - term_neg) # R_im basis
        
    Phi[:, 2 * N] = 1.0 # D basis
    
    # We need the coefficients (Residues and D) to be strictly REAL numbers
    # So we stack the real and imaginary equations vertically
    Phi_real = np.vstack([np.real(Phi), np.imag(Phi)]) # Shape: (802, 81)
    
    S_pred = np.zeros_like(S_target)
    
    print("Solving exact closed-form math for 4x4 matrix...")
    for i in range(4):
        for j in range(4):
            S_ij = S_target[:, i, j]
            S_ij_stacked = np.hstack([np.real(S_ij), np.imag(S_ij)])
            
            # Linear Least Squares (Flawless mathematical fit)
            X, residuals, rank, s = np.linalg.lstsq(Phi_real, S_ij_stacked, rcond=None)
            
            # Reconstruct the prediction
            S_pred[:, i, j] = Phi @ X

    # Calculate dB MAE for Sdd11 and Sdd21
    p_db = 20 * np.log10(np.clip(np.abs(S_pred), 1e-5, None))
    t_db = 20 * np.log10(np.clip(np.abs(S_target), 1e-5, None))
    
    s11_mae = np.mean(np.abs(p_db[:, 0, 0] - t_db[:, 0, 0]))
    s21_mae = np.mean(np.abs(p_db[:, 1, 0] - t_db[:, 1, 0]))
    
    print("\n--- RESULTS ---")
    print(f"S11 (Return Loss) MAE:    {s11_mae:.3f} dB")
    print(f"S21 (Insertion Loss) MAE: {s21_mae:.3f} dB")
    
    if s21_mae > 2.0:
        print("\n🚨 VERDICT: YOUR GUT WAS RIGHT. The VFIT poles are garbage.")
        print("The 40 poles we extracted mathematically cannot represent this dataset.")
    else:
        print("\n✅ VERDICT: The poles are perfect. The Neural Network was the bottleneck.")

if __name__ == "__main__":
    main()