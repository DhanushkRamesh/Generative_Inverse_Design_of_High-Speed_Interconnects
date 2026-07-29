import torch
import numpy as np
import os
import matplotlib.pyplot as plt

def generate_synthetic_data(num_samples=2000, num_points=401):
    print("Generating Synthetic Data (With Ground Truth Poles)...")
    X = torch.rand(num_samples, 10)
    f_ghz = np.linspace(0.25, 100, num_points)
    s = 1j * 2 * np.pi * f_ghz
    s_tensor = torch.tensor(s, dtype=torch.complex64)
    
    Y_complex = torch.zeros((num_samples, num_points, 2), dtype=torch.complex64)
    
    # NEW: Arrays to store the exact ground truth parameters!
    # 3 resonances = 3 poles/residues per target
    P_gt = torch.zeros((num_samples, 2, 3), dtype=torch.complex64)
    C_gt = torch.zeros((num_samples, 2, 3), dtype=torch.complex64)
    
    for i in range(num_samples):
        H_s11 = torch.zeros(num_points, dtype=torch.complex64)
        H_s21 = torch.zeros(num_points, dtype=torch.complex64) # Keep it simple, pure rational
        
        for res in range(3):
            alpha = -(0.1 + X[i, res] * 2.0) 
            beta = 2 * np.pi * (10 + X[i, res+3] * 80)
            c_real = X[i, res+6] * 5.0
            c_imag = (X[i, res+1] - 0.5) * 5.0
            
            p1 = complex(alpha, beta)
            c1 = complex(c_real, c_imag)
            
            # Save ground truth for S11
            P_gt[i, 0, res] = p1
            C_gt[i, 0, res] = c1
            
            # Save ground truth for S21 (Inverted residues)
            P_gt[i, 1, res] = p1
            C_gt[i, 1, res] = -c1 
            
            H_s11 += (c1 / (s_tensor - p1)) + (np.conj(c1) / (s_tensor - np.conj(p1)))
            H_s21 += (-c1 / (s_tensor - p1)) + (np.conj(-c1) / (s_tensor - np.conj(p1)))
            
        Y_complex[i, :, 0] = H_s11
        Y_complex[i, :, 1] = H_s21

    Y_real = Y_complex.real.unsqueeze(2).repeat(1, 1, 2, 1) 
    Y_imag = Y_complex.imag.unsqueeze(2).repeat(1, 1, 2, 1)

    output_dir = os.path.expanduser("~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/data/processed/Synthetic-Link")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "synthetic_poles_dataset.pt")
    
    torch.save({
        'X_global': X[:, :5], 'X_local': X[:, 5:],
        'Y_real': Y_real, 'Y_imag': Y_imag,
        'P_gt': P_gt, 'C_gt': C_gt, # THE SECRET SAUCE
        'frequencies': torch.tensor(f_ghz * 1e9, dtype=torch.float32)
    }, save_path)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    generate_synthetic_data()