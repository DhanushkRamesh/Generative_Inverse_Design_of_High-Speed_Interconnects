import torch
import numpy as np
import os
import matplotlib.pyplot as plt

def generate_synthetic_data(num_samples=1000, num_points=401):
    print("Generating Synthetic Pole-Residue Dataset...")
    
    # 1. Generate Dummy "Geometry" Inputs (10 features, normalized 0 to 1)
    X = torch.rand(num_samples, 10)
    
    # Frequency array: 0.25 GHz to 100 GHz
    f_ghz = np.linspace(0.25, 100, num_points)
    s = 1j * 2 * np.pi * f_ghz # s = jw
    s_tensor = torch.tensor(s, dtype=torch.complex64)
    
    Y_complex = torch.zeros((num_samples, num_points, 2), dtype=torch.complex64)
    
    for i in range(num_samples):
        # We will generate 3 resonance phenomena (3 pairs of complex conjugate poles)
        H_s11 = torch.zeros(num_points, dtype=torch.complex64)
        H_s21 = torch.ones(num_points, dtype=torch.complex64) # Start with perfect transmission
        
        for resonance in range(3):
            # Map the dummy inputs X to physical pole/residue locations
            # Real part (alpha) must be negative for a passive/stable system
            alpha = -(0.1 + X[i, resonance] * 2.0) 
            
            # Imag part (beta) dictates the resonance frequency (e.g., 10 to 90 GHz)
            beta = 2 * np.pi * (10 + X[i, resonance+3] * 80)
            
            # Residues (Amplitude of the resonance)
            c_real = X[i, resonance+6] * 5.0
            c_imag = (X[i, resonance+1] - 0.5) * 5.0
            
            # Form the complex pole and residue
            p1 = complex(alpha, beta)
            p2 = np.conj(p1) # Complex conjugate pair
            c1 = complex(c_real, c_imag)
            c2 = np.conj(c1)
            
            # Apply the Pole-Residue Formula for Sdd11 (Reflection peaks)
            H_s11 += (c1 / (s_tensor - p1)) + (c2 / (s_tensor - p2))
            
            # Apply inverted formula for Sdd21 (Insertion loss dips)
            H_s21 -= (c1 / (s_tensor - p1)) + (c2 / (s_tensor - p2))
            
        # Add a little generic loss scaling to make it look like a real trace
        loss_baseline = torch.tensor(np.exp(-0.01 * f_ghz), dtype=torch.complex64)
        H_s21 = H_s21 * loss_baseline
            
        Y_complex[i, :, 0] = H_s11
        Y_complex[i, :, 1] = H_s21

    # Split into Real and Imaginary tensors to match your pipeline perfectly
    Y_real = Y_complex.real.unsqueeze(2) # Add dummy port dimensions
    Y_real = torch.cat([Y_real, Y_real], dim=2) # Shape: (Samples, 401, 2, 1) to mimic your [:, :, 1, 0] setup
    
    Y_imag = Y_complex.imag.unsqueeze(2)
    Y_imag = torch.cat([Y_imag, Y_imag], dim=2)

    # Save exactly like your parse_touchstone script
    output_dir = os.path.expanduser("~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/data/processed/Synthetic-Link")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "synthetic_poles_dataset.pt")
    
    # Create dummy local/global splits to appease your dataset loader
    torch.save({
        'X_global': X[:, :5],
        'X_local': X[:, 5:],
        'Y_real': Y_real,
        'Y_imag': Y_imag,
        'frequencies': torch.tensor(f_ghz * 1e9, dtype=torch.float32) # Save as Hz
    }, save_path)
    
    print(f"Saved to {save_path}")

    # Plot one to show you what it looks like
    mag_s21 = 20 * np.log10(np.abs(Y_complex[0, :, 1].numpy()) + 1e-12)
    plt.plot(f_ghz, mag_s21)
    plt.title("Synthetic Pole-Residue Sdd21 Curve")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid()
    plt.savefig("synthetic_sample.png")
    print("Saved sample plot to synthetic_sample.png")

if __name__ == "__main__":
    generate_synthetic_data()