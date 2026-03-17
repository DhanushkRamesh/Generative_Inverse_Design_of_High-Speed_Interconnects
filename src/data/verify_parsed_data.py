import torch
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def verify_dataset(pt_path, raw_base_dir):
    print(f"Loading processed dataset: {pt_path}")
    data = torch.load(pt_path, weights_only=True)
    
    # 1. Pick a random index to verify
    total_samples = data['X'].shape[0]
    idx = random.randint(0, total_samples - 1)
    sim_id = data['sim_ids'][idx]
    
    print(f"Verifying Index: {idx} | Simulation ID: {sim_id}")

    # 2. Reconstruct Tensor Data
    # Convert [-1, 1] back to complex S-parameters
    freqs = data['frequencies'].numpy()
    Y_real = data['Y_real'][idx].numpy()
    Y_imag = data['Y_imag'][idx].numpy()
    S_tensor = Y_real + 1j * Y_imag
    
    # 3. Load Raw Ground Truth
    # Use the sim_id to find the folder
    search_path = os.path.join(raw_base_dir, "variation", sim_id, "*.s*p")
    import glob
    raw_files = glob.glob(search_path)
    
    if not raw_files:
        print(f"Error: Could not find raw file for {sim_id}")
        return

    ntwk_raw = rf.Network(raw_files[0])
    
    # 4. Apply the same Slicing Logic used in Parser
    num_ports = ntwk_raw.number_of_ports
    half = num_ports // 2
    port_idx = [0, 1, half, half+1]
    
    # Extract the same 4x4 from the raw high-port network
    ntwk_raw_4x4 = ntwk_raw.s[:, port_idx, :][:, :, port_idx]

    # 5. Plotting Comparison (S11 and S21)
    plt.figure(figsize=(12, 5))
    
    # Port 1 to Port 1 (Return Loss)
    plt.subplot(1, 2, 1)
    plt.plot(ntwk_raw.f/1e9, 20*np.log10(np.abs(ntwk_raw_4x4[:, 0, 0])), 'r-', label='Raw Ground Truth', linewidth=3, alpha=0.5)
    plt.plot(freqs/1e9, 20*np.log10(np.abs(S_tensor[:, 0, 0])), 'k--', label='Tensor Reconstruction')
    plt.title(f'Return Loss (S11) - {sim_id}')
    plt.ylabel('Magnitude (dB)'); plt.xlabel('Frequency (GHz)')
    plt.grid(True); plt.legend()

    # Port 1 to Port 3 (Insertion Loss / Through Path)
    # Note: Port 3 in our 4x4 matrix corresponds to 'half' in the raw matrix
    plt.subplot(1, 2, 2)
    plt.plot(ntwk_raw.f/1e9, 20*np.log10(np.abs(ntwk_raw_4x4[:, 2, 0])), 'b-', label='Raw Ground Truth', linewidth=3, alpha=0.5)
    plt.plot(freqs/1e9, 20*np.log10(np.abs(S_tensor[:, 2, 0])), 'k--', label='Tensor Reconstruction')
    plt.title(f'Insertion Loss (S31) - {sim_id}')
    plt.ylabel('Magnitude (dB)'); plt.xlabel('Frequency (GHz)')
    plt.grid(True); plt.legend()

    plt.tight_layout()
    plt.show()

    # 6. Geometric Verification (Optional but Recommended)
    radius_idx = data['feature_names'].index('VIA_RADIUS')
    scaled_radius = data['X'][idx, radius_idx].item()
    
    # Un-scale: Normalized [-1, 1] -> Physical
    r_min = data['X_min'][radius_idx].item()
    r_max = data['X_max'][radius_idx].item()
    physical_radius = ((scaled_radius + 1) / 2) * (r_max - r_min) + r_min
    
    print(f"Geometry Check (VIA_RADIUS):")
    print(f"   - Tensor (Unscaled): {physical_radius:.4f}")
    print(f"   - Check this against your parameter.csv for {sim_id}!")

if __name__ == "__main__":
    PROJ_ROOT = os.path.expanduser("~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects")
    
    # Test Array
    print("\n--- Testing ARRAY Dataset ---")
    array_pt = os.path.join(PROJ_ROOT, "data/processed/Universal-Diff-SI-Array/via_array_dataset.pt")
    array_raw = os.path.join(PROJ_ROOT, "data/raw/Universal-Diff-SI-Array")
    verify_dataset(array_pt, array_raw)
    
    # Test Link
    print("\n--- Testing LINK Dataset ---")
    link_pt = os.path.join(PROJ_ROOT, "data/processed/Universal-Diff-SI-Link/via_link_dataset.pt")
    link_raw = os.path.join(PROJ_ROOT, "data/raw/Universal-Diff-SI-Link")
    verify_dataset(link_pt, link_raw)