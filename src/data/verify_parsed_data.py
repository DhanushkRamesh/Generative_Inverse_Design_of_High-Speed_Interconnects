# src/data/verify_parsed_data.py
import torch
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import sys

# utils for physics conversions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.physics_utils import convert_to_mixed_mode, enforce_reciprocity, s_to_db, unscale_tensor

def verify_dataset(pt_path, raw_base_dir):
    print(f"Loading processed dataset: {pt_path}")
    data = torch.load(pt_path, weights_only=False)
    
    # Pick a random index to verify
    total_samples = data['X'].shape[0]
    idx = random.randint(0, total_samples - 1)
    sim_id = data['sim_ids'][idx]
    
    print(f"Verifying Index: {idx} | Simulation ID: {sim_id}")

    # Reconstruct Tensor Data
    # Our processed tensors are already in Mixed-Mode!
    freqs = data['frequencies'].numpy()
    Y_real = data['Y_real'][idx].numpy()
    Y_imag = data['Y_imag'][idx].numpy()
    S_tensor_mm = Y_real + 1j * Y_imag
    
    #Load Raw Ground Truth
    # Use the sim_id to find the folder
    search_path = os.path.join(raw_base_dir, "variation", sim_id, "*.s*p")
    import glob
    raw_files = glob.glob(search_path)
    
    if not raw_files:
        print(f"Error: Could not find raw file for {sim_id}")
        return

    ntwk_raw = rf.Network(raw_files[0])
    
    # Apply the same Slicing Logic used in Parser
    num_ports = ntwk_raw.number_of_ports
    half = num_ports // 2
    port_idx = [0, 1, half, half+1]
    
    # Extract the same 4x4 from the raw high-port network (Single-Ended)
    ntwk_raw_4x4_se = ntwk_raw.s[:, port_idx, :][:, :, port_idx]
    # enforce reciprocity to kill numerical noise before conversion exactly what was done in the parser
    ntwk_raw_4x4_se = enforce_reciprocity(ntwk_raw_4x4_se)
    # 5. Convert Raw Data to Mixed-Mode for a fair 1-to-1 comparison
    ntwk_raw_4x4_mm = convert_to_mixed_mode(ntwk_raw_4x4_se)

    # 6. Plotting Comparison (Sdd11 and Sdd21)
    plt.figure(figsize=(12, 5))
    
    # Differential Return Loss (Sdd11) -> Index [:, 0, 0]
    plt.subplot(1, 2, 1)
    plt.plot(ntwk_raw.f/1e9, s_to_db(ntwk_raw_4x4_mm[:, 0, 0]), 'r-', label='Raw (Converted to MM)', linewidth=3, alpha=0.5)
    plt.plot(freqs/1e9, s_to_db(S_tensor_mm[:, 0, 0]), 'k--', label='Tensor Reconstruction')
    plt.title(f'Diff Return Loss (Sdd11) - {sim_id}')
    plt.ylabel('Magnitude (dB)'); plt.xlabel('Frequency (GHz)')
    plt.grid(True); plt.legend()

    # Differential Insertion Loss (Sdd21) -> Index [:, 1, 0]
    plt.subplot(1, 2, 2)
    plt.plot(ntwk_raw.f/1e9, s_to_db(ntwk_raw_4x4_mm[:, 1, 0]), 'b-', label='Raw (Converted to MM)', linewidth=3, alpha=0.5)
    plt.plot(freqs/1e9, s_to_db(S_tensor_mm[:, 1, 0]), 'k--', label='Tensor Reconstruction')
    plt.title(f'Diff Insertion Loss (Sdd21) - {sim_id}')
    plt.ylabel('Magnitude (dB)'); plt.xlabel('Frequency (GHz)')
    plt.grid(True); plt.legend()

    plt.tight_layout()
    plt.show()

    # 7. Geometric Verification
    radius_idx = data['feature_names'].index('VIA_RADIUS')
    scaled_radius = data['X'][idx, radius_idx].item()
    
    # Un-scale: Z-score normalization reversal - > physical_value = (scaled_value * std) + mean
    physical_radius = unscale_tensor(
        scaled_radius, 
        data['X_mean'][radius_idx], 
        data['X_std'][radius_idx]
    ).item()
    
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