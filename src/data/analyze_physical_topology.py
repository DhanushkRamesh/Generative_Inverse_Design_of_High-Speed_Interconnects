# src/data/topology_analysis.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

#functions to unscale features and plot topology impacts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#import from utils.physics import unscale_feature, s_to_db 
from utils.physics_utils import unscale_tensor, s_to_db

def plot_topology_trends(pt_path, proj_root, dataset_type="Array"):
    print(f"\n--- Generating {dataset_type.capitalize()} Array Topology Analysis ---")
    data = torch.load(pt_path, weights_only=False)
    
    X = data['X']
    #reconstructing the complex S-parameters from real and imaginary parts
    S_all = data['Y_real'].numpy() + 1j * data['Y_imag'].numpy()
    freqs = data['frequencies'].numpy() / 1e9  # Convert to GHz
    # The driver feature is NUM_PORTS for the array dataset and LENGTH for the link dataset
    driver_name = 'NUM_PORTS' if dataset_type == 'array' else 'LENGTH'
    feat_idx = data['feature_names'].index(driver_name)
    #get physical values for the driver feature by unscaling the normalized tensor values
    physical_vals = unscale_tensor(X[:, feat_idx], data['X_mean'][feat_idx], data['X_std'][feat_idx])
    
    # statistical split - dataset split int0 Low and High groups based on median value.
    median_val = torch.median(physical_vals)
    low_mask = (physical_vals <= median_val).numpy()
    high_mask = (physical_vals > median_val).numpy()
    #compute the mean S-parameters for the low and high groups
    #average absolute magnitudes first and then convert to dB
    S_low_mean_db = s_to_db(np.mean(np.abs(S_all[low_mask]), axis=0))
    S_high_mean_db = s_to_db(np.mean(np.abs(S_all[high_mask]), axis=0))

    # Plot
    plt.figure(figsize=(12, 5))
    plt.suptitle(f"Statistical Topology Impact: {driver_name} ({dataset_type.capitalize()} Dataset)", fontsize=14)
    
    # Sdd11 (Return Loss)
    plt.subplot(1, 2, 1)
    plt.plot(freqs, S_low_mean_db[:, 0, 0], 'b-', linewidth=2, label=f'Low {driver_name} (Mean)')
    plt.plot(freqs, S_high_mean_db[:, 0, 0], 'r--', linewidth=2, label=f'High {driver_name} (Mean)')
    plt.title('Return Loss (Sdd11)')
    plt.xlabel('Frequency (GHz)'); plt.ylabel('Magnitude (dB)')
    plt.grid(True, alpha=0.3); plt.legend()
    
    # Sdd21 (Insertion Loss)
    plt.subplot(1, 2, 2)
    plt.plot(freqs, S_low_mean_db[:, 1, 0], 'b-', linewidth=2, label=f'Low {driver_name} (Mean)')
    plt.plot(freqs, S_high_mean_db[:, 1, 0], 'r--', linewidth=2, label=f'High {driver_name} (Mean)')
    plt.title('Insertion Loss (Sdd21)')
    plt.xlabel('Frequency (GHz)'); plt.ylabel('Magnitude (dB)')
    plt.grid(True, alpha=0.3); plt.legend()

    plt.tight_layout()
    save_fn = f"{dataset_type}_statistical_topology.png"
    save_path = os.path.join(proj_root, "results/figures", save_fn)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
    print(f"Statistical report saved to: {save_path}")


if __name__ == "__main__":
    # Point to the processed datasets (make sure to update the paths if needed)
    PROJ_ROOT = os.path.expanduser("~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects")
    
    datasets = [
        ('array', "data/processed/Universal-Diff-SI-Array/via_array_dataset.pt"),
        ('link', "data/processed/Universal-Diff-SI-Link/via_link_dataset.pt")
    ]
    
    for d_type, rel_path in datasets:
        full_pt_path = os.path.join(PROJ_ROOT, rel_path)
        if os.path.exists(full_pt_path):
            plot_topology_trends(full_pt_path, PROJ_ROOT, d_type)
        else:
             print(f"Could not find file at {full_pt_path}")