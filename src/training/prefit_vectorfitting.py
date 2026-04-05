import sys
import os
import torch
import numpy as np
import skrf as rf
from tqdm import tqdm

"""
Stage 1: Classical Vector Fitting Pre-Fitting
==============================================
This script runs classical Vector Fitting (Gustavsen & Semlyen, 1999) on every
sample in the dataset to produce ground-truth pole/residue supervision targets.

Why this is needed:
  - Training a neural network to discover pole locations purely through
    backpropagation is extremely difficult because the loss landscape is
    highly non-convex — moving a pole by 0.5 GHz can shift a resonance
    dramatically, creating sharp loss gradients that confuse the optimizer.
  - Classical VF solves this problem iteratively (linear solve for residues,
    eigenvalue relocation for poles) and reliably converges to good solutions.
  - By pre-computing VF targets, we give the neural network direct supervision
    on WHERE poles should be and HOW BIG residues should be for each geometry.
    The network only needs to learn how these vary with geometry — a much
    smoother and easier optimization problem.

Usage:
  python prefit_vectorfitting.py --dataset_type array
  python prefit_vectorfitting.py --dataset_type link
"""

# Add project root to path
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJ_ROOT)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_POLES_COMPLEX = 40  # Number of complex conjugate pairs for VF fitting.
                         # Must match num_poles_half in RationalNet (num_poles=80 → 40 pairs).
                         # VF can handle this robustly whereas the NN struggled.

def run_vectorfitting_on_dataset(dataset_type='array'):
    """
    Loads the processed dataset, runs scikit-rf VectorFitting on each sample,
    and saves the fitted poles/residues/d_term as a new .pt file for Stage 2 training.
    """
    # Load the processed dataset
    dataset_folder = 'Universal-Diff-SI-Array' if dataset_type == 'array' else 'Universal-Diff-SI-Link'
    data_path = os.path.join(PROJ_ROOT, f"data/processed/{dataset_folder}/via_{dataset_type}_dataset.pt")
    
    print(f"[INFO] Loading dataset from {data_path}")
    data = torch.load(data_path, weights_only=False)
    
    Y_real = data['Y_real']  # shape: [num_samples, num_freqs, 4, 4]
    Y_imag = data['Y_imag']  # shape: [num_samples, num_freqs, 4, 4]
    num_samples = Y_real.shape[0]
    num_freqs = Y_real.shape[1]
    
    print(f"[INFO] Dataset: {num_samples} samples, {num_freqs} frequency points")
    print(f"[INFO] Running Vector Fitting with {NUM_POLES_COMPLEX} conjugate pairs per sample...")
    
    # Construct the frequency axis matching the dataset
    # TUHH standard: 0.25 GHz to 100 GHz
    freq_obj = rf.Frequency(0.25, 100, num_freqs, 'ghz')
    
    # Storage for VF results
    # We only store the upper-half poles (positive imaginary) since the model
    # enforces conjugate symmetry — the lower half is mirrored automatically.
    all_poles_real = []       # [num_samples, NUM_POLES_COMPLEX]
    all_poles_imag = []       # [num_samples, NUM_POLES_COMPLEX]
    all_residues_real = []    # [num_samples, NUM_POLES_COMPLEX, 4, 4]
    all_residues_imag = []    # [num_samples, NUM_POLES_COMPLEX, 4, 4]
    all_d_term = []           # [num_samples, 4, 4]
    
    failed_indices = []
    
    for idx in tqdm(range(num_samples), desc="Vector Fitting"):
        # Reconstruct the complex S-parameter matrix for this sample
        s_complex = Y_real[idx].numpy() + 1j * Y_imag[idx].numpy()
        # s_complex shape: [num_freqs, 4, 4]
        
        # Create a scikit-rf Network object
        ntwk = rf.Network(frequency=freq_obj, s=s_complex)
        
        try:
            # Run Vector Fitting
            vf = rf.VectorFitting(ntwk)
            vf.vector_fit(n_poles_real=0, n_poles_cmplx=NUM_POLES_COMPLEX)
            
            # Extract fitted poles
            # VF poles come as a 1D array of complex values.
            # For conjugate pairs, VF returns both p and p* — we only keep the
            # upper half (positive imaginary) and let the model mirror them.
            poles = vf.poles  # shape: [2 * NUM_POLES_COMPLEX]
            
            # Select only poles with positive imaginary part (upper half-plane)
            upper_mask = poles.imag >= 0
            poles_upper = poles[upper_mask]
            
            # If VF returned real poles or unexpected count, handle gracefully
            if len(poles_upper) != NUM_POLES_COMPLEX:
                # Sort all poles by imaginary part and take the top half
                sorted_idx = np.argsort(poles.imag)
                poles_upper = poles[sorted_idx[NUM_POLES_COMPLEX:]]
            
            # Sort by imaginary part (ascending frequency) for consistent ordering
            # across samples. This ensures pole k always corresponds to roughly
            # the same frequency region, making the supervised loss meaningful.
            sort_idx = np.argsort(poles_upper.imag)
            poles_upper = poles_upper[sort_idx]
            
            # Extract residues
            # scikit-rf VectorFitting stores residues as [n_ports, n_ports, n_poles]
            # We need [n_poles, n_ports, n_ports] to match our model convention.
            residues_raw = vf.residues  # shape: [4, 4, 2*NUM_POLES_COMPLEX]
            
            # Select residues corresponding to the upper-half poles and reorder
            # to match the sorted pole ordering
            residues_upper = residues_raw[:, :, upper_mask][:, :, sort_idx]
            # Transpose to [n_poles, 4, 4]
            residues_upper = np.transpose(residues_upper, (2, 0, 1))
            
            # Extract direct term (D matrix)
            d_term = vf.proportional_coeff  # shape: [4, 4]
            
            # Scale poles to match model convention (GHz · rad/s)
            # VF in scikit-rf works with the same frequency units as the Network object.
            # Our model normalises frequencies by 1e9 (divides Hz by 1e9 to get GHz)
            # and poles are in GHz·rad/s space. scikit-rf VF returns poles in the
            # natural units of the frequency axis (rad/s if freq is in Hz, or
            # GHz·rad/s if freq is in GHz). Since we created the Network with
            # rf.Frequency in GHz, poles should already be in GHz·rad/s.
            # Verify and store.
            
            all_poles_real.append(poles_upper.real.astype(np.float32))
            all_poles_imag.append(poles_upper.imag.astype(np.float32))
            all_residues_real.append(residues_upper.real.astype(np.float32))
            all_residues_imag.append(residues_upper.imag.astype(np.float32))
            all_d_term.append(d_term.real.astype(np.float32))
            
        except Exception as e:
            # Some samples may fail VF (ill-conditioned, noisy, etc.)
            # Record the failure and fill with NaN for later filtering.
            print(f"\n  [WARN] VF failed for sample {idx}: {e}")
            failed_indices.append(idx)
            all_poles_real.append(np.full(NUM_POLES_COMPLEX, np.nan, dtype=np.float32))
            all_poles_imag.append(np.full(NUM_POLES_COMPLEX, np.nan, dtype=np.float32))
            all_residues_real.append(np.full((NUM_POLES_COMPLEX, 4, 4), np.nan, dtype=np.float32))
            all_residues_imag.append(np.full((NUM_POLES_COMPLEX, 4, 4), np.nan, dtype=np.float32))
            all_d_term.append(np.full((4, 4), np.nan, dtype=np.float32))
    
    # Convert to tensors
    vf_targets = {
        'poles_real': torch.tensor(np.stack(all_poles_real)),       # [num_samples, NUM_POLES_COMPLEX]
        'poles_imag': torch.tensor(np.stack(all_poles_imag)),       # [num_samples, NUM_POLES_COMPLEX]
        'residues_real': torch.tensor(np.stack(all_residues_real)), # [num_samples, NUM_POLES_COMPLEX, 4, 4]
        'residues_imag': torch.tensor(np.stack(all_residues_imag)), # [num_samples, NUM_POLES_COMPLEX, 4, 4]
        'd_term': torch.tensor(np.stack(all_d_term)),               # [num_samples, 4, 4]
        'num_poles_complex': NUM_POLES_COMPLEX,
        'failed_indices': failed_indices,
        'dataset_type': dataset_type,
    }
    
    # Save VF targets
    save_path = os.path.join(PROJ_ROOT, f"data/processed/{dataset_folder}/vf_targets_{dataset_type}.pt")
    torch.save(vf_targets, save_path)
    
    success_count = num_samples - len(failed_indices)
    print(f"\n[INFO] Vector Fitting complete:")
    print(f"  Successful: {success_count}/{num_samples}")
    print(f"  Failed:     {len(failed_indices)}/{num_samples}")
    print(f"  Saved to:   {save_path}")
    
    # Print pole statistics for sanity checking
    valid_mask = ~torch.isnan(vf_targets['poles_real'][:, 0])
    valid_poles_real = vf_targets['poles_real'][valid_mask]
    valid_poles_imag = vf_targets['poles_imag'][valid_mask]
    
    print(f"\n  Pole real parts:  min={valid_poles_real.min():.4f}, max={valid_poles_real.max():.4f}, mean={valid_poles_real.mean():.4f}")
    print(f"  Pole imag parts:  min={valid_poles_imag.min():.4f}, max={valid_poles_imag.max():.4f} (GHz·rad/s)")
    print(f"  Pole freq range:  {valid_poles_imag.min()/(2*np.pi):.2f} to {valid_poles_imag.max()/(2*np.pi):.2f} GHz")

if __name__ == "__main__":
    for dtype in ['array', 'link']:
        print(f"\n{'='*70}")
        print(f"  Processing {dtype.upper()} dataset")
        print(f"{'='*70}")
        run_vectorfitting_on_dataset(dataset_type=dtype)