import sys
import os
import torch
import numpy as np
import skrf as rf
from tqdm import tqdm
import time

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
  python prefit_vectorfitting.py
  (Runs both array and link datasets automatically)
"""

# Add project root to path
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJ_ROOT)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_POLES_COMPLEX = 40  # Number of complex conjugate pairs for VF fitting.
                         # Must match num_poles_half in RationalNet (num_poles=80 → 40 pairs).

def extract_residues_3d(vf, num_ports=4):
    """
    Extracts residues from scikit-rf VectorFitting into a consistent 3D array.
    
    scikit-rf stores residues differently depending on the version:
      - Some versions: numpy object array of shape (n_ports, n_ports) where
        each element is a 1D array of length n_poles
      - Some versions: 2D array of shape (n_responses, n_poles)
      - Some versions: 3D array of shape (n_ports, n_ports, n_poles)
    
    This function normalises all formats to shape [n_ports, n_ports, n_poles].
    """
    residues_raw = vf.residues
    
    # Case 1: Already 3D — (n_ports, n_ports, n_poles)
    if hasattr(residues_raw, 'ndim') and residues_raw.ndim == 3:
        return residues_raw
    
    # Case 2: Object array of shape (n_ports, n_ports) with 1D arrays inside
    if hasattr(residues_raw, 'dtype') and residues_raw.dtype == object:
        n_poles = len(residues_raw.flat[0])
        result = np.zeros((num_ports, num_ports, n_poles), dtype=complex)
        for i in range(num_ports):
            for j in range(num_ports):
                result[i, j, :] = residues_raw[i, j]
        return result
    
    # Case 3: 2D array of shape (n_responses, n_poles) where n_responses = n_ports^2
    if hasattr(residues_raw, 'ndim') and residues_raw.ndim == 2:
        n_responses, n_poles = residues_raw.shape
        if n_responses == num_ports * num_ports:
            return residues_raw.reshape(num_ports, num_ports, n_poles)
        else:
            # Might be (n_poles, n_responses) transposed
            return residues_raw.T.reshape(num_ports, num_ports, -1)
    
    # Case 4: List of lists or other iterable
    n_poles = len(vf.poles)
    result = np.zeros((num_ports, num_ports, n_poles), dtype=complex)
    for i in range(num_ports):
        for j in range(num_ports):
            result[i, j, :] = np.array(residues_raw[i][j])
    return result

def extract_d_term(vf, num_ports=4):
    """
    Extracts the direct/proportional term from VectorFitting.
    
    scikit-rf may store this as:
      - vf.proportional_coeff (2D array)
      - vf.d (2D array)
      - vf.constant_coeff
    
    Returns shape [n_ports, n_ports].
    """
    # Try different attribute names across scikit-rf versions
    for attr in ['proportional_coeff', 'd', 'constant_coeff']:
        if hasattr(vf, attr):
            d_raw = getattr(vf, attr)
            if d_raw is not None:
                d_arr = np.array(d_raw)
                if d_arr.ndim == 2 and d_arr.shape == (num_ports, num_ports):
                    return d_arr.real
                elif d_arr.ndim == 1 and d_arr.shape[0] == num_ports * num_ports:
                    return d_arr.real.reshape(num_ports, num_ports)
    
    # If no D term found, return zeros (common for many passive structures)
    return np.zeros((num_ports, num_ports), dtype=np.float32)

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
    start_time = time.time()
    
    # Run VF on the first sample with debug output to verify format
    print(f"\n[DEBUG] Running first sample to verify scikit-rf VF output format...")
    s_complex_0 = Y_real[0].numpy() + 1j * Y_imag[0].numpy()
    ntwk_0 = rf.Network(frequency=freq_obj, s=s_complex_0)
    vf_0 = rf.VectorFitting(ntwk_0)
    vf_0.vector_fit(n_poles_real=0, n_poles_cmplx=NUM_POLES_COMPLEX)
    
    print(f"  vf.poles:    type={type(vf_0.poles)}, shape={np.array(vf_0.poles).shape}")
    print(f"  vf.residues: type={type(vf_0.residues)}, ", end="")
    if hasattr(vf_0.residues, 'shape'):
        print(f"shape={vf_0.residues.shape}, dtype={vf_0.residues.dtype}")
    else:
        print(f"len={len(vf_0.residues)}")
    
    # Check which D term attribute exists
    for attr in ['proportional_coeff', 'd', 'constant_coeff']:
        if hasattr(vf_0, attr):
            val = getattr(vf_0, attr)
            if val is not None:
                print(f"  vf.{attr}: type={type(val)}, shape={np.array(val).shape}")
    
    # Verify extraction works on first sample
    try:
        residues_test = extract_residues_3d(vf_0, num_ports=4)
        d_test = extract_d_term(vf_0, num_ports=4)
        print(f"  Extracted residues shape: {residues_test.shape}")
        print(f"  Extracted D term shape:   {d_test.shape}")
        print(f"[DEBUG] Format verified successfully!\n")
    except Exception as e:
        print(f"[ERROR] Residue extraction failed: {e}")
        print(f"        Please check scikit-rf version and VectorFitting API.")
        return
    
    # Main loop over all samples
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
            poles = np.array(vf.poles)  # shape: [2 * NUM_POLES_COMPLEX]
            
            # Select only poles with positive imaginary part (upper half-plane)
            # For conjugate pairs, VF returns both p and p* 
            upper_mask = poles.imag >= 0
            poles_upper = poles[upper_mask]
            
            # Handle edge cases: if VF returned unexpected pole count
            if len(poles_upper) < NUM_POLES_COMPLEX:
                # Some poles may be purely real — include them and pad if needed
                all_poles_sorted = poles[np.argsort(-poles.imag)]  # descending by imag
                poles_upper = all_poles_sorted[:NUM_POLES_COMPLEX]
            elif len(poles_upper) > NUM_POLES_COMPLEX:
                # Take the NUM_POLES_COMPLEX with largest imaginary parts
                sort_by_imag = np.argsort(poles_upper.imag)
                poles_upper = poles_upper[sort_by_imag[-NUM_POLES_COMPLEX:]]
            
            # Sort by imaginary part (ascending frequency) for consistent ordering
            sort_idx = np.argsort(poles_upper.imag)
            poles_upper = poles_upper[sort_idx]
            
            # Extract residues as 3D array [4, 4, n_poles]
            residues_3d = extract_residues_3d(vf, num_ports=4)
            
            # Select residues corresponding to upper-half poles and sort to match
            residues_upper = residues_3d[:, :, upper_mask]
            if residues_upper.shape[2] < NUM_POLES_COMPLEX:
                all_res_sorted = residues_3d[:, :, np.argsort(-poles.imag)]
                residues_upper = all_res_sorted[:, :, :NUM_POLES_COMPLEX]
            elif residues_upper.shape[2] > NUM_POLES_COMPLEX:
                sort_by_imag = np.argsort(poles[upper_mask].imag)
                residues_upper = residues_upper[:, :, sort_by_imag[-NUM_POLES_COMPLEX:]]
            
            residues_upper = residues_upper[:, :, sort_idx]
            # Transpose to [n_poles, 4, 4] to match model convention
            residues_upper = np.transpose(residues_upper, (2, 0, 1))
            
            # Extract direct term (D matrix)
            d_term = extract_d_term(vf, num_ports=4)
            
            all_poles_real.append(poles_upper.real.astype(np.float32))
            all_poles_imag.append(poles_upper.imag.astype(np.float32))
            all_residues_real.append(residues_upper.real.astype(np.float32))
            all_residues_imag.append(residues_upper.imag.astype(np.float32))
            all_d_term.append(d_term.astype(np.float32))
            
        except Exception as e:
            # Some samples may fail VF (ill-conditioned, noisy, etc.)
            failed_indices.append(idx)
            all_poles_real.append(np.full(NUM_POLES_COMPLEX, np.nan, dtype=np.float32))
            all_poles_imag.append(np.full(NUM_POLES_COMPLEX, np.nan, dtype=np.float32))
            all_residues_real.append(np.full((NUM_POLES_COMPLEX, 4, 4), np.nan, dtype=np.float32))
            all_residues_imag.append(np.full((NUM_POLES_COMPLEX, 4, 4), np.nan, dtype=np.float32))
            all_d_term.append(np.full((4, 4), np.nan, dtype=np.float32))
            
            # Only print first 10 failures to avoid flooding the console
            if len(failed_indices) <= 10:
                print(f"\n  [WARN] VF failed for sample {idx}: {e}")
    
    elapsed = time.time() - start_time
    
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
    print(f"\n[INFO] Vector Fitting complete ({elapsed:.1f}s):")
    print(f"  Successful: {success_count}/{num_samples}")
    print(f"  Failed:     {len(failed_indices)}/{num_samples}")
    print(f"  Saved to:   {save_path}")
    
    if len(failed_indices) > 10:
        print(f"  (Showing first 10 failures only, total: {len(failed_indices)})")
    
    # Print pole statistics for sanity checking
    valid_mask = ~torch.isnan(vf_targets['poles_real'][:, 0])
    if valid_mask.any():
        valid_poles_real = vf_targets['poles_real'][valid_mask]
        valid_poles_imag = vf_targets['poles_imag'][valid_mask]
        
        print(f"\n  Pole Statistics:")
        print(f"    Real parts:  min={valid_poles_real.min():.4f}, max={valid_poles_real.max():.4f}, mean={valid_poles_real.mean():.4f}")
        print(f"    Imag parts:  min={valid_poles_imag.min():.4f}, max={valid_poles_imag.max():.4f} (GHz·rad/s)")
        print(f"    Freq range:  {valid_poles_imag.min()/(2*np.pi):.2f} to {valid_poles_imag.max()/(2*np.pi):.2f} GHz")

if __name__ == "__main__":
    for dtype in ['array', 'link']:
        print(f"\n{'='*70}")
        print(f"  Processing {dtype.upper()} dataset")
        print(f"{'='*70}")
        run_vectorfitting_on_dataset(dataset_type=dtype)