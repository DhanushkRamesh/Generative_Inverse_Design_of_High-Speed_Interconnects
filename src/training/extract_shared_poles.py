import sys
import os
import torch
import numpy as np
import skrf as rf
from sklearn.cluster import KMeans

"""
Shared Pole Extraction via Classical Vector Fitting
=====================================================
This script runs Vector Fitting on a small number of representative samples
(~10 per dataset) and clusters the resulting poles into a single shared pole
basis set for the entire dataset.

The key insight from Gustavsen and Semlyen (1999) is that VF convergence
depends critically on starting pole placement. For a family of structures
sharing the same physical class (PCB via arrays/links), the cavity resonance
frequencies are approximately geometry-invariant — what changes across
geometries is primarily the coupling strength (residues), not the resonance
locations (poles). This observation, confirmed by recent Neuro-TF work
(Liu et al., 2025, MDPI Micromachines) and pole-residue surrogate
modelling literature (Silva Rezende et al., 2025, Springer), motivates
extracting a shared pole basis from a handful of representative samples
rather than running VF on every sample (which takes ~19 hours and
defeats the purpose of a fast surrogate).

The extracted poles are saved as a .pt file and loaded by RationalNet
as a fixed (non-trainable) buffer. The neural network then only needs
to predict residues and the direct term — a smooth regression problem
that backpropagation handles well.

Usage:
    python extract_shared_poles.py
    (Processes both array and link datasets automatically)
"""

# Add project root to path
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJ_ROOT)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_REPRESENTATIVE_SAMPLES = 10  # Number of samples to run VF on (per dataset)
NUM_POLES_PER_SAMPLE = 40        # Complex conjugate pairs per VF fit
NUM_SHARED_POLES = 40            # Final number of conjugate pairs after clustering.
                                  # This determines num_poles_half in RationalNet.
                                  # Total poles in model = 2 * NUM_SHARED_POLES = 80.

def select_representative_samples(data, dataset_type='array'):
    """
    Selects geometrically diverse samples from the dataset using
    a simple strategy: pick samples at equal intervals across the
    first principal component of the feature space.
    
    This ensures we cover the extremes and middle of the geometry
    distribution — small vias, large vias, tight pitch, wide pitch —
    so the resulting pole basis spans the resonance structure of
    the entire dataset, not just one corner of the design space.
    
    Args:
        data: dict from torch.load() containing 'X' (features) and 'feature_names'
        dataset_type: 'array' or 'link'
    Returns:
        indices: list of sample indices to run VF on
    """
    X = data['X'].numpy()
    num_samples = X.shape[0]
    
    # Simple strategy: sort by first principal component and pick evenly spaced
    # This ensures geometric diversity without requiring sklearn PCA
    # (the Z-scored features already have zero mean, so the largest-variance
    # direction is well approximated by the first SVD component)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pc1_scores = U[:, 0] * S[0]  # Project onto first principal component
    sorted_indices = np.argsort(pc1_scores)
    
    # Pick evenly spaced samples across the sorted order
    step = max(1, num_samples // NUM_REPRESENTATIVE_SAMPLES)
    selected = sorted_indices[::step][:NUM_REPRESENTATIVE_SAMPLES]
    
    print(f"  Selected {len(selected)} representative samples from {num_samples} total")
    print(f"  Indices: {selected.tolist()}")
    
    return selected.tolist()


def run_vf_on_sample(s_complex, freq_obj, num_poles_cmplx):
    """
    Runs scikit-rf VectorFitting on a single S-parameter matrix.
    
    Args:
        s_complex: numpy array [num_freqs, 4, 4] complex128
        freq_obj: scikit-rf Frequency object
        num_poles_cmplx: number of complex conjugate pairs to fit
    Returns:
        poles: numpy array of complex poles (upper half-plane only)
        success: bool indicating whether VF converged
    """
    ntwk = rf.Network(frequency=freq_obj, s=s_complex)
    vf = rf.VectorFitting(ntwk)
    
    try:
        vf.vector_fit(n_poles_real=0, n_poles_cmplx=num_poles_cmplx)
        poles = np.array(vf.poles)
        
        # CRITICAL SCALING: scikit-rf VectorFitting returns poles in rad/s
        # (the natural unit of the Laplace variable s = jω where ω is in rad/s).
        # Our RationalNet normalises frequencies by dividing Hz by 1e9 to get GHz,
        # then computes omega = 2π * f_GHz. So the model's internal Laplace variable
        # is s = j * 2π * f_GHz, which means poles must be in GHz·rad/s units.
        # To convert: divide raw rad/s poles by 1e9.
        #
        # Without this scaling, poles end up at ~350 billion GHz instead of ~50 GHz,
        # and the rational function denominators (s - Pn) never produce resonances
        # in the 0-100 GHz band the model operates in.
        poles = poles / 1e9
        
        # Keep only upper half-plane poles (positive imaginary)
        # For conjugate pairs, VF stores both p and p*
        upper_mask = poles.imag >= 0
        poles_upper = poles[upper_mask]
        
        # If we got real poles (imag=0), keep them too
        # Handle count mismatch gracefully
        if len(poles_upper) > num_poles_cmplx:
            # Sort by imaginary part, keep the ones spread across the band
            sort_idx = np.argsort(poles_upper.imag)
            poles_upper = poles_upper[sort_idx[-num_poles_cmplx:]]
        
        return poles_upper, True
        
    except Exception as e:
        print(f"    [WARN] VF failed: {e}")
        return None, False


def cluster_poles(all_poles, num_clusters):
    """
    Clusters poles from multiple VF runs into a shared pole basis
    using K-Means on the 2D (real, imaginary) coordinates.
    
    This addresses the fundamental pole-ordering problem identified
    by Silva Rezende et al. (2025): poles from different VF runs
    are not consistently ordered, making direct averaging meaningless.
    Clustering finds natural groupings of "similar" poles across
    samples and returns their centroids as the shared basis.
    
    Args:
        all_poles: numpy array [total_poles_collected, ] complex
        num_clusters: number of shared poles to produce
    Returns:
        shared_poles: numpy array [num_clusters, ] complex — cluster centroids
    """
    # Stack real and imaginary parts as 2D features for KMeans
    # Scale real parts relative to imaginary parts since the damping
    # range (~0 to -50) is much smaller than the frequency range (~0 to 600 rad/s)
    pole_features = np.column_stack([
        all_poles.real,
        all_poles.imag
    ])
    
    print(f"  Clustering {len(all_poles)} poles into {num_clusters} shared poles...")
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(pole_features)
    
    # Reconstruct complex pole centroids from cluster centers
    centroids = kmeans.cluster_centers_
    shared_poles = centroids[:, 0] + 1j * centroids[:, 1]
    
    # Sort by imaginary part (ascending frequency) for consistent ordering
    sort_idx = np.argsort(shared_poles.imag)
    shared_poles = shared_poles[sort_idx]
    
    # Verify all poles have negative real parts (causal)
    # VF on passive structures should always produce stable poles,
    # but clustering centroids could theoretically shift a pole
    # slightly into the right half-plane. Force them left if needed.
    if np.any(shared_poles.real >= 0):
        num_fixed = np.sum(shared_poles.real >= 0)
        print(f"  [WARN] {num_fixed} poles had non-negative real parts — forcing to -0.01")
        shared_poles.real = np.where(shared_poles.real >= 0, -0.01, shared_poles.real)
    
    return shared_poles


def extract_shared_poles(dataset_type='array'):
    """
    Main function: loads dataset, selects representative samples,
    runs VF on each, clusters the poles, and saves the shared pole set.
    """
    # Load the processed dataset
    dataset_folder = 'Universal-Diff-SI-Array' if dataset_type == 'array' else 'Universal-Diff-SI-Link'
    data_path = os.path.join(PROJ_ROOT, f"data/processed/{dataset_folder}/via_{dataset_type}_dataset.pt")
    
    if not os.path.exists(data_path):
        print(f"  [SKIP] Dataset not found at {data_path}")
        return
    
    print(f"  Loading dataset from {data_path}")
    data = torch.load(data_path, weights_only=False)
    
    Y_real = data['Y_real']  # [num_samples, num_freqs, 4, 4]
    Y_imag = data['Y_imag']
    num_freqs = Y_real.shape[1]
    
    # Construct frequency axis matching the dataset
    freq_obj = rf.Frequency(0.25, 100, num_freqs, 'ghz')
    
    # Select representative samples spanning the geometry space
    selected_indices = select_representative_samples(data, dataset_type)
    
    # Run VF on each selected sample and collect poles
    all_poles_collected = []
    successful = 0
    
    for i, idx in enumerate(selected_indices):
        print(f"  [{i+1}/{len(selected_indices)}] Running VF on sample {idx}...", end=" ")
        
        s_complex = Y_real[idx].numpy() + 1j * Y_imag[idx].numpy()
        poles, success = run_vf_on_sample(s_complex, freq_obj, NUM_POLES_PER_SAMPLE)
        
        if success and poles is not None:
            all_poles_collected.append(poles)
            successful += 1
            print(f"OK ({len(poles)} poles)")
        else:
            print("FAILED")
    
    print(f"\n  VF succeeded on {successful}/{len(selected_indices)} samples")
    
    if successful < 3:
        print(f"  [ERROR] Too few successful VF fits. Need at least 3.")
        return
    
    # Concatenate all collected poles
    all_poles = np.concatenate(all_poles_collected)
    print(f"  Total poles collected: {len(all_poles)}")
    
    # Cluster into shared pole basis
    shared_poles = cluster_poles(all_poles, NUM_SHARED_POLES)
    
    # Print summary statistics
    print(f"\n  Shared Pole Basis ({len(shared_poles)} poles):")
    print(f"    Real parts:  min={shared_poles.real.min():.4f}, max={shared_poles.real.max():.4f}")
    print(f"    Freq range:  {shared_poles.imag.min()/(2*np.pi):.2f} to {shared_poles.imag.max()/(2*np.pi):.2f} GHz")
    
    # All poles should be causal (negative real part)
    assert np.all(shared_poles.real < 0), "FATAL: Some shared poles are non-causal!"
    print(f"    Causality:   ALL POLES IN LEFT HALF-PLANE ✓")
    
    # Save as a .pt file that RationalNet can load directly
    # We store only the upper-half poles — the model mirrors them
    # for conjugate symmetry in the forward pass.
    save_data = {
        'poles_real': torch.tensor(shared_poles.real.astype(np.float32)),  # [NUM_SHARED_POLES]
        'poles_imag': torch.tensor(shared_poles.imag.astype(np.float32)),  # [NUM_SHARED_POLES]
        'num_poles_half': NUM_SHARED_POLES,
        'num_poles_total': 2 * NUM_SHARED_POLES,
        'dataset_type': dataset_type,
        'num_representative_samples': successful,
        'source_indices': selected_indices,
    }
    
    save_path = os.path.join(PROJ_ROOT, f"data/processed/{dataset_folder}/shared_poles_{dataset_type}.pt")
    torch.save(save_data, save_path)
    print(f"\n  Saved to: {save_path}")


if __name__ == "__main__":
    for dtype in ['array', 'link']:
        print(f"\n{'='*70}")
        print(f"  Extracting Shared Poles for {dtype.upper()} Dataset")
        print(f"{'='*70}")
        extract_shared_poles(dataset_type=dtype)
    
    print(f"\n{'='*70}")
    print(f"  Done. Next step: train the RationalNet with these shared poles as a fixed basis.")
    print(f"{'='*70}")