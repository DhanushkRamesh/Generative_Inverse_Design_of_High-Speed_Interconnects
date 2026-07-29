import sys
from pathlib import Path
import torch
import numpy as np
import skrf as rf
from skrf.vectorFitting import VectorFitting
from sklearn.cluster import KMeans
from tqdm import tqdm

def main():
    print("=== PHASE 1: UNIVERSAL POLE BASIS EXTRACTION ===")
    sandbox_root = Path(__file__).resolve().parent
    if str(sandbox_root) not in sys.path: sys.path.insert(0, str(sandbox_root))
    
    dataset_path = sandbox_root.parent / "data" / "processed" / "Universal-Diff-SI-Array" / "diff_pair_dataset.pt"
    out_path = sandbox_root / "data" / "universal_pole_basis.pt"
    out_path.parent.mkdir(exist_ok=True)
    
    data = torch.load(dataset_path, weights_only=False)
    freqs_hz = data["frequencies"].numpy().astype(np.float64)
    Y_real, Y_imag = data["Y_real"].numpy(), data["Y_imag"].numpy()
    
    sim_ids = np.array(data["sim_ids"])
    unique_sims = np.unique(sim_ids)
    rng = np.random.default_rng(42)
    chosen_sims = rng.choice(unique_sims, size=min(50, len(unique_sims)), replace=False)
    
    all_extracted_poles = []
    freq = rf.Frequency.from_f(freqs_hz, unit='hz')
    
    print(f"Running Vector Fitting on {len(chosen_sims)} representative geometries...")
    for sim_id in tqdm(chosen_sims):
        idx = int(np.where(sim_ids == sim_id)[0][0])
        
        # Enforce complex128 to prevent skrf numerical crashes
        S_matrix = Y_real[idx].astype(np.float64) + 1j * Y_imag[idx].astype(np.float64)
        
        ntwk = rf.Network(frequency=freq, s=S_matrix)
        vf = VectorFitting(ntwk)
        
        try:
            # Removed the suspicious keyword arg. 
            vf.vector_fit(n_poles_real=0, n_poles_cmplx=40)
            
            # Keep only upper half-plane (Im > 0)
            pos_idx = np.where(np.imag(vf.poles) > 0)[0]
            all_extracted_poles.extend(vf.poles[pos_idx])
            
        except Exception as e:
            # DO NOT swallow the error. Print it so we can fix it!
            print(f"\nCRITICAL VFIT ERROR on Sim {sim_id}: {e}")
            sys.exit(1) # Stop immediately
            
    all_extracted_poles = np.array(all_extracted_poles)
    print(f"\nExtracted {len(all_extracted_poles)} raw upper-plane poles.")
    
    if len(all_extracted_poles) < 40:
        print("Error: Not enough poles extracted to cluster into 40 centroids.")
        sys.exit(1)
        
    print("Clustering into 40 universal complex-conjugate pairs...")
    X_cluster = np.column_stack((np.real(all_extracted_poles), np.imag(all_extracted_poles)))
    kmeans = KMeans(n_clusters=40, random_state=42, n_init=10).fit(X_cluster)
    
    centroids = kmeans.cluster_centers_
    poles_c = centroids[:, 0] + 1j * centroids[:, 1]
    
    # ENFORCE STRICT STABILITY (Re < 0)
    poles_c = -np.abs(np.real(poles_c)) + 1j * np.abs(np.imag(poles_c))
    poles_c = poles_c[np.argsort(np.imag(poles_c))]
    
    pole_tensor = torch.tensor(poles_c, dtype=torch.complex128)
    torch.save(pole_tensor, out_path)
    
    print(f"Saved locked pole basis to {out_path}")
    print(f"Frequency range of poles: {np.imag(poles_c).min()/(2*np.pi*1e9):.2f} GHz to {np.imag(poles_c).max()/(2*np.pi*1e9):.2f} GHz")

if __name__ == "__main__":
    main()