# src/data/analyze_design_topology.py
import os
import pandas as pd
import numpy as np

def analyze_dataset_topology(csv_path, dataset_name):
    print(f"\n{'='*50}")
    print(f"Loading {dataset_name.upper()} dataset: {csv_path.split('/')[-2]}")
    df = pd.read_csv(csv_path)
    
    # Correct the typo if it exists (matching your parser logic)
    df.rename(columns={'LOSTANGENT': 'LOSSTANGENT'}, inplace=True, errors='ignore')
    
    # Drop the ID columns to isolate just the physics features
    features_df = df.drop(columns=['SIM_ID', 'SIMULATION'])
    #normalize the features to ensure tolerance is meaningful across different scales
    raw_features = features_df.values
    f_min = raw_features.min(axis=0)
    f_max = raw_features.max(axis=0)
    range_vals = np.where((f_max - f_min) == 0, 1e-5, (f_max - f_min))
    features = (raw_features - f_min) / range_vals
    
    n_samples, n_features = features.shape
    print(f"Analyzing {n_samples} simulations across {n_features} features...")
    
    perfect_pairs = 0
    lhs_pairs = 0
    
    # tolerance is set to 0.01 in normalized feature space, meaning we consider a feature "changed" if it differs by more than 1% of its range
    tol = 0.01
    
    print("Computing pairwise feature distances... blink thrice!!! or maybe more...")
    
    # Loop through all unique pairs
    for i in range(n_samples):
        # Compare row 'i' to all rows below it to avoid double-counting
        differences = np.abs(features[i+1:] - features[i])
        
        # Count how many features are mathematically different (greater than our tolerance)
        changed_features_count = np.sum(differences > tol, axis=1)
        
        # If exactly 1 feature changed, it's a perfect sensitivity pair!
        perfect_pairs += np.sum(changed_features_count == 1)
        
        # If ALL features changed, it's a completely disjoint LHS pair
        lhs_pairs += np.sum(changed_features_count == n_features)

    print("-" * 50)
    print(f" {dataset_name.upper()} DATASET TOPOLOGY RESULTS ")
    print("-" * 50)
    print(f"Perfect 1-Variable Pairs Found : {perfect_pairs}")
    print(f"Completely Random (LHS) Pairs  : {lhs_pairs}")
    print("-" * 50)
    
    if perfect_pairs > 100:
        print(" CONCLUSION: This is a Grid/One-At-A-Time dataset!")
        print("We CAN group the data for Finite-Difference Hessian optimization.")
    elif perfect_pairs == 0:
        print("CONCLUSION: This is a Latin Hypercube (LHS) / Space-Filling dataset!")
        print("Every simulation changes all variables at once. We CANNOT use physical grouping.")
        print("Instead, we must use PyTorch Autograd to compute the Hessian during training.")
    else:
        print("CONCLUSION: Mixed or fragmented dataset. A few pairs exist, but not enough for a structured grid.")

if __name__ == "__main__":
    # Point this to the actual raw parameter.csv files for both datasets
    PROJ_ROOT = os.path.expanduser("~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects")
    
    datasets_to_test = [
        ('array', 'data/raw/Universal-Diff-SI-Array/parameter.csv'),
        ('link',  'data/raw/Universal-Diff-SI-Link/parameter.csv')
    ]

    for dataset_type, rel_path in datasets_to_test:
        full_path = os.path.join(PROJ_ROOT, rel_path)
        if os.path.exists(full_path):
            analyze_dataset_topology(full_path, dataset_type)
        else:
            print(f"Could not find {full_path}")