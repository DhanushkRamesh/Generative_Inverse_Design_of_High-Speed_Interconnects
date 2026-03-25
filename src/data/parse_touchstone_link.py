# src/data/parse_touchstone_link.py
import os
import glob
import numpy as np 
import pandas as pd
import torch
import skrf as rf
from tqdm import tqdm
import datetime
import subprocess
import sys

# ensure the utils directory is in the path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.physics_utils import convert_to_mixed_mode, check_passivity, enforce_reciprocity
# function to get the current git hash for traceability
def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    except:
        return 'unknown'

def parse_touchstone_link(base_dir, output_dir):
    print("Parsing Touchstone files...")
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    csv_path=os.path.join(base_dir, "parameter.csv")
    variation_dir=os.path.join(base_dir, "variation")
    #load the full dataset
    df=pd.read_csv(csv_path)
    print(f"Loaded dataset with {len(df)} samples.")
    #adding this line to change the spelling and correct the typo "LOSSTANGENT" instead of "LOSTANGENT" in the via_link dataset
    df.rename(columns={'LOSTANGENT':'LOSSTANGENT'}, inplace=True)
    print(f"DataFrame columns after rename: {df.columns.tolist()}")
    print(f"Feature count: {len(df.drop(columns=['SIM_ID','SIMULATION']).columns)}")

    sim_ids = df['SIMULATION'].values
    features = df.drop(columns=['SIM_ID', 'SIMULATION'])
    #track log-scalling for inverser design correction later
    log_features = []
    #Convert exponential material properties to linear scale for better training stability
    if 'LOSSTANGENT' in features.columns:
        features['LOSSTANGENT'] = np.log10(features['LOSSTANGENT'].clip(lower=1e-6))
        log_features.append('LOSSTANGENT')
    if 'SL_WIDTH' in features.columns:
        features['SL_WIDTH'] = np.log10(features['SL_WIDTH'].clip(lower=1e-6))
        log_features.append('SL_WIDTH')
    if 'TDIEL' in features.columns:
        features['TDIEL'] = np.log10(features['TDIEL'].clip(lower=1e-6))
        log_features.append('TDIEL')
    if 'LENGTH' in features.columns:
        features['LENGTH'] = np.log10(features['LENGTH'].clip(lower=1e-6))
        log_features.append('LENGTH')

    feature_names = features.columns.tolist()
    #z-score normalization of features for better training stability
    X_raw_all = features.values
    #extracting core features 4 port s-parameters
    Y_real_list = []
    Y_imag_list = []
    valid_X_raw_list = []
    valid_sim_ids = [] # Traceability for the simulations that successfully passed through the parser and passivity check
    passivity_violations = 0
    print(f"extracting s-parameters for {len(sim_ids)} simulations...")
    #STANDARDIZED INTERPOLATION (0 to 100 GHz, 401 points)
    std_rf_freq = rf.Frequency(0, 100, 401, 'ghz') # For scikit-rf interpolation
    master_freqs = None # To ensure all samples have the same frequency axis after interpolation

    # NOTE: sim_ids and X_raw_all are rigidly parallel from the CSV. 
    # Do NOT shuffle sim_ids before this loop!

    for idx, sim_id in tqdm(enumerate(sim_ids), total=len(sim_ids)):
        sim_folder = os.path.join(variation_dir, sim_id)
        search_pattern = os.path.join(sim_folder, "*.s*p")
        files = glob.glob(search_pattern)

        if not files:
            print(f"Warning: No Touchstone file found for simulation ID {sim_id} in {sim_folder}")
            continue
        touchstone_file = files[0]

        try:
            #load the network from the touchstone file
            network = rf.Network(touchstone_file)
            #STANDARDIZED INTERPOLATION (0 to 100 GHz, 401 points)
            # This guarantees every tensor has exactly 401 frequency steps
            network.interpolate(
                std_rf_freq,
                bounds_error=False, 
                fill_value="extrapolate")
            # After interpolation, all networks should have the same frequency axis (std_rf_freq)
            if master_freqs is None:
                master_freqs = network.f
            #extract the s-parameters and reshape to (4,4,201)
            #dynamic port extraction based on the number of ports in the touchstone file, but we will only take the core 4x4 for the dataset
            num_ports=network.s.shape[1]
            half=num_ports//2
            #grab indicies for TX+,TX- (0,1) and RX+,RX- (half, half+1)
            port_idx = [0, 1, half, half+1]
            #dynamically extract the core 4x4 s-parameter matrix based on the identified indices
            core_s_matrix=network.s[:, port_idx, :][:, :, port_idx]
            #Enforce reciprocity to kill numerical noise before passivity check
            core_s_matrix = enforce_reciprocity(core_s_matrix)
             # passivity check to ensure physical validity of the data. If it fails, we skip this sample.
            is_passive, min_eig = check_passivity(core_s_matrix, threshold=-1e-6)
            if not is_passive:
                print(f"Passivity violation detected in {sim_id}, min eig: {min_eig:.2e}. Dropping from dataset.")
                passivity_violations += 1
                continue # Skip this simulation!
            # MIXED-MODE CONVERSION
            # Transform from (TX+, TX-, RX+, RX-) to (Sdd, Sdc, Scd, Scc)
            mixed_mode_matrix = convert_to_mixed_mode(core_s_matrix)
            # Convert to PyTorch tensors (separating real and imaginary parts)
            s_matrix_real= torch.tensor(np.real(mixed_mode_matrix), dtype=torch.float32)
            s_matrix_imag= torch.tensor(np.imag(mixed_mode_matrix), dtype=torch.float32)
            Y_real_list.append(s_matrix_real)
            Y_imag_list.append(s_matrix_imag)
            valid_sim_ids.append(sim_id) # Save the ID
            #append num_ports dynamically to the feature set for the inverse design model to learn link density variance.
            current_x_raw = np.append(X_raw_all[idx], num_ports)
            valid_X_raw_list.append(current_x_raw)

        except Exception as e:
            print(f"Error processing Touchstone file for simulation ID {sim_id}: {e}")
            continue
    #Finilize tensors
    if not valid_X_raw_list:
        print("Error: No valid Touchstone files were processed. Please check the dataset and file paths.")
        return
    #add the num_ports feature to the feature names list for traceability
    feature_names.append('NUM_PORTS')
    #Z-score normalization of features for better training stability
    X_raw_valid = np.stack(valid_X_raw_list)
    X_mean = X_raw_valid.mean(axis=0)
    X_std = X_raw_valid.std(axis=0)
    X_std_safe = np.where(X_std == 0, 1e-5, X_std) # Prevent divide by zero
    X_normalized = (X_raw_valid - X_mean) / X_std_safe # Normalize features
    X_final = torch.tensor(X_normalized, dtype=torch.float32)
    Y_real=torch.stack(Y_real_list)
    Y_imag=torch.stack(Y_imag_list)
    #save the tensors
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "via_link_dataset.pt")
    #save everything including tensors and metadata for traceability and future analysis
    torch.save({
        #training data
        'X': X_final,
        'Y_real': Y_real,
        'Y_imag': Y_imag,
        'feature_names': feature_names,
        'frequencies':torch.tensor(master_freqs, dtype=torch.float32),
        #metadata for traceability
        'sim_ids': valid_sim_ids, #for design traceability
        'X_mean': torch.tensor(X_mean, dtype=torch.float32), #for inverse design rescaling
        'X_std': torch.tensor(X_std_safe, dtype=torch.float32), #for inverse design
        'log_features': log_features, #to remember which features were log-scaled for inverse design correction
        'metadata': {
            'creation_date': datetime.datetime.now().isoformat(),
            'git_hash': get_git_hash(),
            'num_samples': len(valid_sim_ids),
            'passivity_threshold': -1e-6,
            'passivity_violations': passivity_violations
        }
    }, save_path)
    print(f"Dataset saved to {save_path}")
    print(f"Metadata included for {len(valid_sim_ids)} simulations.")
    print(f"Final dataset shapes - X: {X_final.shape}, Y_real: {Y_real.shape}, Y_imag: {Y_imag.shape} (frequency points, 4, 4)")

if __name__ == "__main__":
    base_dir = os.path.expanduser("~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/data/raw/Universal-Diff-SI-Link")
    output_dir = os.path.expanduser("~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/data/processed/Universal-Diff-SI-Link")
    parse_touchstone_link(base_dir, output_dir)