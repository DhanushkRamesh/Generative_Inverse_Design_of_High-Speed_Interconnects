import os
import glob
import numpy as np 
import pandas as pd
import torch
import skrf as rf
from tqdm import tqdm

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

    sim_ids = df['SIMULATION'].values
    features = df.drop(columns=['SIM_ID', 'SIMULATION'])
    #track log-scalling for inverser design correction later
    log_features = []
    #Convert exponential material properties to linear scale for better training stability
    if 'CONDUCTIVITY' in features.columns:
        features['CONDUCTIVITY'] = np.log10(features['CONDUCTIVITY'])
        log_features.append('CONDUCTIVITY')
    if 'LOSSTANGENT' in features.columns:
        features['LOSSTANGENT'] = np.log10(features['LOSSTANGENT'])
        log_features.append('LOSSTANGENT')

    feature_names = features.columns.tolist()
    feature_values = features.values
    #min-max scaling of the features
    X_raw = feature_values
    X_min = X_raw.min(axis=0)
    X_max = X_raw.max(axis=0)
    range_values = np.where((X_max - X_min) == 0, 1e-5,(X_max - X_min))
    X_scaled = (X_raw - X_min) / range_values
    X_normalized= 2.0*X_scaled - 1.0
    X_tensor = torch.tensor(X_normalized, dtype=torch.float32)

    #extracting core features 4 port s-parameters
    Y_real_list = []
    Y_imag_list = []
    valid_indexes = []
    valid_sim_ids = [] # Traceability
    print(f"extracting s-parameters for {len(sim_ids)} simulations...")

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
            #extract the s-parameters and reshape to (4,4,201)
            #dynamic port extraction based on the number of ports in the touchstone file, but we will only take the core 4x4 for the dataset
            num_ports=network.s.shape[1]
            half=num_ports//2
            #grab indicies for TX+,TX- (0,1) and RX+,RX- (half, half+1)
            port_idx = [0, 1, half, half+1]
            #dynamically extract the core 4x4 s-parameter matrix based on the identified indices
            core_s_matrix=network.s[:, port_idx, :][:, :, port_idx]

            s_matrix_real= torch.tensor(np.real(core_s_matrix), dtype=torch.float32)
            s_matrix_imag= torch.tensor(np.imag(core_s_matrix), dtype=torch.float32)
            Y_real_list.append(s_matrix_real)
            Y_imag_list.append(s_matrix_imag)
            valid_indexes.append(idx)
            valid_sim_ids.append(sim_id) # Save the ID

        except Exception as e:
            print(f"Error processing Touchstone file for simulation ID {sim_id}: {e}")
            continue
    #Finilize tensors
    if not valid_indexes:
        print("Error: No valid Touchstone files were processed. Please check the dataset and file paths.")
        return
    X_final = X_tensor[valid_indexes]
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
        'frequencies':torch.tensor(network.f, dtype=torch.float32),
        #metadata for traceability
        'sim_ids': valid_sim_ids, #for design traceability
        'X_min': torch.tensor(X_min), #for inverse design rescaling
        'X_max': torch.tensor(X_max), #for inverse design rescaling
        'log_features': log_features #to remember which features were log-scaled for inverse design correction
    }, save_path)
    print(f"Dataset saved to {save_path}")
    print(f"Metadata included for {len(valid_sim_ids)} simulations.")
    print(f"Final dataset shapes - X: {X_final.shape}, Y_real: {Y_real.shape}, Y_imag: {Y_imag.shape} (frequency points, 4, 4)")

if __name__ == "__main__":
    base_dir = os.path.expanduser("~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/data/raw/Universal-Diff-SI-Link")
    output_dir = os.path.expanduser("~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/data/processed/Universal-Diff-SI-Link")
    parse_touchstone_link(base_dir, output_dir)