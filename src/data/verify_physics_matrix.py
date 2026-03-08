import torch
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
import os

def verify_extracted_physics(pt_file_path):
    print(f"Loading dataset from: {pt_file_path}")
    data = torch.load(pt_file_path, weights_only=True)
    
    # Grab the very first board design
    frequencies = data['frequencies'].numpy()
    Y_real = data['Y_real'][0].numpy()
    Y_imag = data['Y_imag'][0].numpy()
    
    # Reconstruct the Complex 4x4 S-Parameter Matrix
    S_4x4 = Y_real + 1j * Y_imag
    freq = rf.Frequency.from_f(frequencies, unit='hz')
    ntwk = rf.Network(frequency=freq, s=S_4x4)
    
    # 1. FIX THE TDR WARNING: Extrapolate to DC (0 Hz)
    ntwk = ntwk.extrapolate_to_dc(kind='linear')
    
    # 2. CORRECT MATHEMATICAL PORT MAPPING FOR OUR NEW MATRIX:
    # Because of our dynamic parser, the 4x4 matrix is always perfectly standard:
    # Index 0 (Port 1): TX+ (Near-End Input 1)
    # Index 1 (Port 2): TX- (Near-End Input 2)
    # Index 2 (Port 3): RX+ (Far-End Output 1)
    # Index 3 (Port 4): RX- (Far-End Output 2)
    
    # Differential Insertion Loss S_DD21 = 0.5 * (S31 - S32 - S41 + S42)
    s31 = ntwk.s[:, 2, 0]
    s32 = ntwk.s[:, 2, 1]
    s41 = ntwk.s[:, 3, 0]
    s42 = ntwk.s[:, 3, 1]
    
    S_DD21_complex = 0.5 * (s31 - s32 - s41 + s42)
    S_DD21_mag_dB = 20 * np.log10(np.abs(S_DD21_complex) + 1e-12)
    
    # Time Domain Step Response (Using the true through-path S31)
    ntwk_s31 = rf.Network(frequency=ntwk.frequency, s=s31)
    t, step_response = ntwk_s31.step_response()

    # --- PLOTTING THE PROOF ---
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Differential Insertion Loss
    plt.subplot(1, 2, 1)
    plt.plot(ntwk.f / 1e9, S_DD21_mag_dB, color='blue', linewidth=2)
    plt.title('Verified: Differential Insertion Loss ($S_{DD21}$)')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.ylim(-40, 5)
    
    # Plot 2: Time Domain Step Response (Transmission)
    plt.subplot(1, 2, 2)
    plt.plot(t * 1e9, step_response, color='green', linewidth=2)
    plt.title('Verified: Time-Domain Step Response (Transmission)')
    plt.xlabel('Time (Nanoseconds)')
    plt.ylabel('Voltage (V)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_path = os.path.expanduser("~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/data/processed/Universal-Diff-SI-Link/via_link_dataset.pt")
    verify_extracted_physics(test_path)