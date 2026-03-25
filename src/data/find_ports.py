import skrf as rf
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

def verify_multiple_port_counts(base_dir, target_port_counts):
    variation_dir = os.path.join(base_dir, "variation")
    
    for ports in target_port_counts:
        print(f"\n--- Hunting for a {ports}-port simulation ---")
        # Search specifically for .s8p, .s16p, .s30p, etc.
        search_pattern = os.path.join(variation_dir, "*", f"*.s{ports}p")
        files = glob.glob(search_pattern)
        
        if not files:
            print(f"No {ports}-port files found in this dataset. Skipping...")
            continue
            
        touchstone_file = files[0]
        print(f"Found! Loading raw file: {touchstone_file}")
        
        # Load the network
        ntwk = rf.Network(touchstone_file)
        num_ports = ntwk.number_of_ports
        
        # Plot the signal going from Port 1 to every other port
        plt.figure(figsize=(10, 6))
        
        for rx_port in range(num_ports):
            # Magnitude of S(rx_port, 1) in dB
            s_mag = 20 * np.log10(np.abs(ntwk.s[:, rx_port, 0]) + 1e-12)
            
            # If it starts better than -5 dB, it's the through-path
            if s_mag[0] > -5: 
                plt.plot(ntwk.f / 1e9, s_mag, label=f'S_{rx_port+1}_1 (Likely Through Path)', linewidth=3)
            else:
                plt.plot(ntwk.f / 1e9, s_mag, label=f'S_{rx_port+1}_1', alpha=0.3, linestyle='--')

        # Calculate where the math THINKS the port should be
        math_prediction = (num_ports // 2) + 1
        
        plt.title(f'{ports}-Port Simulation: Signal injected at Port 1\n Math predicts Output is Port {math_prediction}')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Magnitude (dB)')
        plt.ylim(-60, 5)
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.tight_layout()
        plt.show() # Close the plot window to load the next one!

if __name__ == "__main__":
    # Point to the RAW data
    base_dir = os.path.expanduser("~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/data/raw/Universal-Diff-SI-Link")
    
    # We want to test 8, 16, and 30 ports specifically
    test_ports = [8, 16, 80]
    
    verify_multiple_port_counts(base_dir, test_ports)