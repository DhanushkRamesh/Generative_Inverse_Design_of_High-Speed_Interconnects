import skrf as rf
import matplotlib.pyplot as plt
import os

# Path to the specific variation folder
file_path = "../../data/raw/Universal-Diff-SI-Array/variation/sim_pkg_3146/sim_pkg_3146.s16p"

if not os.path.exists(file_path):
    print(f"Error: File {file_path} does not exist. Please check the path and try again.")
    exit(1)
else:
    print(f"Loading {file_path}...")
    network = rf.Network(file_path)
    print(f"Successfully loaded a {network.number_of_ports}-port network!")
    print(f"Frequency range: {network.f[0]/1e9} GHz to {network.f[-1]/1e9} GHz")

    plt.figure(figsize=(10, 6))
    # S11: Return Loss (Signal bouncing back from the top of the via)
    network.plot_s_db(m=0, n=0, label='GT S11 (Return Loss)', color='blue')

    # S91: Insertion Loss (Signal making it through to the bottom of the via)
    # n=0 (Port 1), m=8 (Port 9)
    network.plot_s_db(m=8, n=0, label='GT S91 (Insertion Loss)', color='orange', linewidth=2)

    plt.title('GROUND TRUTH (Fixed Ports): TUHH sim_pkg_3146')
    plt.ylabel('Magnitude (dB)')
    plt.xlabel('Frequency (GHz)')
    plt.ylim([-50, 5])
    plt.grid(True)
    plt.legend()
    plt.show()