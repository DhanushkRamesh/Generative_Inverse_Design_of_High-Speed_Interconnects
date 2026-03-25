import skrf as rf
import matplotlib.pyplot as plt
import os

# Added "../" to go up one level to the root directory before going into data/
file_path = "../data/raw/Universal-Diff-SI-Array/variation/sim_pkg_3146/sim_pkg_3146.s16p"

print(f"Loading {file_path}...")
network = rf.Network(file_path)

print(f"Successfully loaded a {network.number_of_ports}-port network!")
print(f"Frequency range: {network.f[0]/1e9} GHz to {network.f[-1]/1e9} GHz")

plt.figure(figsize=(10, 5))

# Plotting S11 (Return Loss of Port 1)
network.plot_s_db(m=0, n=0, label='S11 (Return Loss)')

# To find S21 (Insertion Loss), we plot Port 2 and Port 9
network.plot_s_db(m=1, n=0, label='Transmission (If Port 2 is bottom)')
network.plot_s_db(m=8, n=0, label='Transmission (If Port 9 is bottom)')

plt.title('GROUND TRUTH: TUHH sim_pkg_3146 (Port 1)')
plt.ylim([-50, 5])
plt.grid(True)
plt.show()