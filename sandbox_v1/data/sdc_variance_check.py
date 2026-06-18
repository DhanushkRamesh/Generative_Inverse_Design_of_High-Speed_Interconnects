import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load dataset
DATA_PT = Path.home() / "mece_project_inverse_model" / "Generative_Inverse_Design_of_High-Speed_Interconnects" / "data" / "processed" / "Universal-Diff-SI-Array" / "diff_pair_dataset.pt"
payload = torch.load(DATA_PT, weights_only=False)

freqs_ghz = payload["frequencies"].numpy() / 1e9
S_complex = torch.complex(payload["Y_real"].to(torch.float64), payload["Y_imag"].to(torch.float64))

# Extract Sdc11
# S_complex shape is (Batch, Freq, 4, 4). 
# Sdc11 is at Row 0, Col 2.
sdc11_complex = S_complex[:, :, 0, 2] 

# Convert to dB magnitude
sdc11_db = 20 * torch.log10(sdc11_complex.abs() + 1e-12).numpy()

# Calculate standard deviation
sdc11_std = np.std(sdc11_db, axis=0)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(freqs_ghz, sdc11_std, color='darkgreen', label='|Sdc11| std (dB)')
plt.title("Mode Conversion (Sdc) Variance: Does the geometry impact it?")
plt.xlabel("Frequency [GHz]")
plt.ylabel("Standard Deviation [dB]")
plt.grid(True, alpha=0.3)
plt.legend()
out_dir = Path.home() / "mece_project_inverse_model" / "Generative_Inverse_Design_of_High-Speed_Interconnects" / "sandbox_v1" / "data" / "frequency_eda"
out_dir.mkdir(parents=True, exist_ok=True)
save_path = out_dir / "sdc_variance_check.png"

plt.savefig(save_path)
print(f"Sdc Variance plot saved to {save_path}")