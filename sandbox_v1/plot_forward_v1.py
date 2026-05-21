import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("=== PLOTTING SURROGATE PREDICTIONS ===")
    sandbox_root = Path(__file__).resolve().parent
    if str(sandbox_root) not in sys.path: sys.path.insert(0, str(sandbox_root))
    
    from models.forward_rational_net import ForwardRationalSurrogate
    
    dataset_path = sandbox_root.parent / "data" / "processed" / "Universal-Diff-SI-Array" / "diff_pair_dataset.pt"
    poles_path = sandbox_root / "data" / "universal_pole_basis.pt"
    model_path = sandbox_root.parent / "results" / "models" / "best_forward_surrogate.pth"
    fig_dir = sandbox_root.parent / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    data = torch.load(dataset_path, weights_only=False)
    poles = torch.load(poles_path, weights_only=False)
    freqs_hz = data["frequencies"]
    
    X_all = torch.cat([data["X_local"], data["X_global"], data["X_context"]], dim=1)
    Y_real, Y_imag = data["Y_real"], data["Y_imag"]
    
    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForwardRationalSurrogate(in_features=21, n_poles=len(poles)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Pick 3 random validation samples (sim_id > 1000 usually puts it late in the dataset)
    rng = np.random.default_rng(42)
    indices = rng.choice(len(X_all), size=3, replace=False)
    
    freqs_dev = freqs_hz.to(device)
    poles_dev = poles.to(device)
    f_ghz = freqs_hz.numpy() / 1e9
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            x = X_all[idx:idx+1].to(device)
            y_true = torch.complex(Y_real[idx:idx+1], Y_imag[idx:idx+1]).to(device)
            
            y_pred = model(x, freqs_dev, poles_dev)
            
            p_db = 20 * torch.log10(torch.clamp(torch.abs(y_pred[0]), min=1e-5)).cpu().numpy()
            t_db = 20 * torch.log10(torch.clamp(torch.abs(y_true[0]), min=1e-5)).cpu().numpy()
            
            # Plot Sdd11 (Return Loss)
            axes[i, 0].plot(f_ghz, t_db[:, 0, 0], 'b-', lw=2, label="Ground Truth (HFSS)")
            axes[i, 0].plot(f_ghz, p_db[:, 0, 0], 'r--', lw=1.5, label="Neural Surrogate")
            axes[i, 0].set_title(f"Sample {idx} - Sdd11 (Return Loss)")
            axes[i, 0].set_ylabel("Magnitude (dB)")
            axes[i, 0].grid(alpha=0.3)
            axes[i, 0].set_ylim([-60, 0])
            
            # Plot Sdd21 (Insertion Loss)
            axes[i, 1].plot(f_ghz, t_db[:, 1, 0], 'b-', lw=2, label="Ground Truth (HFSS)")
            axes[i, 1].plot(f_ghz, p_db[:, 1, 0], 'r--', lw=1.5, label="Neural Surrogate")
            axes[i, 1].set_title(f"Sample {idx} - Sdd21 (Insertion Loss)")
            axes[i, 1].grid(alpha=0.3)
            axes[i, 1].set_ylim([-80, 0])
            
            if i == 0:
                axes[i, 0].legend()
                axes[i, 1].legend()

    axes[2, 0].set_xlabel("Frequency (GHz)")
    axes[2, 1].set_xlabel("Frequency (GHz)")
    
    fig.tight_layout()
    save_path = fig_dir / "surrogate_validation_check.png"
    fig.savefig(save_path, dpi=150)
    print(f"Plot saved to: {save_path}")

if __name__ == "__main__":
    main()