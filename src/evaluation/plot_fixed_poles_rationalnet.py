import sys
import os
import json
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJ_ROOT)

from src.data.dataset import get_dataloaders
from src.models.rational_net import RationalNet

def plot_fixedpole_learning_curve(dataset_type='link'):
    """Plots the learning curve for fixed-pole training."""
    history_path = os.path.join(PROJ_ROOT, f"results/checkpoints/training_history_fixedpole_{dataset_type}.json")
    
    if not os.path.exists(history_path):
        print(f"  [SKIP] No fixed-pole history found for {dataset_type.upper()}")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(epochs, history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    
    plt.title(f'Learning Curve - {dataset_type.upper()} Dataset (Fixed-Pole RationalNet)', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Combined Loss (MSE + dB)', fontsize=12)
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(fontsize=12)
    
    save_path = os.path.join(PROJ_ROOT, f"results/learning_curve_fixedpole_{dataset_type}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  [INFO] Learning curve saved to {save_path}")
    plt.close()

def plot_fixedpole_s_parameters(dataset_type='link'):
    """Plots S-parameter predictions from the fixed-pole model against ground truth."""
    device = torch.device('cpu')
    
    # Check for checkpoint
    checkpoint_path = os.path.join(PROJ_ROOT, f"results/checkpoints/best_rational_net_fixedpole_{dataset_type}.pth")
    if not os.path.exists(checkpoint_path):
        print(f"  [SKIP] No fixed-pole checkpoint found for {dataset_type.upper()}")
        return
    
    # Load checkpoint to get hyperparameters and shared poles path
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load data
    dataset_folder = 'Universal-Diff-SI-Array' if dataset_type == 'array' else 'Universal-Diff-SI-Link'
    data_path = os.path.join(PROJ_ROOT, f"data/processed/{dataset_folder}/via_{dataset_type}_dataset.pt")
    poles_path = os.path.join(PROJ_ROOT, f"data/processed/{dataset_folder}/shared_poles_{dataset_type}.pt")
    
    if not os.path.exists(data_path) or not os.path.exists(poles_path):
        print(f"  [SKIP] Missing data or poles file for {dataset_type.upper()}")
        return
    
    _, _, test_loader = get_dataloaders(data_path, dataset_type=dataset_type, batch_size=1)
    
    x_loc, x_glob, y_r, y_i = next(iter(test_loader))
    num_local = x_loc.shape[1]
    num_global = x_glob.shape[1]
    num_freqs = y_r.shape[1]
    frequencies_hz = torch.linspace(0.25e9, 100e9, num_freqs)
    f_ghz = frequencies_hz.numpy() / 1e9
    
    # Instantiate model with correct architecture
    model = RationalNet(
        num_poles_half=checkpoint.get('num_poles_half', 40),
        num_local_features=num_local,
        num_global_features=num_global,
        shared_poles_path=poles_path,
        num_ports=2,
        hidden_dim=checkpoint.get('hidden_dim', 512)
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # Predict
    with torch.no_grad():
        poles, residues, d_term = model(x_loc, x_glob)
        S_pred = model.predict_frequency_response(poles, residues, d_term, frequencies_hz)
    
    S_pred_np = S_pred[0].numpy()
    S_true_np = torch.complex(y_r[0, :, :2, :2], y_i[0, :, :2, :2]).numpy()
    
    eps = 1e-10
    S_pred_db = 20 * np.log10(np.abs(S_pred_np) + eps)
    S_true_db = 20 * np.log10(np.abs(S_true_np) + eps)
    
    # S-parameter comparison plot
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'S-Parameter Prediction vs Ground Truth ({dataset_type.upper()}) - Fixed-Pole RationalNet', fontsize=16, fontweight='bold')
    
    indices = [(0,0), (0,1), (1,0), (1,1)]
    titles = ['Sdd11 (Diff Return Loss)', 'Sdd12 (Diff FEXT)', 'Sdd21 (Diff Insertion Loss)', 'Sdd22 (Diff Return Loss)']
    
    for i, (row, col) in enumerate(indices):
        ax = axs[row, col]
        ax.plot(f_ghz, S_true_db[:, row, col], 'k-', linewidth=2, label='Ground Truth (HFSS)')
        ax.plot(f_ghz, S_pred_db[:, row, col], 'r--', linewidth=2, label='Fixed-Pole (Predicted)')
        
        ax.set_title(titles[i], fontsize=12)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.grid(True, ls="--", alpha=0.6)
        ax.set_ylim([-60, 5])
        if i == 0:
            ax.legend()

    plt.tight_layout()
    save_path = os.path.join(PROJ_ROOT, f"results/s_params_comparison_fixedpole_{dataset_type}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  [INFO] S-Parameter plot saved to {save_path}")
    plt.close()

    # Pole map — shows fixed poles (same for all samples) with residue magnitudes
    # to visualise which poles are "active" for this particular geometry
    with torch.no_grad():
        poles_np = poles[0].numpy()
        # Average residue magnitude per pole (across all port combinations)
        res_mag = torch.abs(residues[0]).mean(dim=(-1, -2)).numpy()  # [num_poles]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # Size dots by residue magnitude — large dot = active pole, small = silent
    sizes = 20 + 200 * (res_mag / res_mag.max())
    ax.scatter(poles_np.real, poles_np.imag / (2 * np.pi), c='red', s=sizes,
               alpha=0.7, edgecolors='black', linewidths=0.5)
    ax.axvline(x=0, color='black', linewidth=1, linestyle='-')
    ax.set_xlabel('Real Part (Damping)', fontsize=12)
    ax.set_ylabel('Imaginary Part (Frequency, GHz)', fontsize=12)
    ax.set_title(f'Fixed Pole Map - {dataset_type.upper()} (dot size = residue magnitude)', fontsize=14, fontweight='bold')
    ax.grid(True, ls="--", alpha=0.6)
    
    save_path_poles = os.path.join(PROJ_ROOT, f"results/pole_map_fixedpole_{dataset_type}.png")
    plt.savefig(save_path_poles, dpi=300, bbox_inches='tight')
    print(f"  [INFO] Pole map saved to {save_path_poles}")
    plt.close()

if __name__ == "__main__":
    os.makedirs(os.path.join(PROJ_ROOT, "results"), exist_ok=True)
    
    # Auto-detect which datasets have fixed-pole results
    datasets_to_plot = []
    for dtype in ['array', 'link']:
        checkpoint_path = os.path.join(PROJ_ROOT, f"results/checkpoints/best_rational_net_fixedpole_2port_{dtype}.pth")
        history_path = os.path.join(PROJ_ROOT, f"results/checkpoints/training_history_fixedpole_2port_{dtype}.json")
        if os.path.exists(checkpoint_path) or os.path.exists(history_path):
            datasets_to_plot.append(dtype)
    
    if not datasets_to_plot:
        print("[ERROR] No fixed-pole checkpoints found. Run train_fixedpole.py first.")
    else:
        print(f"[INFO] Found fixed-pole data for: {', '.join(d.upper() for d in datasets_to_plot)}\n")
        for dtype in datasets_to_plot:
            print(f"{'='*60}")
            print(f"  Generating fixed-pole plots for {dtype.upper()} dataset")
            print(f"{'='*60}")
            plot_fixedpole_learning_curve(dataset_type=dtype)
            plot_fixedpole_s_parameters(dataset_type=dtype)
            print()
        
        print("[SUCCESS] All fixed-pole plots generated.")