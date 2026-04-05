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

def plot_learning_curve(dataset_type='link'):
    """Plots Train and Validation loss from the saved JSON history."""
    history_path = os.path.join(PROJ_ROOT, f"results/checkpoints/training_history_{dataset_type}.json")
    
    if not os.path.exists(history_path):
        print(f"[ERROR] History file not found at {history_path}")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(epochs, history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    
    plt.title(f'Learning Curve - {dataset_type.upper()} Dataset', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Combined Loss (MSE + dB)', fontsize=12)
    plt.yscale('log') # Log scale is best for viewing loss convergence
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(fontsize=12)
    
    save_path = os.path.join(PROJ_ROOT, f"results/learning_curve_{dataset_type}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Learning curve saved to {save_path}")
    plt.close()

def plot_s_parameters(dataset_type='link'):
    """Runs a test sample through the model and plots the S-parameters against ground truth."""
    device = torch.device('cpu') # Inference is fine on CPU
    
    # Load Data
    dataset_folder = 'Universal-Diff-SI-Array' if dataset_type == 'array' else 'Universal-Diff-SI-Link'
    data_path = os.path.join(PROJ_ROOT, f"data/processed/{dataset_folder}/via_{dataset_type}_dataset.pt")
    _, _, test_loader = get_dataloaders(data_path, dataset_type=dataset_type, batch_size=1)
    
    # Grab the first test sample
    x_loc, x_glob, y_r, y_i = next(iter(test_loader))
    num_local = x_loc.shape[1]
    num_global = x_glob.shape[1]
    num_freqs = y_r.shape[1]
    frequencies_hz = torch.linspace(0.25e9, 100e9, num_freqs)
    f_ghz = frequencies_hz.numpy() / 1e9 # For plotting x-axis
    
    # Load Model
    model = RationalNet(num_poles=40, num_local_features=num_local, num_global_features=num_global, num_ports=4)
    checkpoint_path = os.path.join(PROJ_ROOT, f"results/checkpoints/best_rational_net_{dataset_type}.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # Predict
    with torch.no_grad():
        poles, residues, d_term = model(x_loc, x_glob)
        S_pred = model.predict_frequency_response(poles, residues, d_term, frequencies_hz)
    
    # Convert to Numpy and dB
    S_pred_np = S_pred[0].numpy() # Drop batch dimension
    S_true_np = torch.complex(y_r[0], y_i[0]).numpy()
    
    eps = 1e-10
    S_pred_db = 20 * np.log10(np.abs(S_pred_np) + eps)
    S_true_db = 20 * np.log10(np.abs(S_true_np) + eps)
    
    # Plotting 2x2 Grid (S11, S12, S21, S22)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'S-Parameter Prediction vs Ground Truth ({dataset_type.upper()})', fontsize=16, fontweight='bold')
    
    indices = [(0,0), (0,1), (1,0), (1,1)]
    titles = ['S11 (Return Loss)', 'S12 (Insertion Loss)', 'S21 (Insertion Loss)', 'S22 (Return Loss)']
    
    for i, (row, col) in enumerate(indices):
        ax = axs[row, col]
        ax.plot(f_ghz, S_true_db[:, row, col], 'k-', linewidth=2, label='Ground Truth (HFSS)')
        ax.plot(f_ghz, S_pred_db[:, row, col], 'r--', linewidth=2, label='RationalNet (Predicted)')
        
        ax.set_title(titles[i], fontsize=12)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.grid(True, ls="--", alpha=0.6)
        ax.set_ylim([-60, 5])
        if i == 0:
            ax.legend()

    plt.tight_layout()
    save_path = os.path.join(PROJ_ROOT, f"results/s_params_comparison_{dataset_type}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] S-Parameter plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    # Ensure results directory exists
    os.makedirs(os.path.join(PROJ_ROOT, "results"), exist_ok=True)
    
    print("[INFO] Generating plots for Link dataset...")
    plot_learning_curve(dataset_type='link')
    plot_s_parameters(dataset_type='link')
    print("[SUCCESS] All plots generated.")