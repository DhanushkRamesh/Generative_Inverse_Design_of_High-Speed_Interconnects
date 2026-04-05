import sys
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the system path
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJ_ROOT)

from src.data.dataset import get_dataloaders
from src.models.hybridnet import HybridNet

# ---------------------------------------------------------------------------
# Configuration (Must match your training script exactly)
# ---------------------------------------------------------------------------
NUM_POLES = 12
IDX_LENGTH = 7 
DATASET_TYPE = 'link'

def plot_learning_curve():
    """Plots Train and Validation loss from the saved JSON history."""
    history_path = os.path.join(PROJ_ROOT, f"results/checkpoints/training_history_{DATASET_TYPE}.json")
    
    if not os.path.exists(history_path):
        print(f"[ERROR] History file not found at {history_path}. Training must complete at least once.")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(epochs, history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    
    plt.title(f'HybridNet TMPT Learning Curve - {DATASET_TYPE.upper()} Dataset', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Combined Loss (MSE + dB)', fontsize=12)
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    
    # Add a vertical line to show where Curriculum Learning ended and Sobolev/Early Stopping began
    if len(epochs) > 50:
        plt.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='Curriculum Ends / Sobolev Begins')
        
    plt.legend(fontsize=12)
    
    save_dir = os.path.join(PROJ_ROOT, "results/figures")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"learning_curve_hybridnet_{DATASET_TYPE}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Learning curve saved to {save_path}")
    plt.close()

def plot_s_parameters():
    """Runs a test sample through HybridNet and plots the S-parameters against HFSS."""
    print(f"--- Running Inference & Plotting for {DATASET_TYPE.upper()} Dataset ---")
    
    # CPU is perfectly fine for single-sample inference plotting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load the Dataset
    dataset_folder = 'Universal-Diff-SI-Array' if DATASET_TYPE == 'array' else 'Universal-Diff-SI-Link'
    data_path = os.path.join(PROJ_ROOT, f"data/processed/{dataset_folder}/via_{DATASET_TYPE}_dataset.pt")
    _, _, test_loader = get_dataloaders(data_path, dataset_type=DATASET_TYPE, batch_size=1)
    
    # 2. Grab ONE unseen sample from the test set
    x_loc, x_glob, y_r, y_i = next(iter(test_loader))
    x_loc, x_glob = x_loc.to(device), x_glob.to(device)
    
    # 3. Auto-Extract Metadata for Length (Double .dataset bypasses PyTorch Subset wrapper)
    dataset_ref = test_loader.dataset.dataset  
    master_length_idx = dataset_ref.feature_names.index('LENGTH')
    length_mean_val = dataset_ref.X_mean[master_length_idx].item()
    length_std_val = dataset_ref.X_std[master_length_idx].item()

    # 4. Initialize the Model safely
    model = HybridNet(
        num_poles=NUM_POLES,
        num_local_features=x_loc.shape[1],
        num_global_features=x_glob.shape[1],
        num_ports=4,
        length_mean=length_mean_val, 
        length_std=length_std_val   
    ).to(device)

    # 5. Load the Best Checkpoint
    checkpoint_path = os.path.join(PROJ_ROOT, f"results/checkpoints/best_hybridnet_{DATASET_TYPE}.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[ERROR] Checkpoint not found at {checkpoint_path}.")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    num_freqs = y_r.shape[1]
    frequencies_hz = torch.linspace(0.25e9, 100e9, num_freqs).to(device)
    freq_ghz = frequencies_hz.cpu().numpy() / 1e9

    # 6. Run the HybridNet T-Matrix Cascade
    with torch.no_grad():
        # Predict physical params
        poles, residues, d_term, z0_pred, eps_eff_pred = model(x_loc, x_glob)
        
        # Isolate Via
        S_via = model.predict_via_response(poles, residues, d_term, frequencies_hz)
        M_via = model.s_to_abcd(S_via)
        
        # --- UNIT CONVERSION FIX ---
        # Converts normalized length to Inches, then Inches to Meters (* 0.0254)
        length_norm = x_glob[:, IDX_LENGTH]
        length_m = ((length_norm * model.length_std) + model.length_mean) * 0.0254
        
        M_line = model.get_line_matrix(z0_pred, eps_eff_pred, frequencies_hz, length_m)
        
        # Cascade
        M_temp = torch.matmul(M_via, M_line)
        M_total = torch.matmul(M_temp, M_via)
        
        # Back to S-Parameters (Passivity implicitly guaranteed by architecture now)
        S_pred = model.enforce_passivity(model.abcd_to_s(M_total))

    # 7. Convert to numpy and calculate dB magnitude
    S_pred_np = S_pred.cpu().numpy()[0]
    Y_true_np = torch.complex(y_r, y_i).cpu().numpy()[0]
    
    def to_db(s_matrix):
        return 20 * np.log10(np.abs(s_matrix) + 1e-12)

    pred_db = to_db(S_pred_np)
    true_db = to_db(Y_true_np)

    # 8. Plotting the 2x2 Grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'HybridNet TMPT vs Ground Truth (Unseen Test Sample: {DATASET_TYPE.upper()})', fontsize=16, fontweight='bold')

    plots = [
        (0, 0, 'S11 (Return Loss)', axes[0, 0]),
        (0, 1, 'S12 (Insertion Loss)', axes[0, 1]),
        (1, 0, 'S21 (Insertion Loss)', axes[1, 0]),
        (1, 1, 'S22 (Return Loss)', axes[1, 1])
    ]

    for port_out, port_in, title, ax in plots:
        ax.plot(freq_ghz, true_db[:, port_out, port_in], 'k-', label='Ground Truth (HFSS)', linewidth=1.5)
        ax.plot(freq_ghz, pred_db[:, port_out, port_in], 'r--', label='HybridNet (Predicted)', linewidth=1.5)
        
        ax.set_title(title)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_ylim([-60, 5])
        ax.set_xlim([0, 100])
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if port_out == 0 and port_in == 0:
            ax.legend(loc='lower left')

    plt.tight_layout()
    save_path = os.path.join(PROJ_ROOT, f"results/figures/s_params_hybridnet_{DATASET_TYPE}.png")
    plt.savefig(save_path, dpi=300)
    print(f"[SUCCESS] S-Parameter plot saved to {save_path}")
    
    print(f"\n--- AI Physics Extraction ---")
    print(f"Predicted Z0 (Impedance):       {z0_pred[0].item():.2f} Ohms")
    print(f"Predicted Eps_eff (Dielectric): {eps_eff_pred[0].item():.2f}")

if __name__ == "__main__":
    os.makedirs(os.path.join(PROJ_ROOT, "results/figures"), exist_ok=True)
    plot_learning_curve()
    plot_s_parameters()