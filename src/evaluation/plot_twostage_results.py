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

def plot_twostage_learning_curve(dataset_type='link'):
    """Plots the two-stage learning curves (Stage 2 + Stage 3) side by side."""
    history_path = os.path.join(PROJ_ROOT, f"results/checkpoints/training_history_2stage_{dataset_type}.json")
    
    if not os.path.exists(history_path):
        print(f"  [SKIP] No 2-stage history found for {dataset_type.upper()}")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Two-Stage Learning Curve - {dataset_type.upper()} Dataset', fontsize=16, fontweight='bold')
    
    # Stage 2: Pole/Residue supervision
    s2_epochs = range(1, len(history['stage2_train_loss']) + 1)
    ax1.plot(s2_epochs, history['stage2_train_loss'], label='Train Loss', color='blue', linewidth=2)
    ax1.plot(s2_epochs, history['stage2_val_loss'], label='Validation Loss', color='orange', linewidth=2)
    ax1.set_title('Stage 2: Pole/Residue Supervision', fontsize=12)
    ax1.set_xlabel('Epochs', fontsize=11)
    ax1.set_ylabel('Pole/Residue MSE Loss', fontsize=11)
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="--", alpha=0.6)
    ax1.legend(fontsize=11)
    
    # Stage 3: End-to-end fine-tuning
    s3_epochs = range(1, len(history['stage3_train_loss']) + 1)
    ax2.plot(s3_epochs, history['stage3_train_loss'], label='Train Loss', color='blue', linewidth=2)
    ax2.plot(s3_epochs, history['stage3_val_loss'], label='Validation Loss', color='orange', linewidth=2)
    ax2.set_title('Stage 3: End-to-End Fine-Tuning', fontsize=12)
    ax2.set_xlabel('Epochs', fontsize=11)
    ax2.set_ylabel('Combined Loss (MSE + dB)', fontsize=11)
    ax2.set_yscale('log')
    ax2.grid(True, which="both", ls="--", alpha=0.6)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    save_path = os.path.join(PROJ_ROOT, f"results/learning_curve_2stage_{dataset_type}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  [INFO] Two-stage learning curve saved to {save_path}")
    plt.close()

def plot_twostage_s_parameters(dataset_type='link'):
    """Plots S-parameter predictions from the two-stage model against ground truth."""
    device = torch.device('cpu')
    
    # Check for checkpoint
    checkpoint_path = os.path.join(PROJ_ROOT, f"results/checkpoints/best_rational_net_2stage_{dataset_type}.pth")
    if not os.path.exists(checkpoint_path):
        print(f"  [SKIP] No 2-stage checkpoint found for {dataset_type.upper()}")
        return
    
    # Load data
    dataset_folder = 'Universal-Diff-SI-Array' if dataset_type == 'array' else 'Universal-Diff-SI-Link'
    data_path = os.path.join(PROJ_ROOT, f"data/processed/{dataset_folder}/via_{dataset_type}_dataset.pt")
    
    if not os.path.exists(data_path):
        print(f"  [SKIP] No dataset found for {dataset_type.upper()}")
        return
    
    _, _, test_loader = get_dataloaders(data_path, dataset_type=dataset_type, batch_size=1)
    
    x_loc, x_glob, y_r, y_i = next(iter(test_loader))
    num_local = x_loc.shape[1]
    num_global = x_glob.shape[1]
    num_freqs = y_r.shape[1]
    frequencies_hz = torch.linspace(0.25e9, 100e9, num_freqs)
    f_ghz = frequencies_hz.numpy() / 1e9
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = RationalNet(
        num_poles=checkpoint.get('num_poles', 80),
        num_local_features=num_local,
        num_global_features=num_global,
        num_ports=4,
        hidden_dim=checkpoint.get('hidden_dim', 512)
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # Predict
    with torch.no_grad():
        poles, residues, d_term = model(x_loc, x_glob)
        S_pred = model.predict_frequency_response(poles, residues, d_term, frequencies_hz)
    
    S_pred_np = S_pred[0].numpy()
    S_true_np = torch.complex(y_r[0], y_i[0]).numpy()
    
    eps = 1e-10
    S_pred_db = 20 * np.log10(np.abs(S_pred_np) + eps)
    S_true_db = 20 * np.log10(np.abs(S_true_np) + eps)
    
    # S-parameter comparison plot
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'S-Parameter Prediction vs Ground Truth ({dataset_type.upper()}) - Two-Stage RationalNet', fontsize=16, fontweight='bold')
    
    indices = [(0,0), (0,1), (1,0), (1,1)]
    titles = ['S11 (Return Loss)', 'S12 (Insertion Loss)', 'S21 (Insertion Loss)', 'S22 (Return Loss)']
    
    for i, (row, col) in enumerate(indices):
        ax = axs[row, col]
        ax.plot(f_ghz, S_true_db[:, row, col], 'k-', linewidth=2, label='Ground Truth (HFSS)')
        ax.plot(f_ghz, S_pred_db[:, row, col], 'r--', linewidth=2, label='Two-Stage (Predicted)')
        
        ax.set_title(titles[i], fontsize=12)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.grid(True, ls="--", alpha=0.6)
        ax.set_ylim([-60, 5])
        if i == 0:
            ax.legend()

    plt.tight_layout()
    save_path = os.path.join(PROJ_ROOT, f"results/s_params_comparison_2stage_{dataset_type}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  [INFO] S-Parameter plot saved to {save_path}")
    plt.close()

    # Pole map
    with torch.no_grad():
        poles_np = poles[0].numpy()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.scatter(poles_np.real, poles_np.imag / (2 * np.pi), c='red', s=40, alpha=0.7, edgecolors='black', linewidths=0.5)
    ax.axvline(x=0, color='black', linewidth=1, linestyle='-')
    ax.set_xlabel('Real Part (Damping)', fontsize=12)
    ax.set_ylabel('Imaginary Part (Frequency, GHz)', fontsize=12)
    ax.set_title(f'Pole Map - {dataset_type.upper()} Dataset (Two-Stage)', fontsize=14, fontweight='bold')
    ax.grid(True, ls="--", alpha=0.6)
    
    save_path_poles = os.path.join(PROJ_ROOT, f"results/pole_map_2stage_{dataset_type}.png")
    plt.savefig(save_path_poles, dpi=300, bbox_inches='tight')
    print(f"  [INFO] Pole map saved to {save_path_poles}")
    plt.close()

if __name__ == "__main__":
    os.makedirs(os.path.join(PROJ_ROOT, "results"), exist_ok=True)
    
    # Auto-detect which datasets have two-stage results
    datasets_to_plot = []
    for dtype in ['array', 'link']:
        checkpoint_path = os.path.join(PROJ_ROOT, f"results/checkpoints/best_rational_net_2stage_{dtype}.pth")
        history_path = os.path.join(PROJ_ROOT, f"results/checkpoints/training_history_2stage_{dtype}.json")
        if os.path.exists(checkpoint_path) or os.path.exists(history_path):
            datasets_to_plot.append(dtype)
    
    if not datasets_to_plot:
        print("[ERROR] No two-stage checkpoints found. Run train_twostage.py first.")
    else:
        print(f"[INFO] Found two-stage data for: {', '.join(d.upper() for d in datasets_to_plot)}\n")
        for dtype in datasets_to_plot:
            print(f"{'='*60}")
            print(f"  Generating two-stage plots for {dtype.upper()} dataset")
            print(f"{'='*60}")
            plot_twostage_learning_curve(dataset_type=dtype)
            plot_twostage_s_parameters(dataset_type=dtype)
            print()
        
        print("[SUCCESS] All two-stage plots generated.")