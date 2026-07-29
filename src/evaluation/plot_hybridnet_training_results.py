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
from src.models.rational_hybrid_net import HybridRationalNet

def plot_hybrid_learning_curve(dataset_type='link'):
    """Plots learning curve and causality ratio over training."""
    history_path = os.path.join(PROJ_ROOT, f"results/checkpoints/training_history_hybrid_{dataset_type}.json")
    
    if not os.path.exists(history_path):
        print(f"  [SKIP] No hybrid history found for {dataset_type.upper()}")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Hybrid RationalNet Training - {dataset_type.upper()} Dataset', fontsize=16, fontweight='bold')
    
    # Loss curve
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    ax1.set_title('Combined Loss', fontsize=12)
    ax1.set_xlabel('Epochs', fontsize=11)
    ax1.set_ylabel('Loss (MSE + dB)', fontsize=11)
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="--", alpha=0.6)
    ax1.legend(fontsize=11)
    
    # Causality ratio over training
    if 'causality_ratio' in history and history['causality_ratio']:
        ax2.plot(epochs, [r * 100 for r in history['causality_ratio']], 
                 color='green', linewidth=2)
        ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% threshold')
        ax2.set_title('Causality Ratio (rational backbone %)', fontsize=12)
        ax2.set_xlabel('Epochs', fontsize=11)
        ax2.set_ylabel('Rational Backbone Contribution (%)', fontsize=11)
        ax2.set_ylim([0, 105])
        ax2.grid(True, ls="--", alpha=0.6)
        ax2.legend(fontsize=11)
    
    plt.tight_layout()
    save_path = os.path.join(PROJ_ROOT, f"results/learning_curve_hybrid_{dataset_type}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  [INFO] Learning curve saved to {save_path}")
    plt.close()

def plot_hybrid_s_parameters(dataset_type='link'):
    """
    Plots S-parameter predictions showing three curves:
    ground truth, total prediction, and rational-only prediction.
    This visualises the contribution of each pathway.
    """
    device = torch.device('cpu')
    
    checkpoint_path = os.path.join(PROJ_ROOT, f"results/checkpoints/best_hybrid_rational_net_{dataset_type}.pth")
    if not os.path.exists(checkpoint_path):
        print(f"  [SKIP] No hybrid checkpoint found for {dataset_type.upper()}")
        return
    
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
    
    model = HybridRationalNet(
        num_poles_half=checkpoint.get('num_poles_half', 40),
        num_local_features=num_local,
        num_global_features=num_global,
        shared_poles_path=poles_path,
        num_ports=4,
        hidden_dim=checkpoint.get('hidden_dim', 512),
        num_freqs=checkpoint.get('num_freqs', num_freqs)
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    with torch.no_grad():
        S_total, S_rational, S_correction, poles, residues = model(x_loc, x_glob, frequencies_hz)
        causal_ratio = model.compute_causality_ratio(S_rational, S_correction)
    
    # Convert to numpy and dB
    S_total_np = S_total[0].numpy()
    S_rational_np = S_rational[0].numpy()
    S_true_np = torch.complex(y_r[0], y_i[0]).numpy()
    
    eps = 1e-10
    S_total_db = 20 * np.log10(np.abs(S_total_np) + eps)
    S_rational_db = 20 * np.log10(np.abs(S_rational_np) + eps)
    S_true_db = 20 * np.log10(np.abs(S_true_np) + eps)
    
    # S-parameter comparison: ground truth vs total vs rational-only
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Hybrid RationalNet - {dataset_type.upper()} (Causal ratio: {causal_ratio:.1%})', 
                 fontsize=16, fontweight='bold')
    
    indices = [(0,0), (0,1), (1,0), (1,1)]
    titles = ['S11 (Return Loss)', 'S12 (Insertion Loss)', 'S21 (Insertion Loss)', 'S22 (Return Loss)']
    
    for i, (row, col) in enumerate(indices):
        ax = axs[row, col]
        ax.plot(f_ghz, S_true_db[:, row, col], 'k-', linewidth=2, label='Ground Truth (HFSS)')
        ax.plot(f_ghz, S_total_db[:, row, col], 'r--', linewidth=2, label='Hybrid (Total)')
        ax.plot(f_ghz, S_rational_db[:, row, col], 'b:', linewidth=1.5, alpha=0.7, label='Rational Only (Causal)')
        
        ax.set_title(titles[i], fontsize=12)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.grid(True, ls="--", alpha=0.6)
        ax.set_ylim([-60, 5])
        if i == 0:
            ax.legend(fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(PROJ_ROOT, f"results/s_params_comparison_hybrid_{dataset_type}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  [INFO] S-Parameter plot saved to {save_path}")
    plt.close()

    # Pole map with residue magnitudes
    with torch.no_grad():
        poles_np = poles[0].numpy()
        res_mag = torch.abs(residues[0]).mean(dim=(-1, -2)).numpy()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sizes = 20 + 200 * (res_mag / (res_mag.max() + 1e-10))
    ax.scatter(poles_np.real, poles_np.imag / (2 * np.pi), c='red', s=sizes,
               alpha=0.7, edgecolors='black', linewidths=0.5)
    ax.axvline(x=0, color='black', linewidth=1, linestyle='-')
    ax.set_xlabel('Real Part (Damping)', fontsize=12)
    ax.set_ylabel('Imaginary Part (Frequency, GHz)', fontsize=12)
    ax.set_title(f'Fixed Pole Map - {dataset_type.upper()} Hybrid (dot size = residue magnitude)', fontsize=14, fontweight='bold')
    ax.grid(True, ls="--", alpha=0.6)
    
    save_path_poles = os.path.join(PROJ_ROOT, f"results/pole_map_hybrid_{dataset_type}.png")
    plt.savefig(save_path_poles, dpi=300, bbox_inches='tight')
    print(f"  [INFO] Pole map saved to {save_path_poles}")
    plt.close()

if __name__ == "__main__":
    os.makedirs(os.path.join(PROJ_ROOT, "results"), exist_ok=True)
    
    datasets_to_plot = []
    for dtype in ['array', 'link']:
        checkpoint_path = os.path.join(PROJ_ROOT, f"results/checkpoints/best_hybrid_rational_net_{dtype}.pth")
        history_path = os.path.join(PROJ_ROOT, f"results/checkpoints/training_history_hybrid_{dtype}.json")
        if os.path.exists(checkpoint_path) or os.path.exists(history_path):
            datasets_to_plot.append(dtype)
    
    if not datasets_to_plot:
        print("[ERROR] No hybrid checkpoints found. Run train_hybridnet.py first.")
    else:
        print(f"[INFO] Found hybrid data for: {', '.join(d.upper() for d in datasets_to_plot)}\n")
        for dtype in datasets_to_plot:
            print(f"{'='*60}")
            print(f"  Generating hybrid plots for {dtype.upper()} dataset")
            print(f"{'='*60}")
            plot_hybrid_learning_curve(dataset_type=dtype)
            plot_hybrid_s_parameters(dataset_type=dtype)
            print()
        
        print("[SUCCESS] All hybrid plots generated.")