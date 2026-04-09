import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
MLP Baseline Test Script (Full 4x4 Mixed-Mode Matrix)
=======================================================
This is a self-contained test script to validate the TUHH dataset
is learnable using a standard MLP approach (similar to Konduru et al., 2024).

Trains on the FULL 4x4 mixed-mode S-parameter matrix and reports both:
  - Overall MAE across all 16 elements
  - Per-element MAE for the key differential terms (Sdd11, Sdd21)
  - Separate MAE for differential (Sdd), common-mode (Scc), and cross-mode (Sdc/Scd)

This confirms whether the dataset quality and preprocessing are correct.
"""

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(PROJ_ROOT, 'src'))
from data.dataset import get_dataloaders

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 400
PATIENCE = 50
LR = 1e-3
HIDDEN_DIM = 512
NUM_PORTS = 4           # Full 4x4 mixed-mode matrix
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'mlp_baseline_results')

# ---------------------------------------------------------------------------
# MLP Model
# ---------------------------------------------------------------------------
class BaselineMLP(nn.Module):
    """
    Standard MLP for S-parameter prediction — no physics constraints.
    Predicts real and imaginary parts of the full 4x4 mixed-mode S-parameter
    matrix at each frequency point directly from geometry features.
    """
    def __init__(self, num_input_features, num_freqs, num_ports=4):
        super(BaselineMLP, self).__init__()
        self.num_freqs = num_freqs
        self.num_ports = num_ports
        output_size = num_freqs * num_ports * num_ports * 2  # real + imag

        self.network = nn.Sequential(
            nn.Linear(num_input_features, HIDDEN_DIM),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, output_size)
        )

    def forward(self, x_local, x_global):
        x = torch.cat([x_local, x_global], dim=1)
        out = self.network(x)
        out = out.view(-1, self.num_freqs, self.num_ports, self.num_ports, 2)
        S_pred = torch.complex(out[..., 0], out[..., 1])
        return S_pred

# ---------------------------------------------------------------------------
# Loss Function
# ---------------------------------------------------------------------------
def mlp_loss(S_pred, Y_true):
    """Combined MSE + dB loss — no physics penalties."""
    diff = S_pred - Y_true
    mse = torch.mean(diff.real**2 + diff.imag**2)
    eps = 1e-4
    db_pred = 20 * torch.log10(torch.abs(S_pred) + eps)
    db_true = 20 * torch.log10(torch.abs(Y_true) + eps)
    db_mae = torch.mean(torch.abs(db_pred - db_true))
    return mse + 0.5 * db_mae

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_baseline(dataset_type='array'):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"  MLP Baseline (4x4) — {dataset_type.upper()} — {device.type.upper()}")
    print(f"{'='*70}")

    dataset_folder = 'Universal-Diff-SI-Array' if dataset_type == 'array' else 'Universal-Diff-SI-Link'
    data_path = os.path.join(PROJ_ROOT, f"data/processed/{dataset_folder}/via_{dataset_type}_dataset.pt")

    train_loader, val_loader, test_loader = get_dataloaders(
        data_path, dataset_type=dataset_type, batch_size=BATCH_SIZE
    )

    x_loc, x_glob, y_r, y_i = next(iter(train_loader))
    num_local = x_loc.shape[1]
    num_global = x_glob.shape[1]
    num_features = num_local + num_global
    num_freqs = y_r.shape[1]

    print(f"  Features: {num_features} | Freqs: {num_freqs} | Ports: {NUM_PORTS}")
    print(f"  Output: {num_freqs * NUM_PORTS * NUM_PORTS * 2} values/sample")

    model = BaselineMLP(num_features, num_freqs, NUM_PORTS).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}\n")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}
    best_state = None
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_train = 0.0
        for x_loc, x_glob, y_r, y_i in tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False):
            x_loc, x_glob = x_loc.to(device), x_glob.to(device)
            Y_true = torch.complex(y_r, y_i).to(device)

            optimizer.zero_grad()
            S_pred = model(x_loc, x_glob)
            loss = mlp_loss(S_pred, Y_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train += loss.item() * x_loc.size(0)
        epoch_train /= len(train_loader.dataset)

        model.eval()
        epoch_val = 0.0
        with torch.no_grad():
            for x_loc, x_glob, y_r, y_i in val_loader:
                x_loc, x_glob = x_loc.to(device), x_glob.to(device)
                Y_true = torch.complex(y_r, y_i).to(device)
                S_pred = model(x_loc, x_glob)
                loss = mlp_loss(S_pred, Y_true)
                epoch_val += loss.item() * x_loc.size(0)
        epoch_val /= len(val_loader.dataset)

        scheduler.step(epoch_val)
        history['train_loss'].append(epoch_train)
        history['val_loss'].append(epoch_val)

        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"  [{epoch:03d}/{EPOCHS}] Train: {epoch_train:.4e} | Val: {epoch_val:.4e} | LR: {lr:.2e}")

        if epoch_val < best_val_loss:
            best_val_loss = epoch_val
            epochs_no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  [INFO] Early stopping at epoch {epoch}")
                break

    # ==========================================
    # Evaluation
    # ==========================================
    print(f"\n{'='*70}")
    print(f"  Evaluation (best epoch: {best_epoch})")
    print(f"{'='*70}")

    model.load_state_dict(best_state)
    model.eval()

    total_db_mae = 0.0
    total_phase_mae = 0.0
    element_db_mae = {(i,j): 0.0 for i in range(NUM_PORTS) for j in range(NUM_PORTS)}
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for x_loc, x_glob, y_r, y_i in test_loader:
            x_loc, x_glob = x_loc.to(device), x_glob.to(device)
            Y_true = torch.complex(y_r, y_i).to(device)
            S_pred = model(x_loc, x_glob)

            eps = 1e-4
            db_pred = 20 * torch.log10(torch.abs(S_pred) + eps)
            db_true = 20 * torch.log10(torch.abs(Y_true) + eps)
            db_err = torch.abs(db_pred - db_true)
            phase_err = torch.abs(torch.angle(S_pred) - torch.angle(Y_true)) * (180.0 / torch.pi)

            total_db_mae += torch.mean(db_err).item() * x_loc.size(0)
            total_phase_mae += torch.mean(phase_err).item() * x_loc.size(0)

            for i in range(NUM_PORTS):
                for j in range(NUM_PORTS):
                    element_db_mae[(i,j)] += torch.mean(db_err[:, :, i, j]).item() * x_loc.size(0)

            all_preds.append(S_pred.cpu())
            all_trues.append(Y_true.cpu())

    n_test = len(test_loader.dataset)
    avg_db_mae = total_db_mae / n_test
    avg_phase_mae = total_phase_mae / n_test

    # Compute sub-matrix averages
    diff_mae = sum(element_db_mae[(i,j)] for i in range(2) for j in range(2)) / (4 * n_test)
    cross_mae = sum(element_db_mae[(i,j)] for i in range(2) for j in range(2,4)) / (4 * n_test)
    cross_mae += sum(element_db_mae[(i,j)] for i in range(2,4) for j in range(2)) / (4 * n_test)
    cross_mae /= 2
    common_mae = sum(element_db_mae[(i,j)] for i in range(2,4) for j in range(2,4)) / (4 * n_test)

    print(f"\n  Overall dB MAE (all 16 elements): {avg_db_mae:.2f} dB")
    print(f"  Overall Phase MAE: {avg_phase_mae:.2f}°")

    print(f"\n  Sub-Matrix Breakdown:")
    print(f"    Differential (Sdd, top-left 2x2):  {diff_mae:.2f} dB")
    print(f"    Cross-mode (Sdc/Scd):               {cross_mae:.2f} dB")
    print(f"    Common-mode (Scc, bottom-right 2x2): {common_mae:.2f} dB")

    print(f"\n  Per-Element dB MAE (Differential Quad):")
    diff_names = {(0,0): 'Sdd11 (Return Loss)', (0,1): 'Sdd12 (FEXT)',
                  (1,0): 'Sdd21 (Insertion Loss)', (1,1): 'Sdd22 (Return Loss)'}
    for (i,j), name in diff_names.items():
        print(f"    {name}: {element_db_mae[(i,j)] / n_test:.2f} dB")

    print(f"\n  Physics Constraints:")
    print(f"    Causality:  NOT ENFORCED (standard MLP)")
    print(f"    Passivity:  NOT ENFORCED (standard MLP)")

    # ==========================================
    # Plots
    # ==========================================
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Learning curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['train_loss'], label='Train', color='blue', linewidth=2)
    ax.plot(history['val_loss'], label='Validation', color='orange', linewidth=2)
    ax.set_title(f'MLP Baseline 4x4 - {dataset_type.upper()}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'learning_curve_mlp_4x4_{dataset_type}.png'), dpi=300)
    plt.close()

    # S-parameter plots — first test sample, differential quad only
    S_pred_np = all_preds[0][0].numpy()
    S_true_np = all_trues[0][0].numpy()
    eps = 1e-10
    pred_db = 20 * np.log10(np.abs(S_pred_np) + eps)
    true_db = 20 * np.log10(np.abs(S_true_np) + eps)
    freqs = np.linspace(0.25, 100, S_pred_np.shape[0])

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'MLP Baseline 4x4 vs Ground Truth ({dataset_type.upper()}) — Diff MAE: {diff_mae:.2f} dB',
                 fontsize=16, fontweight='bold')

    titles = ['Sdd11 (Diff Return Loss)', 'Sdd12 (Diff FEXT)',
              'Sdd21 (Diff Insertion Loss)', 'Sdd22 (Diff Return Loss)']
    indices = [(0,0), (0,1), (1,0), (1,1)]

    for idx, (r, c) in enumerate(indices):
        ax = axs[r, c]
        ax.plot(freqs, true_db[:, r, c], 'k-', linewidth=2, label='Ground Truth (HFSS)')
        ax.plot(freqs, pred_db[:, r, c], 'r--', linewidth=2, label='MLP Baseline')
        ax.set_title(titles[idx], fontsize=12)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.grid(True, ls="--", alpha=0.6)
        ax.set_ylim([-60, 5])
        if idx == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f's_params_mlp_4x4_{dataset_type}.png'), dpi=300)
    plt.close()

    # Save summary
    summary = {
        'dataset': dataset_type,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'overall_db_mae': avg_db_mae,
        'differential_mae': diff_mae,
        'cross_mode_mae': cross_mae,
        'common_mode_mae': common_mae,
        'per_element_differential': {f'{diff_names[(i,j)]}': element_db_mae[(i,j)] / n_test
                                     for (i,j) in diff_names},
        'total_params': total_params,
    }
    with open(os.path.join(RESULTS_DIR, f'summary_mlp_4x4_{dataset_type}.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to {RESULTS_DIR}/")
    return avg_db_mae, diff_mae

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {}
    for dtype in ['array', 'link']:
        overall, diff = run_baseline(dataset_type=dtype)
        results[dtype] = {'overall': overall, 'diff': diff}

    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Dataset':<10} {'MLP Overall':<15} {'MLP Diff Only':<15} {'Rational (4port)':<18}")
    print(f"  {'─'*58}")
    rational = {'array': 7.74, 'link': 13.94}
    for d in results:
        print(f"  {d.upper():<10} {results[d]['overall']:<15.2f} {results[d]['diff']:<15.2f} {rational[d]:<18.2f}")
    print(f"\n  'MLP Overall' = all 16 elements of 4x4 mixed-mode matrix")
    print(f"  'MLP Diff Only' = Sdd quad (top-left 2x2) — comparable to Konduru/Sreekumar")
    print(f"  'Rational' = your fixed-pole model on full 4x4")