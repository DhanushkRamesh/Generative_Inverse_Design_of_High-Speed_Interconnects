import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

"""
Frequency-Conditioned MLP Test
================================
Instead of predicting all 401 frequency points at once from geometry alone,
this model takes [geometry, frequency] as input and predicts the S-parameter
matrix at THAT SINGLE frequency point.

Why this changes everything:
  - Current approach: 15 features → 12,832 outputs (401 × 4 × 4 × 2)
    with 1,912 training samples. That's a 15-to-12832 mapping.
  - This approach: 16 features (15 + frequency) → 32 outputs (4 × 4 × 2)
    with 1,912 × 401 = 767,012 training pairs.
    
  The model sees 400× more training examples and the output dimensionality
  drops by 400×. Sharp resonances become learnable because the network
  sees the frequency value explicitly and can learn "at 25 GHz with
  this via radius, there is a dip."

  This is inspired by Schultz et al. (2023) who used frequency as an
  input feature for predicting frequency response functions of
  multimass oscillators, achieving high accuracy with limited samples.

Place in: tests/test_freq_conditioned_mlp.py
Run: python tests/test_freq_conditioned_mlp.py
"""

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(PROJ_ROOT, 'src'))
from data.dataset import get_dataloaders

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BATCH_SIZE = 512        # Larger batch size — we have 767K training pairs
EPOCHS = 100            # Fewer epochs needed — much more data per epoch
PATIENCE = 15
LR = 1e-3
HIDDEN_DIM = 256        # Smaller model — each prediction is just 32 values
NUM_PORTS = 4           # Full 4x4 mixed-mode matrix
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'freq_conditioned_results')

# ---------------------------------------------------------------------------
# Frequency-Conditioned Dataset
# ---------------------------------------------------------------------------
class FreqConditionedDataset(Dataset):
    """
    Explodes each sample into num_freqs training pairs.
    Each pair is: (geometry_features + frequency_value) → S-parameter at that frequency.
    
    A dataset with 1912 samples × 401 frequencies = 767,012 training pairs.
    This is the key insight — we're not changing the physics or the data,
    just reformulating how the network sees it.
    """
    def __init__(self, data_path, dataset_type='array'):
        data = torch.load(data_path, weights_only=False)
        feature_names = data['feature_names']
        
        # Feature splitting (same as your SIPIDataset)
        if dataset_type == 'link':
            local_features = ['VIA_RADIUS', 'PITCH', 'ANTIPAD_RADIUS', 'TMET', 'TDIEL',
                              'CONDUCTIVITY', 'PERMITTIVITY', 'LOSSTANGENT', 'SL_WIDTH']
            global_features = ['VIAS_X_AMOUNT', 'VIAS_Y_AMOUNT', 'SIGNAL_AMOUNT',
                               'GROUND_AMOUNT', 'POWER_AMOUNT', 'LAYER_AMOUNT', 'NUM_PORTS', 'LENGTH']
        else:
            local_features = ['VIA_RADIUS', 'PITCH', 'ANTIPAD_RADIUS', 'TMET', 'TDIEL',
                              'CONDUCTIVITY', 'PERMITTIVITY', 'LOSSTANGENT']
            global_features = ['VIAS_X_AMOUNT', 'VIAS_Y_AMOUNT', 'SIGNAL_AMOUNT',
                               'GROUND_AMOUNT', 'POWER_AMOUNT', 'LAYER_AMOUNT', 'NUM_PORTS']
        
        local_idx = [feature_names.index(f) for f in local_features if f in feature_names]
        global_idx = [feature_names.index(f) for f in global_features if f in feature_names]
        
        X = data['X']
        self.x_local = X[:, local_idx]    # [num_samples, num_local]
        self.x_global = X[:, global_idx]  # [num_samples, num_global]
        self.y_real = data['Y_real']      # [num_samples, num_freqs, 4, 4]
        self.y_imag = data['Y_imag']      # [num_samples, num_freqs, 4, 4]
        
        self.num_samples = X.shape[0]
        self.num_freqs = self.y_real.shape[1]
        self.num_local = len(local_idx)
        self.num_global = len(global_idx)
        
        # Create normalised frequency values [0, 1] for each frequency point
        # Normalised so the network sees frequency in the same scale as other features
        self.freq_values = torch.linspace(0, 1, self.num_freqs, dtype=torch.float32)
        
        # Total number of (sample, frequency) pairs
        self.total_pairs = self.num_samples * self.num_freqs
        
        print(f"  FreqConditioned Dataset: {self.num_samples} samples × {self.num_freqs} freqs = {self.total_pairs:,} pairs")
    
    def __len__(self):
        return self.total_pairs
    
    def __getitem__(self, idx):
        # Convert flat index to (sample_idx, freq_idx)
        sample_idx = idx // self.num_freqs
        freq_idx = idx % self.num_freqs
        
        # Geometry features
        x_loc = self.x_local[sample_idx]     # [num_local]
        x_glob = self.x_global[sample_idx]   # [num_global]
        
        # Frequency value as a feature
        freq_val = self.freq_values[freq_idx].unsqueeze(0)  # [1]
        
        # Concatenate: [local_features, global_features, frequency]
        x_combined = torch.cat([x_loc, x_glob, freq_val])   # [num_local + num_global + 1]
        
        # Target: S-parameter at this single frequency point
        y_r = self.y_real[sample_idx, freq_idx]  # [4, 4]
        y_i = self.y_imag[sample_idx, freq_idx]  # [4, 4]
        
        return x_combined, y_r, y_i

# ---------------------------------------------------------------------------
# Frequency-Conditioned MLP
# ---------------------------------------------------------------------------
class FreqConditionedMLP(nn.Module):
    """
    Takes geometry + frequency as input, outputs S-parameter at that frequency.
    Input:  [num_local + num_global + 1] (the +1 is frequency)
    Output: [num_ports × num_ports × 2] (real + imag of S-matrix at one freq)
    
    This is a MUCH simpler mapping than predicting all 401 points at once.
    The network can learn frequency-dependent features — e.g., "at high
    frequencies, this geometry has strong resonances" — because frequency
    is an explicit input.
    """
    def __init__(self, num_input_features, num_ports=4):
        super(FreqConditionedMLP, self).__init__()
        self.num_ports = num_ports
        output_size = num_ports * num_ports * 2  # real + imag
        
        self.network = nn.Sequential(
            nn.Linear(num_input_features, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, output_size)
        )
    
    def forward(self, x):
        out = self.network(x)
        # Reshape to [batch, num_ports, num_ports, 2]
        out = out.view(-1, self.num_ports, self.num_ports, 2)
        return out

# ---------------------------------------------------------------------------
# Loss Function
# ---------------------------------------------------------------------------
def freq_loss(pred, y_r, y_i):
    """Combined MSE + dB loss on single-frequency predictions."""
    S_pred = torch.complex(pred[..., 0], pred[..., 1])
    S_true = torch.complex(y_r, y_i)
    
    # Complex MSE
    diff = S_pred - S_true
    mse = torch.mean(diff.real**2 + diff.imag**2)
    
    # dB magnitude loss
    eps = 1e-4
    db_pred = 20 * torch.log10(torch.abs(S_pred) + eps)
    db_true = 20 * torch.log10(torch.abs(S_true) + eps)
    db_mae = torch.mean(torch.abs(db_pred - db_true))
    
    return mse + 0.5 * db_mae

# ---------------------------------------------------------------------------
# Evaluation: reconstruct full frequency response and compute MAE
# ---------------------------------------------------------------------------
def evaluate_full_response(model, data_path, dataset_type, device):
    """
    Evaluates by reconstructing the full frequency response for each test sample.
    Iterates over all 401 frequencies per sample and assembles the prediction.
    """
    # Load original data for proper test split comparison
    data = torch.load(data_path, weights_only=False)
    feature_names = data['feature_names']
    
    if dataset_type == 'link':
        local_features = ['VIA_RADIUS', 'PITCH', 'ANTIPAD_RADIUS', 'TMET', 'TDIEL',
                          'CONDUCTIVITY', 'PERMITTIVITY', 'LOSSTANGENT', 'SL_WIDTH']
        global_features = ['VIAS_X_AMOUNT', 'VIAS_Y_AMOUNT', 'SIGNAL_AMOUNT',
                           'GROUND_AMOUNT', 'POWER_AMOUNT', 'LAYER_AMOUNT', 'NUM_PORTS', 'LENGTH']
    else:
        local_features = ['VIA_RADIUS', 'PITCH', 'ANTIPAD_RADIUS', 'TMET', 'TDIEL',
                          'CONDUCTIVITY', 'PERMITTIVITY', 'LOSSTANGENT']
        global_features = ['VIAS_X_AMOUNT', 'VIAS_Y_AMOUNT', 'SIGNAL_AMOUNT',
                           'GROUND_AMOUNT', 'POWER_AMOUNT', 'LAYER_AMOUNT', 'NUM_PORTS']
    
    local_idx = [feature_names.index(f) for f in local_features if f in feature_names]
    global_idx = [feature_names.index(f) for f in global_features if f in feature_names]
    
    X = data['X']
    Y_real = data['Y_real']
    Y_imag = data['Y_imag']
    num_samples = X.shape[0]
    num_freqs = Y_real.shape[1]
    
    # Use same test split as get_dataloaders (seed=42, 80/10/10)
    n_train = int(0.8 * num_samples)
    n_val = int(0.1 * num_samples)
    n_test = num_samples - n_train - n_val
    
    generator = torch.Generator().manual_seed(42)
    all_indices = torch.randperm(num_samples, generator=generator)
    test_indices = all_indices[n_train + n_val:]
    
    freq_values = torch.linspace(0, 1, num_freqs, dtype=torch.float32)
    
    model.eval()
    
    total_db_mae = 0.0
    diff_db_mae = 0.0
    element_mae = {(i,j): 0.0 for i in range(NUM_PORTS) for j in range(NUM_PORTS)}
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for test_idx in tqdm(test_indices[:min(50, len(test_indices))], desc="Evaluating", leave=False):
            # Get geometry features for this sample
            x_loc = X[test_idx, local_idx]
            x_glob = X[test_idx, global_idx]
            
            # Predict at all frequencies by batching frequency queries
            # Build input: [num_freqs, num_features + 1]
            x_geom = torch.cat([x_loc, x_glob]).unsqueeze(0).expand(num_freqs, -1)  # [401, 15]
            freq_col = freq_values.unsqueeze(1)  # [401, 1]
            x_input = torch.cat([x_geom, freq_col], dim=1).to(device)  # [401, 16]
            
            # Predict all frequencies at once
            pred = model(x_input)  # [401, 4, 4, 2]
            S_pred = torch.complex(pred[..., 0], pred[..., 1]).cpu()  # [401, 4, 4]
            S_true = torch.complex(Y_real[test_idx], Y_imag[test_idx])  # [401, 4, 4]
            
            eps = 1e-4
            db_pred = 20 * torch.log10(torch.abs(S_pred) + eps)
            db_true = 20 * torch.log10(torch.abs(S_true) + eps)
            db_err = torch.abs(db_pred - db_true)
            
            total_db_mae += torch.mean(db_err).item()
            diff_db_mae += torch.mean(db_err[:, :2, :2]).item()
            
            for i in range(NUM_PORTS):
                for j in range(NUM_PORTS):
                    element_mae[(i,j)] += torch.mean(db_err[:, i, j]).item()
            
            all_preds.append(S_pred.numpy())
            all_trues.append(S_true.numpy())
    
    n_eval = min(50, len(test_indices))
    total_db_mae /= n_eval
    diff_db_mae /= n_eval
    for k in element_mae:
        element_mae[k] /= n_eval
    
    return total_db_mae, diff_db_mae, element_mae, all_preds, all_trues

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_freq_conditioned(dataset_type='array'):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"  Frequency-Conditioned MLP — {dataset_type.upper()} — {device.type.upper()}")
    print(f"{'='*70}")
    
    dataset_folder = 'Universal-Diff-SI-Array' if dataset_type == 'array' else 'Universal-Diff-SI-Link'
    data_path = os.path.join(PROJ_ROOT, f"data/processed/{dataset_folder}/via_{dataset_type}_dataset.pt")
    
    # Create frequency-conditioned dataset
    full_dataset = FreqConditionedDataset(data_path, dataset_type=dataset_type)
    
    num_input = full_dataset.num_local + full_dataset.num_global + 1  # +1 for frequency
    print(f"  Input features: {num_input} ({full_dataset.num_local} local + {full_dataset.num_global} global + 1 freq)")
    print(f"  Output: {NUM_PORTS}×{NUM_PORTS}×2 = {NUM_PORTS*NUM_PORTS*2} values per prediction")
    
    # Split into train/val/test BY SAMPLE (not by pair)
    # This ensures test samples are never seen during training at ANY frequency
    num_samples = full_dataset.num_samples
    num_freqs = full_dataset.num_freqs
    n_train_samples = int(0.8 * num_samples)
    n_val_samples = int(0.1 * num_samples)
    n_test_samples = num_samples - n_train_samples - n_val_samples
    
    # Generate sample-level split indices
    generator = torch.Generator().manual_seed(42)
    sample_perm = torch.randperm(num_samples, generator=generator)
    train_samples = sample_perm[:n_train_samples]
    val_samples = sample_perm[n_train_samples:n_train_samples + n_val_samples]
    
    # Convert sample indices to pair indices
    # Each sample s has pairs [s*num_freqs, s*num_freqs+1, ..., s*num_freqs+num_freqs-1]
    train_pairs = []
    for s in train_samples:
        train_pairs.extend(range(s.item() * num_freqs, (s.item() + 1) * num_freqs))
    val_pairs = []
    for s in val_samples:
        val_pairs.extend(range(s.item() * num_freqs, (s.item() + 1) * num_freqs))
    
    train_set = torch.utils.data.Subset(full_dataset, train_pairs)
    val_set = torch.utils.data.Subset(full_dataset, val_pairs)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"  Train pairs: {len(train_pairs):,} ({n_train_samples} samples × {num_freqs} freqs)")
    print(f"  Val pairs:   {len(val_pairs):,} ({n_val_samples} samples × {num_freqs} freqs)")
    print(f"  Test samples: {n_test_samples} (evaluated as full frequency sweeps)")
    
    # Model
    model = FreqConditionedMLP(num_input, NUM_PORTS).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Architecture: 3-layer MLP ({num_input}→{HIDDEN_DIM}→{HIDDEN_DIM}→{HIDDEN_DIM}→{NUM_PORTS*NUM_PORTS*2})\n")
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val = float('inf')
    no_improve = 0
    best_state = None
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_train = 0.0
        n_batches = 0
        for x, y_r, y_i in tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False):
            x = x.to(device)
            y_r, y_i = y_r.to(device), y_i.to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = freq_loss(pred, y_r, y_i)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train += loss.item()
            n_batches += 1
        epoch_train /= n_batches
        
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for x, y_r, y_i in val_loader:
                x = x.to(device)
                y_r, y_i = y_r.to(device), y_i.to(device)
                val_loss += freq_loss(model(x), y_r, y_i).item()
                n_val_batches += 1
        val_loss /= n_val_batches
        
        scheduler.step(val_loss)
        history['train_loss'].append(epoch_train)
        history['val_loss'].append(val_loss)
        
        if epoch % 5 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"  [{epoch:03d}/{EPOCHS}] Train: {epoch_train:.4e} | Val: {val_loss:.4e} | LR: {lr:.2e}")
        
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  [INFO] Early stopping at epoch {epoch}")
                break
    
    # ==========================================
    # Evaluation — reconstruct full frequency responses
    # ==========================================
    print(f"\n{'='*70}")
    print(f"  Evaluation (best epoch: {best_epoch})")
    print(f"{'='*70}")
    
    model.load_state_dict(best_state)
    
    total_mae, diff_mae, element_mae, all_preds, all_trues = evaluate_full_response(
        model, data_path, dataset_type, device
    )
    
    print(f"\n  Overall MAE (all 16 elements): {total_mae:.2f} dB")
    print(f"  Diff MAE (Sdd 2x2):           {diff_mae:.2f} dB")
    print(f"\n  Per-Element dB MAE (Differential):")
    diff_names = {(0,0): 'Sdd11 (Return Loss)', (0,1): 'Sdd12 (FEXT)',
                  (1,0): 'Sdd21 (Insertion Loss)', (1,1): 'Sdd22 (Return Loss)'}
    for (i,j), name in diff_names.items():
        print(f"    {name}: {element_mae[(i,j)]:.2f} dB")
    
    print(f"\n  Physics: NOT ENFORCED (baseline test)")
    
    # ==========================================
    # Plots
    # ==========================================
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Learning curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['train_loss'], label='Train', color='blue', linewidth=2)
    ax.plot(history['val_loss'], label='Validation', color='orange', linewidth=2)
    ax.set_title(f'Freq-Conditioned MLP — {dataset_type.upper()}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'learning_curve_freq_{dataset_type}.png'), dpi=300)
    plt.close()
    
    # S-parameter plots for first 3 test samples
    num_plot = min(3, len(all_preds))
    for sample_idx in range(num_plot):
        S_pred_np = all_preds[sample_idx]
        S_true_np = all_trues[sample_idx]
        
        eps = 1e-10
        pred_db = 20 * np.log10(np.abs(S_pred_np) + eps)
        true_db = 20 * np.log10(np.abs(S_true_np) + eps)
        freqs = np.linspace(0.25, 100, S_pred_np.shape[0])
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Freq-Conditioned MLP ({dataset_type.upper()}) — Test Sample {sample_idx+1} — Diff MAE: {diff_mae:.2f} dB',
                     fontsize=16, fontweight='bold')
        
        titles = ['Sdd11 (Diff Return Loss)', 'Sdd12 (Diff FEXT)',
                  'Sdd21 (Diff Insertion Loss)', 'Sdd22 (Diff Return Loss)']
        indices = [(0,0), (0,1), (1,0), (1,1)]
        
        for idx, (r, c) in enumerate(indices):
            ax = axs[r, c]
            ax.plot(freqs, true_db[:, r, c], 'k-', linewidth=2, label='Ground Truth')
            ax.plot(freqs, pred_db[:, r, c], 'r--', linewidth=2, label='Freq-Cond MLP')
            ax.set_title(titles[idx], fontsize=12)
            ax.set_xlabel('Frequency (GHz)')
            ax.set_ylabel('Magnitude (dB)')
            ax.grid(True, ls="--", alpha=0.6)
            ax.set_ylim([-60, 5])
            if idx == 0:
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f's_params_freq_sample{sample_idx+1}_{dataset_type}.png'), dpi=300)
        plt.close()
    
    # Save summary
    summary = {
        'dataset': dataset_type,
        'best_epoch': best_epoch,
        'overall_mae': float(total_mae),
        'diff_mae': float(diff_mae),
        'per_element': {str(k): float(v) for k, v in element_mae.items()},
        'total_params': total_params,
        'train_pairs': len(train_pairs),
        'architecture': 'Freq-Conditioned MLP (geometry + frequency as input)',
    }
    with open(os.path.join(RESULTS_DIR, f'summary_freq_{dataset_type}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  Results saved to {RESULTS_DIR}/")
    return total_mae, diff_mae

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    results = {}
    for dtype in ['array', 'link']:
        overall, diff = run_freq_conditioned(dataset_type=dtype)
        results[dtype] = {'overall': overall, 'diff': diff}
    
    print(f"\n{'='*70}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"  {'Model':<30} {'Array Diff':<15} {'Link Diff':<15}")
    print(f"  {'─'*60}")
    print(f"  {'Standard MLP (4x4)':<30} {'5.47':<15} {'4.27':<15}")
    print(f"  {'Hybrid Rational (4x4)':<30} {'7.73':<15} {'9.45':<15}")
    
    arr = results.get('array', {})
    lnk = results.get('link', {})
    print(f"  {'Freq-Conditioned MLP':<30} {arr.get('diff', 'N/A'):<15.2f} {lnk.get('diff', 'N/A'):<15.2f}")
    
    print(f"\n  If Freq-Conditioned << Standard MLP → frequency conditioning is the key")
    print(f"  Apply same principle to the rational layer for physics-guaranteed version")