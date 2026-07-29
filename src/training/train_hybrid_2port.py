import sys
import os
import json
import torch
import torch.optim as optim
from tqdm import tqdm

"""
Training Pipeline for 2-Port Differential Hybrid RationalNet
================================================================
This script trains the HybridRationalNet on ONLY the 2x2 differential
sub-matrix (Sdd11, Sdd12, Sdd21, Sdd22) instead of the full 4x4 mixed-mode
matrix that includes common-mode and cross-mode coupling terms.

Why 2-port instead of 4-port:
  - The full 4x4 mixed-mode matrix contains Sdd (differential), Scc (common-mode),
    and Sdc/Scd (cross-mode coupling). The cross-mode terms are typically 20-40 dB
    below the differential terms and are driven by subtle structural asymmetries
    that are extremely hard to predict from geometry features alone.
  - These cross-mode terms were inflating the MAE by contributing disproportionate
    error to the average (a 5 dB error at -35 dB counts the same as 5 dB at -5 dB).
  - Konduru et al. [4], Sreekumar et al. [5], and Akinwande et al. [6] all focused
    on the dominant differential parameters for their forward surrogates.
  - For signal integrity design specs, engineers specify requirements on Sdd11
    (return loss) and Sdd21 (insertion loss) — not on cross-mode coupling.
  - With 2 ports instead of 4, the rational function now has 10 poles per
    S-parameter element (40 pairs / 4 elements) instead of 2.5 (40 pairs / 16).

Data handling:
  - The existing 4x4 dataset is loaded normally via get_dataloaders
  - Y_real and Y_imag are sliced to [:, :, :2, :2] to extract the differential quad
  - No re-processing of the dataset or re-extraction of poles is needed

Usage:
    python train_hybrid_2port.py
    (Trains on both array and link datasets automatically)
"""

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJ_ROOT)

from src.data.dataset import get_dataloaders
from src.models.rational_hybrid_net import HybridRationalNet

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
NUM_POLES_HALF = 40     # Same poles as 4-port — poles are frequency locations, port-independent
HIDDEN_DIM = 512
NUM_PORTS = 2           # 2-port differential sub-matrix
BATCH_SIZE = 32
EPOCHS = 400
PATIENCE = 50

LR_BASE = 1e-3
WEIGHT_DECAY = 1e-3

LOSS_ALPHA = 1.0
LOSS_BETA  = 0.5

# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------
def complex_mse_loss(S_pred, Y_true):
    """Calculates Mean Squared Error on real and imaginary parts equally."""
    diff = S_pred - Y_true
    return torch.mean(diff.real**2 + diff.imag**2)

def passivity_penalty(residues):    
    """Soft penalty for non-passive residues in the rational backbone."""    
    R_real = residues.real  
    R_sym = (R_real + R_real.transpose(-1, -2)) / 2.0
    eye = torch.eye(NUM_PORTS, device=R_real.device).view(1, 1, NUM_PORTS, NUM_PORTS) * 1e-6
    R_reg = R_sym + eye
    eigvals = torch.linalg.eigvalsh(R_reg.view(-1, NUM_PORTS, NUM_PORTS))     
    negative_eigvals = torch.clamp(eigvals, max=0.0)    
    return torch.mean(torch.abs(negative_eigvals))

def correction_magnitude_penalty(S_rational, S_correction):
    """Penalises the MLP correction for being too large relative to the rational backbone."""
    rational_energy = torch.mean(torch.abs(S_rational)**2)
    correction_energy = torch.mean(torch.abs(S_correction)**2)
    return correction_energy / (rational_energy + 1e-8)

def combined_loss(S_total, S_rational, S_correction, Y_true, poles, residues, epoch=0):
    """Combines S-parameter fitting losses with physics penalties."""
    mse = complex_mse_loss(S_total, Y_true)

    # Frequency weighting
    num_freqs = S_total.shape[1]
    freq_weights = torch.logspace(0.0, 1.0, steps=num_freqs).to(S_total.device).view(1, -1, 1, 1)

    # dB magnitude loss
    epsilon = 1e-4
    db_pred = 20 * torch.log10(torch.abs(S_total) + epsilon)
    db_true = 20 * torch.log10(torch.abs(Y_true) + epsilon)
    db_mae = torch.abs(db_pred - torch.clamp(db_true, min=-70.0))
    final_db_loss = torch.mean(db_mae * freq_weights)

    # Sobolev slope loss
    diff_pred = S_total[:, 1:, :, :] - S_total[:, :-1, :, :]
    diff_true = Y_true[:, 1:, :, :] - Y_true[:, :-1, :, :]
    slope_loss = torch.mean(torch.abs(diff_pred - diff_true))

    # Soft passivity penalty
    p_loss = passivity_penalty(residues)

    # Correction penalty — reduced weight to allow the MLP to actually contribute
    corr_penalty = correction_magnitude_penalty(S_rational, S_correction)
    corr_weight = min(0.3, 0.05 + (0.25 * epoch / 100.0))

    slope_weight = 0.1 if epoch > 10 else 0.0
    passivity_weight = min(10.0, 1.0 + (9.0 * epoch / 50.0))

    return (LOSS_ALPHA * mse) + (LOSS_BETA * final_db_loss) + (slope_weight * slope_loss) + \
           (passivity_weight * p_loss) + (corr_weight * corr_penalty)

# ---------------------------------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------------------------------
def train_hybrid_2port(dataset_type='array'):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"  Training 2-Port Hybrid RationalNet on {device.type.upper()} ({dataset_type.upper()})")
    print(f"{'='*70}")

    # Check shared poles
    dataset_folder = 'Universal-Diff-SI-Array' if dataset_type == 'array' else 'Universal-Diff-SI-Link'
    poles_path = os.path.join(PROJ_ROOT, f"data/processed/{dataset_folder}/shared_poles_{dataset_type}.pt")
    
    if not os.path.exists(poles_path):
        print(f"[ERROR] Shared poles not found at {poles_path}")
        return

    # Load data (full 4-port, we slice to 2-port in the training loop)
    data_path = os.path.join(PROJ_ROOT, f"data/processed/{dataset_folder}/via_{dataset_type}_dataset.pt")
    train_loader, val_loader, test_loader = get_dataloaders(data_path, dataset_type=dataset_type, batch_size=BATCH_SIZE)
    
    # Extract dimensions
    x_local, x_global, y_real, y_imag = next(iter(train_loader))
    num_local = x_local.shape[1]
    num_global = x_global.shape[1]
    num_freqs = y_real.shape[1]
    frequencies_hz = torch.linspace(0.25e9, 100e9, num_freqs).to(device)

    # Instantiate model with num_ports=2
    model = HybridRationalNet(
        num_poles_half=NUM_POLES_HALF,
        num_local_features=num_local,
        num_global_features=num_global,
        shared_poles_path=poles_path,
        num_ports=NUM_PORTS,
        hidden_dim=HIDDEN_DIM,
        num_freqs=num_freqs
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Architecture:         2-Port Hybrid (Sdd only)")
    print(f"  Num poles:            {NUM_POLES_HALF * 2} ({NUM_POLES_HALF} pairs) [FIXED]")
    print(f"  Ports:                {NUM_PORTS} (differential sub-matrix)")
    print(f"  Poles per element:    {NUM_POLES_HALF * 2 / (NUM_PORTS * NUM_PORTS):.0f}\n")

    optimizer = optim.AdamW(model.parameters(), lr=LR_BASE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'causality_ratio': []}
    
    save_dir = os.path.join(PROJ_ROOT, "results/checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"best_hybrid_2port_{dataset_type}.pth")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_train_loss = 0.0

        progress_ratio = min(1.0, epoch / 150.0)
        current_limit = 0.2 + (0.8 * progress_ratio)
        freq_limit = int(num_freqs * current_limit)
        
        for x_loc, x_glob, y_r, y_i in tqdm(train_loader, desc=f"Epoch {epoch:03d} [Train]", leave=False):
            x_loc, x_glob = x_loc.to(device), x_glob.to(device)
            # SLICE to 2x2 differential sub-matrix [:, :, :2, :2]
            Y_true = torch.complex(y_r[:, :, :2, :2], y_i[:, :, :2, :2]).to(device)
            
            optimizer.zero_grad()
            
            S_total, S_rational, S_correction, poles, residues = model(
                x_loc, x_glob, frequencies_hz[:freq_limit]
            )
            
            loss = combined_loss(
                S_total, S_rational, S_correction,
                Y_true[:, :freq_limit], poles, residues, epoch=epoch
            )
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item() * x_loc.size(0)
            
        epoch_train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_causal_ratio = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for x_loc, x_glob, y_r, y_i in tqdm(val_loader, desc=f"Epoch {epoch:03d} [Val]", leave=False):
                x_loc, x_glob = x_loc.to(device), x_glob.to(device)
                Y_true = torch.complex(y_r[:, :, :2, :2], y_i[:, :, :2, :2]).to(device)
                
                S_total, S_rational, S_correction, poles, residues = model(
                    x_loc, x_glob, frequencies_hz[:freq_limit]
                )
                
                loss = combined_loss(
                    S_total, S_rational, S_correction,
                    Y_true[:, :freq_limit], poles, residues, epoch=epoch
                )
                epoch_val_loss += loss.item() * x_loc.size(0)
                epoch_causal_ratio += model.compute_causality_ratio(S_rational, S_correction)
                val_batches += 1
                
        epoch_val_loss /= len(val_loader.dataset)
        avg_causal_ratio = epoch_causal_ratio / max(val_batches, 1)
        
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['lr'].append(current_lr)
        history['causality_ratio'].append(avg_causal_ratio)

        if epoch % 5 == 0 or epoch == 1:
            freq_pct = current_limit * 100
            print(f"  [{epoch:03d}/{EPOCHS}] | Train: {epoch_train_loss:.4e} | Val: {epoch_val_loss:.4e} | LR: {current_lr:.2e} | Freq: {freq_pct:.0f}% | Causal: {avg_causal_ratio:.1%}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'num_poles_half': NUM_POLES_HALF,
                'hidden_dim': HIDDEN_DIM,
                'num_local': num_local,
                'num_global': num_global,
                'num_freqs': num_freqs,
                'num_ports': NUM_PORTS,
                'dataset_type': dataset_type,
                'shared_poles_path': poles_path,
                'architecture': 'hybrid_2port',
                'causality_ratio': avg_causal_ratio,
            }, best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  [INFO] Early stopping triggered at epoch {epoch}")
                break

    # Save history
    history_path = os.path.join(save_dir, f"training_history_hybrid_2port_{dataset_type}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f"\n  [INFO] History saved to {history_path}")

    # ==========================================
    # Final Evaluation
    # ==========================================
    print(f"\n{'='*70}")
    print(f"  Final Evaluation on Test Set (2-Port Differential)")
    print(f"{'='*70}")
    
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print(f"  Loaded epoch {checkpoint['epoch']} (val loss: {checkpoint['val_loss']:.4e})")
    
    test_loss = 0.0
    total_db_mae = 0.0
    total_phase_mae = 0.0
    total_causal_ratio = 0.0
    # Per-element MAE tracking for detailed analysis
    element_db_mae = {(i,j): 0.0 for i in range(NUM_PORTS) for j in range(NUM_PORTS)}
    all_poles = []
    all_residues = []
    test_batches = 0
    
    with torch.no_grad():
        for x_loc, x_glob, y_r, y_i in tqdm(test_loader, desc="Testing", leave=False):
            x_loc, x_glob = x_loc.to(device), x_glob.to(device)
            Y_true = torch.complex(y_r[:, :, :2, :2], y_i[:, :, :2, :2]).to(device)
            
            S_total, S_rational, S_correction, poles, residues = model(
                x_loc, x_glob, frequencies_hz
            )
            
            loss = combined_loss(
                S_total, S_rational, S_correction,
                Y_true, poles, residues, epoch=EPOCHS
            )
            test_loss += loss.item() * x_loc.size(0)
            
            eps = 1e-4
            db_pred = 20*torch.log10(torch.abs(S_total)+eps)
            db_true = 20*torch.log10(torch.abs(Y_true)+eps)
            db_err = torch.abs(db_pred - db_true)
            phase_err = torch.abs(torch.angle(S_total) - torch.angle(Y_true)) * (180.0 / torch.pi)
            
            total_db_mae += torch.mean(db_err).item() * x_loc.size(0)
            total_phase_mae += torch.mean(phase_err).item() * x_loc.size(0)
            
            # Per-element MAE
            for i in range(NUM_PORTS):
                for j in range(NUM_PORTS):
                    element_db_mae[(i,j)] += torch.mean(db_err[:, :, i, j]).item() * x_loc.size(0)
            
            total_causal_ratio += model.compute_causality_ratio(S_rational, S_correction)
            test_batches += 1
            
            all_poles.append(poles)
            all_residues.append(residues)
            
    n_test = len(test_loader.dataset)
    test_loss /= n_test
    avg_db_mae = total_db_mae / n_test
    avg_phase_mae = total_phase_mae / n_test
    avg_causal_ratio = total_causal_ratio / max(test_batches, 1)
    
    print(f"\n  Overall Test Loss: {test_loss:.4e}")
    print(f"  Overall dB MAE:   {avg_db_mae:.2f} dB")
    print(f"  Overall Phase:    {avg_phase_mae:.2f}°")
    print(f"  Causality Ratio:  {avg_causal_ratio:.1%}")
    
    # Per-element breakdown — this is the key metric
    print(f"\n  Per-Element dB MAE:")
    element_names = {(0,0): 'Sdd11 (Return Loss)', (0,1): 'Sdd12',
                     (1,0): 'Sdd21 (Insertion Loss)', (1,1): 'Sdd22 (Return Loss)'}
    for (i,j), name in element_names.items():
        mae = element_db_mae[(i,j)] / n_test
        print(f"    {name}: {mae:.2f} dB")
    
    all_poles = torch.cat(all_poles, dim=0)
    all_residues = torch.cat(all_residues, dim=0)
    physics_checks = model.verify_physics_constraints(all_poles, all_residues)
    
    print(f"\n  Physics Constraints (rational backbone):")
    print(f"    Causality:  {'PASSED' if physics_checks['causality_preserved'] else 'FAILED'} (by construction)")
    print(f"    Symmetry:   {'PASSED' if physics_checks['conjugate_symmetry_preserved'] else 'FAILED'}")
    print(f"    Passivity:  {'PASSED' if physics_checks['passivity_preserved'] else 'FAILED'} (Min Eig: {physics_checks['min_residue_eigenvalue']:.2e})")

if __name__ == "__main__":
    for dtype in ['array', 'link']:
        train_hybrid_2port(dataset_type=dtype)