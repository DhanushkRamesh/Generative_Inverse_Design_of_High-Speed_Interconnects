import sys
import os
import json
import torch
import torch.optim as optim
from tqdm import tqdm

"""
Training Pipeline for VF-Informed Fixed-Pole RationalNet
==========================================================
This script trains the fixed-pole version of RationalNet where:
  - Poles are PRE-EXTRACTED from classical Vector Fitting on representative samples
    and loaded as non-trainable buffers (see extract_shared_poles.py)
  - The neural network only predicts residues (coupling strengths) and D term
  - Causality is guaranteed by construction — no soft penalty needed for poles

The training procedure is nearly identical to train_rationalnet.py, with two
key differences:
  1. No pole parameters in the optimiser (they're fixed buffers)
  2. Faster convergence expected because the network solves an easier problem
     (smooth residue regression vs non-convex joint pole-residue optimisation)

Usage:
    python train_fixedpole.py
    (Trains on both array and link datasets automatically)
"""

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJ_ROOT)

from src.data.dataset import get_dataloaders
from src.models.rational_net import RationalNet

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
NUM_POLES_HALF = 40     # Must match the shared poles file (40 pairs → 80 total poles)
HIDDEN_DIM = 512
NUM_PORTS = 2
BATCH_SIZE = 32
EPOCHS = 400
PATIENCE = 50

# Optimizer Hyperparameters — no pole LR needed since poles are fixed
LR_BASE = 1e-3
WEIGHT_DECAY = 1e-3

# Loss Function Hyperparameters
LOSS_ALPHA = 1.0   # Weight on the complex MSE loss
LOSS_BETA  = 0.5   # Weight on the dB magnitude loss

# ---------------------------------------------------------------------------
# Loss Functions (identical to train_rationalnet.py)
# ---------------------------------------------------------------------------
def complex_mse_loss(S_pred, Y_true):
    """
    Calculates Mean Squared Error on real and imaginary parts equally.
    """
    diff = S_pred - Y_true
    return torch.mean(diff.real**2 + diff.imag**2)

def passivity_penalty(residues):    
    """
    Soft penalty for non-passive residues (stops massive energy blowouts).
    With fixed poles, this is the only physics penalty needed during training.
    Causality is handled architecturally (fixed left half-plane poles).
    """    
    R_real = residues.real  
    R_sym = (R_real + R_real.transpose(-1, -2)) / 2.0
    eye = torch.eye(NUM_PORTS, device=R_real.device).view(1, 1, NUM_PORTS, NUM_PORTS) * 1e-6
    R_reg = R_sym + eye
    eigvals = torch.linalg.eigvalsh(R_reg.view(-1, 4, 4))     
    negative_eigvals = torch.clamp(eigvals, max=0.0)    
    return torch.mean(torch.abs(negative_eigvals))

def combined_loss(S_pred, Y_true, poles, residues, alpha=LOSS_ALPHA, beta=LOSS_BETA, epoch=0):
    """
    Combines Complex MSE with dB Magnitude loss and soft physics penalties.
    Same structure as train_rationalnet.py for fair comparison.
    """
    mse = complex_mse_loss(S_pred, Y_true)

    # Frequency weighting: more weight on higher frequencies where deep resonances occur
    num_freqs = S_pred.shape[1]
    freq_weights = torch.logspace(0.0, 1.0, steps=num_freqs).to(S_pred.device).view(1, -1, 1, 1)

    # dB magnitude loss with frequency weighting
    epsilon = 1e-4
    db_pred = 20 * torch.log10(torch.abs(S_pred) + epsilon)
    db_true = 20 * torch.log10(torch.abs(Y_true) + epsilon)
    db_mae = torch.abs(db_pred - torch.clamp(db_true, min=-70.0))
    final_db_loss = torch.mean(db_mae * freq_weights)

    # Sobolev Slope Loss - derivative loss to encourage correct slope at resonances
    diff_pred = S_pred[:, 1:, :, :] - S_pred[:, :-1, :, :]
    diff_true = Y_true[:, 1:, :, :] - Y_true[:, :-1, :, :]
    slope_loss = torch.mean(torch.abs(diff_pred - diff_true))

    # Soft passivity penalty with warmup schedule
    p_loss = passivity_penalty(residues)

    slope_weight = 0.1 if epoch > 10 else 0.0
    passivity_weight = min(10.0, 1.0 + (9.0 * epoch / 50.0))

    return (alpha * mse) + (beta * final_db_loss) + (slope_weight * slope_loss) + (passivity_weight * p_loss)

# ---------------------------------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------------------------------
def train_fixedpole(dataset_type='array'):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"  Training Fixed-Pole RationalNet on {device.type.upper()} ({dataset_type.upper()} Dataset)")
    print(f"{'='*70}")

    # Check that shared poles file exists
    dataset_folder = 'Universal-Diff-SI-Array' if dataset_type == 'array' else 'Universal-Diff-SI-Link'
    poles_path = os.path.join(PROJ_ROOT, f"data/processed/{dataset_folder}/shared_poles_{dataset_type}.pt")
    
    if not os.path.exists(poles_path):
        print(f"[ERROR] Shared poles not found at {poles_path}")
        print(f"        Run extract_shared_poles.py first.")
        return

    # Load data
    data_path = os.path.join(PROJ_ROOT, f"data/processed/{dataset_folder}/via_{dataset_type}_dataset.pt")
    train_loader, val_loader, test_loader = get_dataloaders(data_path, dataset_type=dataset_type, batch_size=BATCH_SIZE)
    
    # Extract dimensions and frequency axis
    x_local, x_global, y_real, y_imag = next(iter(train_loader))
    num_local = x_local.shape[1]
    num_global = x_global.shape[1]
    num_freqs = y_real.shape[1]
    frequencies_hz = torch.linspace(0.25e9, 100e9, num_freqs).to(device)

    # Instantiate Fixed-Pole RationalNet
    model = RationalNet(
        num_poles_half=NUM_POLES_HALF,
        num_local_features=num_local,
        num_global_features=num_global,
        shared_poles_path=poles_path,
        num_ports=NUM_PORTS,
        hidden_dim=HIDDEN_DIM
    ).to(device)

    # Print model summary — note: fewer trainable params than v2 (no pole heads)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fixed_params = total_params - trainable_params
    print(f"\n  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} (residues + D + backbone)")
    print(f"  Fixed parameters:     {fixed_params:,} (poles + Fourier projection)")
    print(f"  Num poles:            {NUM_POLES_HALF * 2} ({NUM_POLES_HALF} conjugate pairs) [FIXED]")
    print(f"  Hidden dim:           {HIDDEN_DIM}")
    print(f"  Residual blocks:      3\n")

    # Simpler optimiser — no pole parameter group needed
    # All trainable params get the same LR since there's no pole-vs-residue tension
    residue_params = [p for n, p in model.named_parameters() if 'residues' in n or 'd_term' in n]
    base_params = [p for n, p in model.named_parameters() if 'residues' not in n and 'd_term' not in n]

    optimizer = optim.AdamW([
        {'params': base_params, 'lr': LR_BASE, 'weight_decay': WEIGHT_DECAY},
        {'params': residue_params, 'lr': LR_BASE, 'weight_decay': 0.0}
    ], lr=LR_BASE)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    save_dir = os.path.join(PROJ_ROOT, "results/checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"best_rational_net_fixedpole_{dataset_type}.pth")

    # ==========================================
    # Epoch Loop
    # ==========================================
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_train_loss = 0.0

        # Curriculum Learning: gradually expand the frequency range
        # Slower curriculum (150 epochs) to let the model learn the smooth
        # low-frequency residue structure before tackling high-freq resonances
        progress_ratio = min(1.0, epoch / 150.0)
        current_limit = 0.2 + (0.8 * progress_ratio)
        freq_limit = int(num_freqs * current_limit)
        
        for x_loc, x_glob, y_r, y_i in tqdm(train_loader, desc=f"Epoch {epoch:03d} [Train]", leave=False):
            x_loc, x_glob = x_loc.to(device), x_glob.to(device)
            Y_true = torch.complex(y_r[:, :, :NUM_PORTS, :NUM_PORTS], y_i[:, :, :NUM_PORTS, :NUM_PORTS]).to(device)
            
            optimizer.zero_grad()
            
            poles, residues, d_term = model(x_loc, x_glob)
            S_pred = model.predict_frequency_response(poles, residues, d_term, frequencies_hz[:freq_limit])
            loss = combined_loss(S_pred, Y_true[:, :freq_limit], poles, residues, epoch=epoch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item() * x_loc.size(0)
            
        epoch_train_loss /= len(train_loader.dataset)

        # Validation Loop
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x_loc, x_glob, y_r, y_i in tqdm(val_loader, desc=f"Epoch {epoch:03d} [Val]", leave=False):
                x_loc, x_glob = x_loc.to(device), x_glob.to(device)
                Y_true = torch.complex(y_r[:, :, :NUM_PORTS, :NUM_PORTS], y_i[:, :, :NUM_PORTS, :NUM_PORTS]).to(device)
                
                poles, residues, d_term = model(x_loc, x_glob)
                S_pred = model.predict_frequency_response(poles, residues, d_term, frequencies_hz[:freq_limit])
                loss = combined_loss(S_pred, Y_true[:, :freq_limit], poles, residues, epoch=epoch)
                epoch_val_loss += loss.item() * x_loc.size(0)
                
        epoch_val_loss /= len(val_loader.dataset)
        
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['lr'].append(current_lr)

        if epoch % 5 == 0 or epoch == 1:
            freq_pct = current_limit * 100
            print(f"  [{epoch:03d}/{EPOCHS}] | Train: {epoch_train_loss:.4e} | Val: {epoch_val_loss:.4e} | LR: {current_lr:.2e} | Freq: {freq_pct:.0f}%")

        # Save Checkpoint & Early Stopping
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
                'dataset_type': dataset_type,
                'shared_poles_path': poles_path,
                'architecture': 'fixed_pole',
            }, best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  [INFO] Early stopping triggered at epoch {epoch}")
                break

    # Save Training History
    history_path = os.path.join(save_dir, f"training_history_fixedpole_{dataset_type}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f"\n  [INFO] Training history saved to {history_path}")

    # ==========================================
    # Final Evaluation on Test Set
    # ==========================================
    print(f"\n{'='*70}")
    print(f"  Final Evaluation on Test Set")
    print(f"{'='*70}")
    
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print(f"  Loaded best checkpoint from epoch {checkpoint['epoch']} (val loss: {checkpoint['val_loss']:.4e})")
    
    test_loss = 0.0
    total_db_mae = 0.0
    total_phase_mae = 0.0
    all_poles = []
    all_residues = []
    
    with torch.no_grad():
        for x_loc, x_glob, y_r, y_i in tqdm(test_loader, desc="Testing", leave=False):
            x_loc, x_glob = x_loc.to(device), x_glob.to(device)
            Y_true = torch.complex(y_r[:, :, :NUM_PORTS, :NUM_PORTS], y_i[:, :, :NUM_PORTS, :NUM_PORTS]).to(device)
            
            poles, residues, d_term = model(x_loc, x_glob)
            S_pred = model.predict_frequency_response(poles, residues, d_term, frequencies_hz)
            
            loss = combined_loss(S_pred, Y_true, poles, residues, epoch=EPOCHS)
            test_loss += loss.item() * x_loc.size(0)
            
            eps = 1e-4
            db_err = torch.abs((20*torch.log10(torch.abs(S_pred)+eps)) - (20*torch.log10(torch.abs(Y_true)+eps)))
            phase_err = torch.abs(torch.angle(S_pred) - torch.angle(Y_true)) * (180.0 / torch.pi)
            
            total_db_mae += torch.mean(db_err).item() * x_loc.size(0)
            total_phase_mae += torch.mean(phase_err).item() * x_loc.size(0)
            
            all_poles.append(poles)
            all_residues.append(residues)
            
    test_loss /= len(test_loader.dataset)
    avg_db_mae = total_db_mae / len(test_loader.dataset)
    avg_phase_mae = total_phase_mae / len(test_loader.dataset)
    
    print(f"\n  Final Test Loss (Combined): {test_loss:.4e}")
    print(f"  Final Mean Absolute Error (dB): {avg_db_mae:.2f} dB")
    print(f"  Final Phase Error (Degrees): {avg_phase_mae:.2f}°")
    
    all_poles = torch.cat(all_poles, dim=0)
    all_residues = torch.cat(all_residues, dim=0)
    physics_checks = model.verify_physics_constraints(all_poles, all_residues)
    
    print(f"\n  Physics Constraint Verification:")
    print(f"    Causality Preserved: {'PASSED' if physics_checks['causality_preserved'] else 'FAILED'} (guaranteed by construction)")
    print(f"    Symmetry Preserved:  {'PASSED' if physics_checks['conjugate_symmetry_preserved'] else 'FAILED'}")
    print(f"    Passivity Preserved: {'PASSED' if physics_checks['passivity_preserved'] else 'FAILED'} (Min Eigenvalue: {physics_checks['min_residue_eigenvalue']:.2e})")

if __name__ == "__main__":
    for dtype in ['array', 'link']:
        train_fixedpole(dataset_type=dtype)