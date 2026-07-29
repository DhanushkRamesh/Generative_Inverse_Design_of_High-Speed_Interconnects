import sys
import os
import json
import torch
import torch.optim as optim
from tqdm import tqdm

"""
Training Pipeline for Hybrid Rational Net
============================================
S(s) = S_rational(s) + ΔS_MLP(s)

The rational backbone provides the causal-by-construction physics baseline.
The MLP correction learns the fine resonance structure the rational layer misses.

Key differences from train_fixedpole.py:
  1. Model returns S_total, S_rational, S_correction — loss is computed on S_total
  2. A correction penalty encourages the MLP correction to remain SMALL relative
     to the rational backbone, preserving the "approximately causal" guarantee
  3. Causality ratio is tracked throughout training and reported at evaluation

Usage:
    python train_hybridnet.py
    (Trains on both array and link datasets automatically)
"""

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJ_ROOT)

from src.data.dataset import get_dataloaders
from src.models.rational_hybrid_net import HybridRationalNet

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
NUM_POLES_HALF = 40     # Must match the shared poles file
HIDDEN_DIM = 512
BATCH_SIZE = 32
EPOCHS = 400
PATIENCE = 50

LR_BASE = 1e-3
WEIGHT_DECAY = 1e-3

LOSS_ALPHA = 1.0   # Weight on the complex MSE loss
LOSS_BETA  = 0.5   # Weight on the dB magnitude loss

# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------
def complex_mse_loss(S_pred, Y_true):
    """
    Calculates Mean Squared Error on real and imaginary parts equally.
    """
    diff = S_pred - Y_true
    return torch.mean(diff.real**2 + diff.imag**2)

def passivity_penalty(residues):    
    """
    Soft penalty for non-passive residues in the rational backbone.
    """    
    R_real = residues.real  
    R_sym = (R_real + R_real.transpose(-1, -2)) / 2.0
    eye = torch.eye(4, device=R_real.device).view(1, 1, 4, 4) * 1e-6
    R_reg = R_sym + eye
    eigvals = torch.linalg.eigvalsh(R_reg.view(-1, 4, 4))     
    negative_eigvals = torch.clamp(eigvals, max=0.0)    
    return torch.mean(torch.abs(negative_eigvals))

def correction_magnitude_penalty(S_rational, S_correction):
    """
    Penalises the MLP correction for being too large relative to the rational backbone.
    
    This is critical for maintaining the "approximately causal" guarantee.
    Without this penalty, the optimizer could route all learning through the
    unconstrained MLP correction and leave the rational backbone unused —
    defeating the entire purpose of the physics-informed architecture.
    
    The penalty is the ratio of correction energy to rational energy.
    A value of 0.0 means the correction is zero (pure rational).
    A value of 1.0 means correction and rational contribute equally.
    We want this to stay well below 0.5 — ideally under 0.2.
    """
    rational_energy = torch.mean(torch.abs(S_rational)**2)
    correction_energy = torch.mean(torch.abs(S_correction)**2)
    # Add epsilon to prevent division by zero early in training
    return correction_energy / (rational_energy + 1e-8)

def combined_loss(S_total, S_rational, S_correction, Y_true, poles, residues,
                  alpha=LOSS_ALPHA, beta=LOSS_BETA, epoch=0):
    """
    Combines S-parameter fitting losses with physics penalties.
    
    Loss components:
      1. Complex MSE on total prediction vs ground truth
      2. Frequency-weighted dB magnitude loss (emphasises resonances at high freq)
      3. Sobolev slope loss (encourages correct derivative at resonances)
      4. Soft passivity penalty on rational backbone residues
      5. Correction magnitude penalty (keeps MLP correction bounded)
    """
    # Primary fitting loss on the TOTAL prediction (rational + correction)
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

    # Soft passivity penalty on rational backbone
    p_loss = passivity_penalty(residues)

    # Correction magnitude penalty — keeps the MLP correction bounded
    # so the causal rational backbone remains the dominant contributor.
    # Weight starts at 0.5 and increases to 2.0 over 100 epochs.
    # This allows the correction to grow freely early in training
    # (when it needs to learn the residual) then tightens to prevent
    # the correction from dominating the response.
    corr_penalty = correction_magnitude_penalty(S_rational, S_correction)
    corr_weight = min(2.0, 0.5 + (1.5 * epoch / 100.0))

    slope_weight = 0.1 if epoch > 10 else 0.0
    passivity_weight = min(10.0, 1.0 + (9.0 * epoch / 50.0))

    return (alpha * mse) + (beta * final_db_loss) + (slope_weight * slope_loss) + \
           (passivity_weight * p_loss) + (corr_weight * corr_penalty)

# ---------------------------------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------------------------------
def train_hybridnet(dataset_type='array'):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"  Training HybridRationalNet on {device.type.upper()} ({dataset_type.upper()} Dataset)")
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

    # Instantiate HybridRationalNet
    model = HybridRationalNet(
        num_poles_half=NUM_POLES_HALF,
        num_local_features=num_local,
        num_global_features=num_global,
        shared_poles_path=poles_path,
        num_ports=4,
        hidden_dim=HIDDEN_DIM,
        num_freqs=num_freqs
    ).to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Architecture:         Rational backbone [FIXED poles] + MLP correction")
    print(f"  Num poles:            {NUM_POLES_HALF * 2} ({NUM_POLES_HALF} conjugate pairs) [FIXED]")
    print(f"  Hidden dim:           {HIDDEN_DIM}")
    print(f"  Num freqs:            {num_freqs}\n")

    # Optimiser — all trainable params together
    optimizer = optim.AdamW(model.parameters(), lr=LR_BASE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'causality_ratio': []}
    
    save_dir = os.path.join(PROJ_ROOT, "results/checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"best_hybrid_rational_net_{dataset_type}.pth")

    # ==========================================
    # Epoch Loop
    # ==========================================
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_train_loss = 0.0

        # Curriculum: expand frequency range over 150 epochs
        progress_ratio = min(1.0, epoch / 150.0)
        current_limit = 0.2 + (0.8 * progress_ratio)
        freq_limit = int(num_freqs * current_limit)
        
        for x_loc, x_glob, y_r, y_i in tqdm(train_loader, desc=f"Epoch {epoch:03d} [Train]", leave=False):
            x_loc, x_glob = x_loc.to(device), x_glob.to(device)
            Y_true = torch.complex(y_r, y_i).to(device)
            
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

        # Validation Loop
        model.eval()
        epoch_val_loss = 0.0
        epoch_causal_ratio = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for x_loc, x_glob, y_r, y_i in tqdm(val_loader, desc=f"Epoch {epoch:03d} [Val]", leave=False):
                x_loc, x_glob = x_loc.to(device), x_glob.to(device)
                Y_true = torch.complex(y_r, y_i).to(device)
                
                S_total, S_rational, S_correction, poles, residues = model(
                    x_loc, x_glob, frequencies_hz[:freq_limit]
                )
                
                loss = combined_loss(
                    S_total, S_rational, S_correction,
                    Y_true[:, :freq_limit], poles, residues, epoch=epoch
                )
                epoch_val_loss += loss.item() * x_loc.size(0)
                
                # Track causality ratio
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
                'num_freqs': num_freqs,
                'dataset_type': dataset_type,
                'shared_poles_path': poles_path,
                'architecture': 'hybrid_rational',
                'causality_ratio': avg_causal_ratio,
            }, best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  [INFO] Early stopping triggered at epoch {epoch}")
                break

    # Save Training History
    history_path = os.path.join(save_dir, f"training_history_hybrid_{dataset_type}.json")
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
    total_causal_ratio = 0.0
    all_poles = []
    all_residues = []
    test_batches = 0
    
    with torch.no_grad():
        for x_loc, x_glob, y_r, y_i in tqdm(test_loader, desc="Testing", leave=False):
            x_loc, x_glob = x_loc.to(device), x_glob.to(device)
            Y_true = torch.complex(y_r, y_i).to(device)
            
            S_total, S_rational, S_correction, poles, residues = model(
                x_loc, x_glob, frequencies_hz
            )
            
            loss = combined_loss(
                S_total, S_rational, S_correction,
                Y_true, poles, residues, epoch=EPOCHS
            )
            test_loss += loss.item() * x_loc.size(0)
            
            eps = 1e-4
            db_err = torch.abs((20*torch.log10(torch.abs(S_total)+eps)) - (20*torch.log10(torch.abs(Y_true)+eps)))
            phase_err = torch.abs(torch.angle(S_total) - torch.angle(Y_true)) * (180.0 / torch.pi)
            
            total_db_mae += torch.mean(db_err).item() * x_loc.size(0)
            total_phase_mae += torch.mean(phase_err).item() * x_loc.size(0)
            total_causal_ratio += model.compute_causality_ratio(S_rational, S_correction)
            test_batches += 1
            
            all_poles.append(poles)
            all_residues.append(residues)
            
    test_loss /= len(test_loader.dataset)
    avg_db_mae = total_db_mae / len(test_loader.dataset)
    avg_phase_mae = total_phase_mae / len(test_loader.dataset)
    avg_causal_ratio = total_causal_ratio / max(test_batches, 1)
    
    print(f"\n  Final Test Loss (Combined): {test_loss:.4e}")
    print(f"  Final Mean Absolute Error (dB): {avg_db_mae:.2f} dB")
    print(f"  Final Phase Error (Degrees): {avg_phase_mae:.2f}°")
    print(f"  Causality Ratio: {avg_causal_ratio:.1%} (rational backbone contribution)")
    
    all_poles = torch.cat(all_poles, dim=0)
    all_residues = torch.cat(all_residues, dim=0)
    physics_checks = model.verify_physics_constraints(all_poles, all_residues)
    
    print(f"\n  Physics Constraint Verification (rational backbone only):")
    print(f"    Causality Preserved: {'PASSED' if physics_checks['causality_preserved'] else 'FAILED'} (guaranteed by construction)")
    print(f"    Symmetry Preserved:  {'PASSED' if physics_checks['conjugate_symmetry_preserved'] else 'FAILED'}")
    print(f"    Passivity Preserved: {'PASSED' if physics_checks['passivity_preserved'] else 'FAILED'} (Min Eigenvalue: {physics_checks['min_residue_eigenvalue']:.2e})")

if __name__ == "__main__":
    for dtype in ['array', 'link']:
        train_hybridnet(dataset_type=dtype)