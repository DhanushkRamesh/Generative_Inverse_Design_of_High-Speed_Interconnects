import sys
import os
import json
import torch
import torch.optim as optim
from tqdm import tqdm

# Add the project root to the system path
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJ_ROOT)

from src.data.dataset import get_dataloaders
from src.models.rational_net import RationalNet

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
NUM_POLES = 80          # Increased from 40 → 80 to capture dense resonance structure.
                        # The TUHH datasets show many sharp dips across 0–100 GHz;
                        # each resonance needs at least one conjugate pole pair.
BATCH_SIZE = 32
EPOCHS = 400
PATIENCE = 50
HIDDEN_DIM = 512        # Width of the deeper residual backbone (v2)

# Optimizer Hyperparameters
LR_BASE = 1e-3
LR_POLES = 1e-3         # Poles move faster to find the right resonant frequencies early on, then slow down for fine-tuning
WEIGHT_DECAY = 1e-3

# Loss Function Hyperparameters
LOSS_ALPHA = 1.0        # Weight on the complex MSE loss
LOSS_BETA  = 0.5        # Weight on the dB magnitude loss

# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------
def complex_mse_loss(S_pred, Y_true):
    """
    Calculates Mean Squared Error on real and imaginary parts equally.
    """
    diff = S_pred - Y_true
    return torch.mean(diff.real**2 + diff.imag**2)

def db_magnitude_loss(S_pred, Y_true):
    """
    Calculates MSE on the dB magnitude of the S-parameters.
    Crucial for fitting deep resonant dips accurately.
    """
    # 1e-4 corresponds to a physical -80 dB noise floor, preventing 
    # the model from chasing -200 dB numerical simulator noise.
    eps = 1e-4
    mag_pred = torch.abs(S_pred) + eps
    mag_true = torch.abs(Y_true) + eps
    
    db_pred = 20 * torch.log10(mag_pred)
    db_true = 20 * torch.log10(mag_true)
    
    return torch.mean((db_pred - db_true)**2)

def passivity_penalty(residues):    
    """
    Soft penalty for non-passive residues (stops massive energy blowouts).
    
    This is the PRIMARY passivity enforcement mechanism during training.
    Unlike v1 where this competed with a hard SVD clamp that crushed gradients,
    this soft penalty now has full responsibility for guiding the model towards
    passive solutions. The weight has been reduced from 500.0 to 10.0 to prevent
    it from dominating the MSE and dB losses early in training.
    """    
    R_real = residues.real  
    # Enforce symmetry so eigvalsh doesn't crash during autograd
    R_sym = (R_real + R_real.transpose(-1, -2)) / 2.0
    # Add a small jitter to the diagonal for numerical stability, preventing
    # zero eigenvalues that can cause NaNs in gradients
    eye = torch.eye(4, device=R_real.device).view(1, 1, 4, 4) * 1e-6
    R_reg = R_sym + eye
    eigvals = torch.linalg.eigvalsh(R_reg.view(-1, 4, 4))     
    negative_eigvals = torch.clamp(eigvals, max=0.0)    
    return torch.mean(torch.abs(negative_eigvals))

def combined_loss(S_pred, Y_true, poles, residues, alpha=LOSS_ALPHA, beta=LOSS_BETA, epoch=0):
    """
    Combines Complex MSE with dB Magnitude loss and soft physics penalties.
    
    Key change from v1: passivity penalty weight reduced from 500.0 to 10.0.
    The hard SVD clamp no longer interferes during training, so the soft penalty
    doesn't need to be as aggressive. A weight of 10.0 allows the model to first
    learn the correct resonance structure, then gradually become passive as
    the penalty pulls residue eigenvalues towards positive semi-definiteness.
    """
    mse = complex_mse_loss(S_pred, Y_true)

    # Frequency weighting: more weight on higher frequencies where deep resonances occur
    num_freqs = S_pred.shape[1]
    freq_weights = torch.logspace(0.0, 1.0, steps=num_freqs).to(S_pred.device).view(1, -1, 1, 1)  # weights from 1.0 to 10.0 across the frequency spectrum

    # dB magnitude loss with an added frequency weighting
    epsilon = 1e-4
    db_pred = 20 * torch.log10(torch.abs(S_pred) + epsilon)
    db_true = 20 * torch.log10(torch.abs(Y_true) + epsilon)
    # The -70 dB Noise Floor Clamp
    db_mae = torch.abs(db_pred - torch.clamp(db_true, min=-70.0))
    final_db_loss = torch.mean(db_mae * freq_weights)

    # Sobolev Slope Loss - derivative loss to encourage correct slope at resonances, improving Q-factor fitting
    diff_pred = S_pred[:, 1:, :, :] - S_pred[:, :-1, :, :]
    diff_true = Y_true[:, 1:, :, :] - Y_true[:, :-1, :, :]
    slope_loss = torch.mean(torch.abs(diff_pred - diff_true))

    # Soft penalty for passivity violations
    p_loss = passivity_penalty(residues)

    # Loss balancing
    # Slope loss delayed to epoch 10 to let the model learn the basic shape first
    slope_weight = 0.1 if epoch > 10 else 0.0

    # Passivity penalty weight schedule:
    # Start very low (1.0) to let the model learn resonance structure freely,
    # then ramp up to 10.0 after epoch 50 to enforce physical constraints.
    # This curriculum prevents the passivity term from dominating early training
    # and collapsing the S-matrix to trivial near-identity solutions.
    passivity_weight = min(10.0, 1.0 + (9.0 * epoch / 50.0))

    # Final weighted loss combination
    return (alpha * mse) + (beta * final_db_loss) + (slope_weight * slope_loss) + (passivity_weight * p_loss)

# ---------------------------------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------------------------------
def train_model(dataset_type='link'):
    # Enforce reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"--- Training RationalNet v2 on {device.type.upper()} ({dataset_type.upper()} Dataset) ---")

    # Dynamic dataset folder routing
    dataset_folder = 'Universal-Diff-SI-Array' if dataset_type == 'array' else 'Universal-Diff-SI-Link'
    data_path = os.path.join(PROJ_ROOT, f"data/processed/{dataset_folder}/via_{dataset_type}_dataset.pt")
    
    train_loader, val_loader, test_loader = get_dataloaders(data_path, dataset_type=dataset_type, batch_size=BATCH_SIZE)
    
    # Extract dimensions and Frequency Axis
    x_local, x_global, y_real, y_imag = next(iter(train_loader))
    num_local = x_local.shape[1]
    num_global = x_global.shape[1]
    num_freqs = y_real.shape[1]
    
    # Matching TUHH standard 250 MHz to 100 GHz sweep
    frequencies_hz = torch.linspace(0.25e9, 100e9, num_freqs).to(device)

    # Instantiate RationalNet v2
    model = RationalNet(
        num_poles=NUM_POLES,
        num_local_features=num_local,
        num_global_features=num_global,
        num_ports=4,
        hidden_dim=HIDDEN_DIM
    ).to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Num poles:            {NUM_POLES} ({NUM_POLES//2} conjugate pairs)")
    print(f"Hidden dim:           {HIDDEN_DIM}")
    print(f"Residual blocks:      3\n")

    # Isolate weight decay to prevent fighting the softplus causality constraint on poles.
    # Poles and residues get zero weight decay — the softplus constraint on poles and the
    # soft passivity penalty on residues handle regularisation through physics, not L2.
    pole_params = []
    residue_params = []
    base_params = []
    for name, param in model.named_parameters():
        if 'poles' in name:
            pole_params.append(param)
        elif 'residues' in name or 'd_term' in name:
            residue_params.append(param)
        else:
            base_params.append(param)

    optimizer = optim.AdamW([
        {'params': base_params, 'lr': LR_BASE, 'weight_decay': WEIGHT_DECAY},
        {'params': pole_params, 'lr': LR_POLES, 'weight_decay': 0.0},
        {'params': residue_params, 'lr': LR_BASE, 'weight_decay': 0.0}
    ], lr=LR_BASE)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'lr_base': [], 'lr_pole': []}
    
    save_dir = os.path.join(PROJ_ROOT, "results/checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"best_rational_net_{dataset_type}.pth")

    # ==========================================
    # Epoch Loop
    # ==========================================
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_train_loss = 0.0

        # Curriculum Learning: gradually expand the frequency range.
        # Stretched from 50 epochs (v1) to 150 epochs (v2) to give the model
        # more time to properly learn the low-frequency behaviour (which is smoother
        # and easier) before introducing the high-frequency resonances.
        # Start at 20% of spectrum, smoothly grow to 100%.
        progress_ratio = min(1.0, epoch / 150.0)
        current_limit = 0.2 + (0.8 * progress_ratio)
        freq_limit = int(num_freqs * current_limit)
        
        for x_loc, x_glob, y_r, y_i in tqdm(train_loader, desc=f"Epoch {epoch:03d} [Train]", leave=False):
            x_loc, x_glob = x_loc.to(device), x_glob.to(device)
            Y_true = torch.complex(y_r, y_i).to(device)
            
            optimizer.zero_grad()
            
            poles, residues, d_term = model(x_loc, x_glob)
            S_pred = model.predict_frequency_response(poles, residues, d_term, frequencies_hz[:freq_limit])
            # The combined loss function now also takes poles and residues as input for the passivity penalty
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
                Y_true = torch.complex(y_r, y_i).to(device)
                
                poles, residues, d_term = model(x_loc, x_glob)
                S_pred = model.predict_frequency_response(poles, residues, d_term, frequencies_hz[:freq_limit])
                
                loss = combined_loss(S_pred, Y_true[:, :freq_limit], poles, residues, epoch=epoch)
                epoch_val_loss += loss.item() * x_loc.size(0)
                
        epoch_val_loss /= len(val_loader.dataset)
        
        scheduler.step(epoch_val_loss)
        
        current_lr_base = optimizer.param_groups[0]['lr']
        current_lr_pole = optimizer.param_groups[1]['lr']
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['lr_base'].append(current_lr_base)
        history['lr_pole'].append(current_lr_pole)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}/{EPOCHS}] | Train Loss: {epoch_train_loss:.4e} | Val Loss: {epoch_val_loss:.4e} | LR_base: {current_lr_base:.2e} | LR_pole: {current_lr_pole:.2e} | Freq: {current_limit*100:.0f}%")

        # Save Checkpoint & Early Stopping Check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch':          epoch,
                'model_state':    model.state_dict(),
                'optimizer_state':optimizer.state_dict(),
                'val_loss':       best_val_loss,
                'num_poles':      NUM_POLES,
                'num_local':      num_local,
                'num_global':     num_global,
                'hidden_dim':     HIDDEN_DIM,
                'dataset_type':   dataset_type,
            }, best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"[INFO] Early stopping triggered at epoch {epoch}")
                break

    # Save Training History to JSON
    history_path = os.path.join(save_dir, f"training_history_{dataset_type}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f"\n[INFO] Training history saved to {history_path}")

    # ==========================================
    # Final Verification & Testing
    # ==========================================
    print("\n--- Running Final Evaluation on Test Set ---")
    
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} with val loss {checkpoint['val_loss']:.4e}")
    
    test_loss = 0.0
    total_db_mae = 0.0
    total_phase_mae = 0.0
    all_poles = []
    all_residues = []
    
    with torch.no_grad():
        for x_loc, x_glob, y_r, y_i in tqdm(test_loader, desc="Testing", leave=False):
            x_loc, x_glob = x_loc.to(device), x_glob.to(device)
            Y_true = torch.complex(y_r, y_i).to(device)
            
            poles, residues, d_term = model(x_loc, x_glob)
            S_pred = model.predict_frequency_response(poles, residues, d_term, frequencies_hz)
            
            loss = combined_loss(S_pred, Y_true, poles, residues, epoch=EPOCHS)
            test_loss += loss.item() * x_loc.size(0)
            # Metric calculations for S-parameter accuracy in dB and phase, averaged across all ports and frequencies
            eps = 1e-4
            db_err = torch.abs((20*torch.log10(torch.abs(S_pred)+eps)) - (20*torch.log10(torch.abs(Y_true)+eps)))
            phase_err = torch.abs(torch.angle(S_pred) - torch.angle(Y_true)) * (180.0 / torch.pi)  # Convert rad to deg
            # Accumulate weighted errors for the entire test set
            total_db_mae += torch.mean(db_err).item() * x_loc.size(0)
            total_phase_mae += torch.mean(phase_err).item() * x_loc.size(0)
            
            all_poles.append(poles)
            all_residues.append(residues)
            
    test_loss /= len(test_loader.dataset)
    # Average the accumulated metrics across the entire test set
    avg_db_mae = total_db_mae / len(test_loader.dataset)
    avg_phase_mae = total_phase_mae / len(test_loader.dataset)
    
    # Print the final metrics
    print(f"\nFinal Test Loss (Combined): {test_loss:.4e}")
    print(f"Final Mean Absolute Error (dB): {avg_db_mae:.2f} dB")
    print(f"Final Phase Error (Degrees): {avg_phase_mae:.2f}°")
    
    all_poles = torch.cat(all_poles, dim=0)
    all_residues = torch.cat(all_residues, dim=0)
    physics_checks = model.verify_physics_constraints(all_poles, all_residues)
        
    print(f"\nPhysics Constraint Verification:")
    print(f"  Causality Preserved: {'PASSED' if physics_checks['causality_preserved'] else 'FAILED'}")
    print(f"  Symmetry Preserved:  {'PASSED' if physics_checks['conjugate_symmetry_preserved'] else 'FAILED'}")
    print(f"  Passivity Preserved: {'PASSED' if physics_checks['passivity_preserved'] else 'FAILED'} (Min Eigenvalue: {physics_checks['min_residue_eigenvalue']:.2e})")

if __name__ == "__main__":
    # To run on the Array dataset later, swap 'link' for 'array'
    train_model(dataset_type='link')