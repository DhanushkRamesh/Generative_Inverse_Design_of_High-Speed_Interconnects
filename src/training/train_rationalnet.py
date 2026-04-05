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
NUM_POLES = 128
BATCH_SIZE = 32
EPOCHS = 200
PATIENCE = 20

# Optimizer Hyperparameters
LR_BASE = 1e-3
LR_POLES = 1e-4  
WEIGHT_DECAY = 1e-4

# Loss Function Hyperparameters
LOSS_ALPHA = 0.1   # Weight on the complex MSE loss
LOSS_BETA  = 5.0  # Weight on the dB magnitude loss

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

def combined_loss(S_pred, Y_true, poles, residues, alpha=LOSS_ALPHA, beta=LOSS_BETA):
    """
    Combines Complex MSE with dB Magnitude loss.
    """
    mse = complex_mse_loss(S_pred, Y_true)
    #dB magnitude loss with an added frequency weighting
    epsilon = 1e-4
    db_pred = 20 * torch.log10(torch.abs(S_pred) + epsilon)
    db_true = 20 * torch.log10(torch.abs(Y_true) + epsilon)
    db_mse = (db_pred - db_true)**2 #element-wise MSE in dB
    #We multiply the error of the diagonal elements (S11, S22) by 5.0
    #To force the optimizer to care about reflections as much as transmission.
    db_mse[:, :, 0, 0] *= 15.0  # S11
    db_mse[:, :, 1, 1] *= 15.0  # S22
    #Frequency weighting: more weight on higher frequencies where deep resonances occur
    num_freqs = S_pred.shape[1]
    freq_weights = torch.linspace(1.0, 10.0, num_freqs).to(S_pred.device) # Linearly increasing weight from 1 to 10 across the frequency spectrum
    #shape frequency weights for broadcasting
    weighted_db_loss = torch.mean(db_mse * freq_weights.view(1, -1, 1, 1))
    #Damping penalty for poles to stay away from the imaginary axis, encouraging causality and numerical stability.
    damping_penalty = torch.mean(1.0 / (torch.abs(poles.real) + 1e-5))
    #residue penalty to encourage passivity by penalizing large positive real parts in residues
    residue_l2 = torch.mean(torch.abs(residues)**2)
    #returened combined loss with appropriate weighting
    return (alpha * mse) + (beta * weighted_db_loss) + (0.05 * damping_penalty) + (1e-3 * residue_l2)

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
    print(f"--- Training RationalNet on {device.type.upper()} ({dataset_type.upper()} Dataset) ---")

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

    # Instantiate RationalNet
    model = RationalNet(
        num_poles=NUM_POLES,
        num_local_features=num_local,
        num_global_features=num_global,
        num_ports=4
    ).to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # Isolate weight decay to prevent fighting the softplus causality constraint on poles
    pole_params = []
    base_params = []
    for name, param in model.named_parameters():
        if 'poles' in name:
            pole_params.append(param)
        else:
            base_params.append(param)

    optimizer = optim.AdamW([
        {'params': base_params, 'lr': LR_BASE, 'weight_decay': WEIGHT_DECAY},
        {'params': pole_params, 'lr': LR_POLES, 'weight_decay': 0.0}
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
        
        for x_loc, x_glob, y_r, y_i in tqdm(train_loader, desc=f"Epoch {epoch:03d} [Train]", leave=False):
            x_loc, x_glob = x_loc.to(device), x_glob.to(device)
            Y_true = torch.complex(y_r, y_i).to(device)
            
            optimizer.zero_grad()
            
            poles, residues, d_term = model(x_loc, x_glob)
            S_pred = model.predict_frequency_response(poles, residues, d_term, frequencies_hz)
            
            loss = combined_loss(S_pred, Y_true, poles, residues, alpha=LOSS_ALPHA, beta=LOSS_BETA)
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
                S_pred = model.predict_frequency_response(poles, residues, d_term, frequencies_hz)
                
                loss = combined_loss(S_pred, Y_true, poles, residues, alpha=LOSS_ALPHA, beta=LOSS_BETA)
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
            print(f"Epoch [{epoch:03d}/{EPOCHS}] | Train Loss: {epoch_train_loss:.4e} | Val Loss: {epoch_val_loss:.4e} | LR_base: {current_lr_base:.2e} | LR_pole: {current_lr_pole:.2e}")

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
    all_poles = []
    all_residues = []
    
    with torch.no_grad():
        for x_loc, x_glob, y_r, y_i in tqdm(test_loader, desc="Testing", leave=False):
            x_loc, x_glob = x_loc.to(device), x_glob.to(device)
            Y_true = torch.complex(y_r, y_i).to(device)
            
            poles, residues, d_term = model(x_loc, x_glob)
            S_pred = model.predict_frequency_response(poles, residues, d_term, frequencies_hz)
            
            loss = combined_loss(S_pred, Y_true, poles, residues, alpha=LOSS_ALPHA, beta=LOSS_BETA)
            test_loss += loss.item() * x_loc.size(0)
            
            all_poles.append(poles)
            all_residues.append(residues)
            
    test_loss /= len(test_loader.dataset)
    print(f"Final Test Loss (Combined MSE + dB): {test_loss:.4e}")
    
    all_poles = torch.cat(all_poles, dim=0)
    all_residues = torch.cat(all_residues, dim=0)
    physics_checks = model.verify_physics_constraints(all_poles, all_residues)
        
    print(f"Causality Preserved: {'PASSED' if physics_checks['causality_preserved'] else 'FAILED'}")
    print(f"Symmetry Preserved:  {'PASSED' if physics_checks['conjugate_symmetry_preserved'] else 'FAILED'}")
    print(f"Passivity Preserved: {'PASSED' if physics_checks['passivity_preserved'] else 'FAILED'} (Min Eigenvalue: {physics_checks['min_residue_eigenvalue']:.2e})")

if __name__ == "__main__":
    # To run on the Array dataset later, swap 'link' for 'array'
    train_model(dataset_type='link')