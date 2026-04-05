import sys
import os
import json
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# Add the project root to the system path
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJ_ROOT)

from src.data.dataset import get_dataloaders
from src.models.hybridnet import HybridNet

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
NUM_POLES = 12       
BATCH_SIZE = 32
EPOCHS = 400
PATIENCE = 50
LR_BASE = 1e-3

# The known index of 'LENGTH' in your global features list
IDX_LENGTH = 7  

# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------
def complex_mse_loss(S_pred, Y_true):
    """Calculates Mean Squared Error on real and imaginary parts equally."""
    return torch.mean(torch.abs(S_pred - Y_true)**2)

def combined_loss(S_pred, Y_true, epoch=0):
    """
    Combined Loss for the Hybrid TMPT.
    Prioritizes phase accuracy and sharp resonance matching.
    """
    # 1. Complex MSE (Phase + Mag Anchor)
    mse = complex_mse_loss(S_pred, Y_true)
    
    # 2. Log-Frequency Weighted dB Magnitude Loss
    num_freqs = S_pred.shape[1]
    freq_weights = torch.logspace(0.0, 1.0, steps=num_freqs).to(S_pred.device).view(1, -1, 1, 1) 
    
    eps = 1e-4
    db_pred = 20 * torch.log10(torch.abs(S_pred) + eps)
    db_true = 20 * torch.log10(torch.abs(Y_true) + eps)
    
    # -70dB noise floor clamp
    db_mae = torch.abs(db_pred - torch.clamp(db_true, min=-70.0))
    final_db_loss = torch.mean(db_mae * freq_weights)
    
    # 3. Sobolev Slope Loss (The Icicle Maker)
    diff_pred = S_pred[:, 1:, :, :] - S_pred[:, :-1, :, :]
    diff_true = Y_true[:, 1:, :, :] - Y_true[:, :-1, :, :]
    slope_loss = torch.mean(torch.abs(diff_pred - diff_true))
    
    # Wait until the curriculum reaches 100% (Epoch 50) before enforcing Sobolev slopes
    slope_weight = 100.0 if epoch > 50 else 0.0
    
    return (10.0 * mse) + (1.0 * final_db_loss) + (slope_weight * slope_loss)

# ---------------------------------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------------------------------
def train_hybrid_model(dataset_type='link'):
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training HybridNet TMPT on {device.type.upper()} ({dataset_type.upper()} Dataset) ---")

    # Data Loading
    dataset_folder = 'Universal-Diff-SI-Array' if dataset_type == 'array' else 'Universal-Diff-SI-Link'
    data_path = os.path.join(PROJ_ROOT, f"data/processed/{dataset_folder}/via_{dataset_type}_dataset.pt")
    train_loader, val_loader, test_loader = get_dataloaders(data_path, dataset_type=dataset_type, batch_size=BATCH_SIZE)
    
    # Frequency setup (0.25 GHz to 100 GHz)
    x_local, x_global, y_real, y_imag = next(iter(train_loader))
    num_freqs = y_real.shape[1]
    frequencies_hz = torch.linspace(0.25e9, 100e9, num_freqs).to(device)

    # Auto-Extract Length Metadata for physical denormalization
    # The double .dataset bypasses the PyTorch Subset wrapper
    dataset_ref = train_loader.dataset.dataset
    master_length_idx = dataset_ref.feature_names.index('LENGTH')
    length_mean_val = dataset_ref.X_mean[master_length_idx].item()
    length_std_val = dataset_ref.X_std[master_length_idx].item()
    print(f"[INFO] Length Stats -> Mean: {length_mean_val:.4f}, Std: {length_std_val:.4f}")

    # Model Setup
    model = HybridNet(
        num_poles=NUM_POLES,
        num_local_features=x_local.shape[1],
        num_global_features=x_global.shape[1],
        num_ports=4,
        length_mean=length_mean_val, 
        length_std=length_std_val   
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR_BASE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    save_dir = os.path.join(PROJ_ROOT, "results/checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"best_hybridnet_{dataset_type}.pth")

    # ==========================================
    # Epoch Loop
    # ==========================================
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_train_loss = 0.0

        # Curriculum Logic: Increase frequency bandwidth over the first 50 epochs
        progress = min(1.0, epoch / 50.0)
        idx_limit = int(num_freqs * (0.2 + 0.8 * progress))
        curr_freqs = frequencies_hz[:idx_limit]
        
        for x_loc, x_glob, y_r, y_i in tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False):
            x_loc, x_glob = x_loc.to(device), x_glob.to(device)
            Y_true = torch.complex(y_r, y_i).to(device)[:, :idx_limit]
            
            optimizer.zero_grad()
            
            # 1. Parameter Extraction
            poles, residues, d_term, z0_pred, eps_eff_pred = model(x_loc, x_glob)
            
            # 2. Physics Cascade
            S_via = model.predict_via_response(poles, residues, d_term, curr_freqs)
            M_via = model.s_to_abcd(S_via)
            
            # --- CRITICAL UNIT FIX ---
            # Denormalize length. Assuming database is in INCHES based on Mean=3.93. 
            # Multiply by 0.0254 to convert to METERS. (Change to 0.01 if cm, 0.001 if mm).
            length_m = ((x_glob[:, IDX_LENGTH] * model.length_std) + model.length_mean) * 0.0254
            
            M_line = model.get_line_matrix(z0_pred, eps_eff_pred, curr_freqs, length_m)
            
            # M_total = Via_in * Line * Via_out
            M_total = torch.matmul(torch.matmul(M_via, M_line), M_via)
            
            # 3. Back to S-domain + Passivity Enforcement
            S_pred = model.enforce_passivity(model.abcd_to_s(M_total))
            
            loss = combined_loss(S_pred, Y_true, epoch=epoch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item() * x_loc.size(0)
            
        epoch_train_loss /= len(train_loader.dataset)

        # Validation Loop
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x_loc, x_glob, y_r, y_i in val_loader:
                x_loc, x_glob = x_loc.to(device), x_glob.to(device)
                Y_true = torch.complex(y_r, y_i).to(device)[:, :idx_limit]
                
                poles, residues, d_term, z0_pred, eps_eff_pred = model(x_loc, x_glob)
                S_via = model.predict_via_response(poles, residues, d_term, curr_freqs)
                
                length_m = ((x_glob[:, IDX_LENGTH] * model.length_std) + model.length_mean) * 0.0254
                M_line = model.get_line_matrix(z0_pred, eps_eff_pred, curr_freqs, length_m)
                
                M_total = torch.matmul(torch.matmul(model.s_to_abcd(S_via), M_line), model.s_to_abcd(S_via))
                S_pred = model.enforce_passivity(model.abcd_to_s(M_total))
                
                epoch_val_loss += combined_loss(S_pred, Y_true, epoch=epoch).item() * x_loc.size(0)
                
        epoch_val_loss /= len(val_loader.dataset)
        
        # --- CURRICULUM CLASH FIX ---
        # Do not adjust the learning rate or check early stopping until the curriculum finishes.
        if epoch > 50:
            scheduler.step(epoch_val_loss)
            
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_no_improve = 0
                torch.save({'model_state': model.state_dict(), 'epoch': epoch}, best_model_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"[INFO] Early stopping triggered at epoch {epoch}")
                    break
        else:
            # During warmup, always save the latest model to guarantee we have a checkpoint
            best_val_loss = epoch_val_loss
            torch.save({'model_state': model.state_dict(), 'epoch': epoch}, best_model_path)
            
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}] | Loss: {epoch_train_loss:.3e} | Val: {epoch_val_loss:.3e} | LR: {history['lr'][-1]:.1e}")

    # ==========================================
    # Final Evaluation (The Thesis Data)
    # ==========================================
    print("\n--- Running Final Evaluation ---")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    total_db_mae = 0.0
    total_phase_mae = 0.0

    with torch.no_grad():
        for x_loc, x_glob, y_r, y_i in test_loader:
            x_loc, x_glob = x_loc.to(device), x_glob.to(device)
            Y_true = torch.complex(y_r, y_i).to(device)
            
            p, r, d, z0, eps_e = model(x_loc, x_glob)
            S_via = model.predict_via_response(p, r, d, frequencies_hz)
            
            length_m = ((x_glob[:, IDX_LENGTH] * model.length_std) + model.length_mean) * 0.0254
            M_line = model.get_line_matrix(z0, eps_e, frequencies_hz, length_m)
            
            M_total = torch.matmul(torch.matmul(model.s_to_abcd(S_via), M_line), model.s_to_abcd(S_via))
            S_pred = model.enforce_passivity(model.abcd_to_s(M_total))
            
            # dB MAE
            db_err = torch.abs(20*torch.log10(torch.abs(S_pred)+1e-4) - 20*torch.log10(torch.abs(Y_true)+1e-4))
            total_db_mae += torch.mean(db_err).item() * x_loc.size(0)
            
            # Phase MAE
            phase_err = torch.abs(torch.angle(S_pred) - torch.angle(Y_true)) * (180/np.pi)
            total_phase_mae += torch.mean(phase_err).item() * x_loc.size(0)

    print(f"Final Test Result -> MAE: {total_db_mae/len(test_loader.dataset):.2f} dB | Phase Error: {total_phase_mae/len(test_loader.dataset):.2f}°")

    # --- THE LOGGING FIX ---
    history_path = os.path.join(save_dir, f"training_history_{dataset_type}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f"[SUCCESS] Training history successfully saved to {history_path}")

if __name__ == "__main__":
    train_hybrid_model(dataset_type='link')