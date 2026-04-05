import sys
import os
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

"""
Two-Stage Training Pipeline for RationalNet
=============================================
Stage 2: Supervised training on VF-computed pole/residue targets.
         The network learns to predict WHERE poles go and HOW BIG residues are
         for each geometry, using direct MSE supervision. This is a much easier
         optimization problem than end-to-end S-parameter fitting because the
         loss landscape is smooth — small changes in predicted poles produce
         small changes in the pole-prediction loss.

Stage 3: End-to-end fine-tuning with the S-parameter combined loss.
         Starting from the Stage 2 initialisation (which already produces
         good poles/residues), gradient descent only needs to make small
         adjustments. This avoids the non-convex optimisation nightmare
         that caused the pure end-to-end approach to fail.

Causality is preserved by construction throughout — the model architecture
is unchanged, with softplus-enforced negative real parts on all poles.
"""

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJ_ROOT)

from src.data.dataset import get_dataloaders
from src.models.rational_net import RationalNet

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
NUM_POLES = 80
HIDDEN_DIM = 512
BATCH_SIZE = 32

# Stage 2: Supervised pole/residue learning
STAGE2_EPOCHS = 200
STAGE2_LR = 1e-3
STAGE2_PATIENCE = 30

# Stage 3: End-to-end fine-tuning on S-parameters
STAGE3_EPOCHS = 200
STAGE3_LR = 1e-4       # Lower LR for fine-tuning — we're refining, not discovering
STAGE3_LR_POLES = 5e-5  # Even lower for poles — they're already close to optimal from VF
STAGE3_PATIENCE = 40

WEIGHT_DECAY = 1e-3

# S-parameter loss weights (same as train_rationalnet.py)
LOSS_ALPHA = 1.0
LOSS_BETA  = 0.5

# ---------------------------------------------------------------------------
# VF-Supervised Dataset
# ---------------------------------------------------------------------------
class VFSupervisedDataset(Dataset):
    """
    Dataset that pairs geometry features with VF-computed pole/residue targets.
    Filters out any samples where VF failed (NaN targets).
    """
    def __init__(self, data_path, vf_targets_path, dataset_type='array'):
        # Load original dataset for geometry features
        data = torch.load(data_path, weights_only=False)
        self.feature_names = data['feature_names']
        
        # Load VF targets
        vf = torch.load(vf_targets_path, weights_only=False)
        
        # Feature splitting (same logic as SIPIDataset)
        if dataset_type == 'link':
            local_features = ['VIA_RADIUS', 'ANTIPAD_RADIUS', 'PITCH']
        else:
            local_features = ['VIA_RADIUS', 'ANTIPAD_RADIUS', 'PITCH',
                              'VIAS_X_AMOUNT', 'VIAS_Y_AMOUNT']
        
        local_indices = [self.feature_names.index(f) for f in local_features if f in self.feature_names]
        global_indices = [i for i in range(len(self.feature_names)) if i not in local_indices]
        
        X = data['X']
        self.x_local = X[:, local_indices]
        self.x_global = X[:, global_indices]
        
        # VF targets (only upper-half poles, model mirrors for conjugate symmetry)
        self.poles_real_target = vf['poles_real']      # [num_samples, num_poles_half]
        self.poles_imag_target = vf['poles_imag']      # [num_samples, num_poles_half]
        self.residues_real_target = vf['residues_real'] # [num_samples, num_poles_half, 4, 4]
        self.residues_imag_target = vf['residues_imag'] # [num_samples, num_poles_half, 4, 4]
        self.d_term_target = vf['d_term']               # [num_samples, 4, 4]
        
        # Also keep S-parameter targets for Stage 3 and evaluation
        self.y_real = data['Y_real']
        self.y_imag = data['Y_imag']
        
        # Filter out failed VF samples (NaN in poles)
        valid_mask = ~torch.isnan(self.poles_real_target[:, 0])
        self.valid_indices = torch.where(valid_mask)[0]
        
        print(f"  VF Supervised Dataset: {len(self.valid_indices)}/{len(X)} valid samples "
              f"({len(X) - len(self.valid_indices)} VF failures filtered)")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        return (
            self.x_local[real_idx],
            self.x_global[real_idx],
            self.poles_real_target[real_idx],
            self.poles_imag_target[real_idx],
            self.residues_real_target[real_idx],
            self.residues_imag_target[real_idx],
            self.d_term_target[real_idx],
            self.y_real[real_idx],
            self.y_imag[real_idx],
        )

# ---------------------------------------------------------------------------
# Stage 2 Loss: Supervised Pole/Residue Prediction
# ---------------------------------------------------------------------------
def pole_residue_loss(pred_poles, pred_residues, pred_d_term,
                      target_poles_real, target_poles_imag,
                      target_res_real, target_res_imag, target_d):
    """
    Direct supervision loss on poles, residues, and D term.
    
    Compares the network's predicted rational function coefficients against
    the VF-computed ground truth. Both are sorted by imaginary part (frequency)
    so pole k always corresponds to roughly the same frequency region.
    
    The loss is a weighted combination of:
      - Pole location MSE (real + imaginary parts separately)
      - Residue MSE (real + imaginary parts)
      - D term MSE
    
    Pole loss is weighted higher because pole locations have the most dramatic
    effect on the frequency response — a misplaced pole shifts an entire resonance.
    """
    batch_size = pred_poles.shape[0]
    num_poles_half = pred_poles.shape[1] // 2
    
    # Extract upper-half predicted poles (positive imaginary) and sort by frequency
    pred_poles_upper = pred_poles[:, :num_poles_half]  # First half = upper
    pred_poles_real = pred_poles_upper.real
    pred_poles_imag = pred_poles_upper.imag
    
    # Sort predicted poles by imaginary part to match VF target ordering
    sort_idx = torch.argsort(pred_poles_imag, dim=1)
    pred_poles_real_sorted = torch.gather(pred_poles_real, 1, sort_idx)
    pred_poles_imag_sorted = torch.gather(pred_poles_imag, 1, sort_idx)
    
    # Sort predicted residues to match
    # Expand sort_idx for the [4,4] residue dimensions
    sort_idx_expanded = sort_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 4)
    pred_res_upper = pred_residues[:, :num_poles_half]
    pred_res_real_sorted = torch.gather(pred_res_upper.real, 1, sort_idx_expanded)
    pred_res_imag_sorted = torch.gather(pred_res_upper.imag, 1, sort_idx_expanded)
    
    # Pole location loss (weighted higher — poles drive resonance positions)
    pole_real_loss = F.mse_loss(pred_poles_real_sorted, target_poles_real)
    pole_imag_loss = F.mse_loss(pred_poles_imag_sorted, target_poles_imag)
    
    # Residue loss
    res_real_loss = F.mse_loss(pred_res_real_sorted, target_res_real)
    res_imag_loss = F.mse_loss(pred_res_imag_sorted, target_res_imag)
    
    # Direct term loss
    d_loss = F.mse_loss(pred_d_term.real, target_d)
    
    # Weighted combination — poles get 5x weight because they control resonance positions
    total = 5.0 * (pole_real_loss + pole_imag_loss) + (res_real_loss + res_imag_loss) + d_loss
    
    return total, {
        'pole_real': pole_real_loss.item(),
        'pole_imag': pole_imag_loss.item(),
        'residue_real': res_real_loss.item(),
        'residue_imag': res_imag_loss.item(),
        'd_term': d_loss.item(),
    }

# ---------------------------------------------------------------------------
# Stage 3 Loss: S-parameter Combined Loss (same as train_rationalnet.py)
# ---------------------------------------------------------------------------
def passivity_penalty(residues):    
    """Soft penalty for non-passive residues."""    
    R_real = residues.real  
    R_sym = (R_real + R_real.transpose(-1, -2)) / 2.0
    eye = torch.eye(4, device=R_real.device).view(1, 1, 4, 4) * 1e-6
    R_reg = R_sym + eye
    eigvals = torch.linalg.eigvalsh(R_reg.view(-1, 4, 4))     
    negative_eigvals = torch.clamp(eigvals, max=0.0)    
    return torch.mean(torch.abs(negative_eigvals))

def sparam_combined_loss(S_pred, Y_true, poles, residues, epoch=0):
    """
    Same combined loss as train_rationalnet.py — Complex MSE + dB MAE + slope + passivity.
    """
    # Complex MSE
    diff = S_pred - Y_true
    mse = torch.mean(diff.real**2 + diff.imag**2)
    
    # Frequency-weighted dB loss
    num_freqs = S_pred.shape[1]
    freq_weights = torch.logspace(0.0, 1.0, steps=num_freqs).to(S_pred.device).view(1, -1, 1, 1)
    epsilon = 1e-4
    db_pred = 20 * torch.log10(torch.abs(S_pred) + epsilon)
    db_true = 20 * torch.log10(torch.abs(Y_true) + epsilon)
    db_mae = torch.abs(db_pred - torch.clamp(db_true, min=-70.0))
    final_db_loss = torch.mean(db_mae * freq_weights)
    
    # Sobolev slope loss
    diff_pred = S_pred[:, 1:, :, :] - S_pred[:, :-1, :, :]
    diff_true = Y_true[:, 1:, :, :] - Y_true[:, :-1, :, :]
    slope_loss = torch.mean(torch.abs(diff_pred - diff_true))
    
    # Soft passivity penalty
    p_loss = passivity_penalty(residues)
    
    slope_weight = 0.1 if epoch > 10 else 0.0
    passivity_weight = min(10.0, 1.0 + (9.0 * epoch / 50.0))
    
    return (LOSS_ALPHA * mse) + (LOSS_BETA * final_db_loss) + (slope_weight * slope_loss) + (passivity_weight * p_loss)

# ---------------------------------------------------------------------------
# Main Two-Stage Training Pipeline
# ---------------------------------------------------------------------------
def train_twostage(dataset_type='array'):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"{'='*70}")
    print(f"  Two-Stage RationalNet Training on {device.type.upper()} ({dataset_type.upper()} Dataset)")
    print(f"{'='*70}")
    
    # Load VF-supervised dataset
    dataset_folder = 'Universal-Diff-SI-Array' if dataset_type == 'array' else 'Universal-Diff-SI-Link'
    data_path = os.path.join(PROJ_ROOT, f"data/processed/{dataset_folder}/via_{dataset_type}_dataset.pt")
    vf_path = os.path.join(PROJ_ROOT, f"data/processed/{dataset_folder}/vf_targets_{dataset_type}.pt")
    
    if not os.path.exists(vf_path):
        print(f"[ERROR] VF targets not found at {vf_path}")
        print(f"        Run prefit_vectorfitting.py --dataset_type {dataset_type} first.")
        return
    
    dataset = VFSupervisedDataset(data_path, vf_path, dataset_type=dataset_type)
    
    # Split into train/val/test (80/10/10)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    
    print(f"  Samples: Train={n_train}, Val={n_val}, Test={n_test}")
    
    # Extract dimensions
    sample = dataset[0]
    num_local = sample[0].shape[0]
    num_global = sample[1].shape[0]
    num_freqs = sample[7].shape[0]  # y_real
    frequencies_hz = torch.linspace(0.25e9, 100e9, num_freqs).to(device)
    
    # Instantiate model
    model = RationalNet(
        num_poles=NUM_POLES,
        num_local_features=num_local,
        num_global_features=num_global,
        num_ports=4,
        hidden_dim=HIDDEN_DIM
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    save_dir = os.path.join(PROJ_ROOT, "results/checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    
    history = {
        'stage2_train_loss': [], 'stage2_val_loss': [],
        'stage3_train_loss': [], 'stage3_val_loss': [],
        'stage2_lr': [], 'stage3_lr': [],
    }

    # ==========================================
    # STAGE 2: Supervised Pole/Residue Learning
    # ==========================================
    print(f"\n{'='*70}")
    print(f"  STAGE 2: Supervised Pole/Residue Learning ({STAGE2_EPOCHS} epochs)")
    print(f"{'='*70}")
    
    optimizer_s2 = optim.AdamW(model.parameters(), lr=STAGE2_LR, weight_decay=WEIGHT_DECAY)
    scheduler_s2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_s2, mode='min', factor=0.5, patience=10)
    
    best_s2_val = float('inf')
    s2_no_improve = 0
    stage2_path = os.path.join(save_dir, f"stage2_rational_net_{dataset_type}.pth")
    
    for epoch in range(1, STAGE2_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"S2 Epoch {epoch:03d} [Train]", leave=False):
            x_loc, x_glob = batch[0].to(device), batch[1].to(device)
            tgt_pr, tgt_pi = batch[2].to(device), batch[3].to(device)
            tgt_rr, tgt_ri = batch[4].to(device), batch[5].to(device)
            tgt_d = batch[6].to(device)
            
            optimizer_s2.zero_grad()
            poles, residues, d_term = model(x_loc, x_glob)
            
            loss, _ = pole_residue_loss(
                poles, residues, d_term,
                tgt_pr, tgt_pi, tgt_rr, tgt_ri, tgt_d
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_s2.step()
            epoch_loss += loss.item() * x_loc.size(0)
        
        epoch_loss /= n_train
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"S2 Epoch {epoch:03d} [Val]", leave=False):
                x_loc, x_glob = batch[0].to(device), batch[1].to(device)
                tgt_pr, tgt_pi = batch[2].to(device), batch[3].to(device)
                tgt_rr, tgt_ri = batch[4].to(device), batch[5].to(device)
                tgt_d = batch[6].to(device)
                
                poles, residues, d_term = model(x_loc, x_glob)
                loss, _ = pole_residue_loss(
                    poles, residues, d_term,
                    tgt_pr, tgt_pi, tgt_rr, tgt_ri, tgt_d
                )
                val_loss += loss.item() * x_loc.size(0)
        
        val_loss /= n_val
        scheduler_s2.step(val_loss)
        
        current_lr = optimizer_s2.param_groups[0]['lr']
        history['stage2_train_loss'].append(epoch_loss)
        history['stage2_val_loss'].append(val_loss)
        history['stage2_lr'].append(current_lr)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"  S2 [{epoch:03d}/{STAGE2_EPOCHS}] | Train: {epoch_loss:.4e} | Val: {val_loss:.4e} | LR: {current_lr:.2e}")
        
        if val_loss < best_s2_val:
            best_s2_val = val_loss
            s2_no_improve = 0
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'val_loss': best_s2_val, 'stage': 2,
                'num_poles': NUM_POLES, 'hidden_dim': HIDDEN_DIM,
                'num_local': num_local, 'num_global': num_global,
                'dataset_type': dataset_type,
            }, stage2_path)
        else:
            s2_no_improve += 1
            if s2_no_improve >= STAGE2_PATIENCE:
                print(f"  [INFO] Stage 2 early stopping at epoch {epoch}")
                break
    
    # Reload best Stage 2 checkpoint before Stage 3
    checkpoint = torch.load(stage2_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    print(f"\n  [INFO] Stage 2 complete. Best val loss: {best_s2_val:.4e} (epoch {checkpoint['epoch']})")

    # ==========================================
    # STAGE 3: End-to-End Fine-Tuning
    # ==========================================
    print(f"\n{'='*70}")
    print(f"  STAGE 3: End-to-End Fine-Tuning ({STAGE3_EPOCHS} epochs)")
    print(f"{'='*70}")
    
    # Separate parameter groups with lower LR for poles (they're already well-positioned)
    pole_params = [p for n, p in model.named_parameters() if 'poles' in n]
    residue_params = [p for n, p in model.named_parameters() if 'residues' in n or 'd_term' in n]
    base_params = [p for n, p in model.named_parameters() if 'poles' not in n and 'residues' not in n and 'd_term' not in n]
    
    optimizer_s3 = optim.AdamW([
        {'params': base_params, 'lr': STAGE3_LR, 'weight_decay': WEIGHT_DECAY},
        {'params': pole_params, 'lr': STAGE3_LR_POLES, 'weight_decay': 0.0},
        {'params': residue_params, 'lr': STAGE3_LR, 'weight_decay': 0.0}
    ], lr=STAGE3_LR)
    
    scheduler_s3 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_s3, mode='min', factor=0.5, patience=10)
    
    best_s3_val = float('inf')
    s3_no_improve = 0
    best_model_path = os.path.join(save_dir, f"best_rational_net_2stage_{dataset_type}.pth")
    
    for epoch in range(1, STAGE3_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        
        # Curriculum: start at 50% bandwidth (model already knows low-freq from Stage 2)
        # and expand to 100% over 50 epochs
        progress = min(1.0, epoch / 50.0)
        freq_limit = int(num_freqs * (0.5 + 0.5 * progress))
        
        for batch in tqdm(train_loader, desc=f"S3 Epoch {epoch:03d} [Train]", leave=False):
            x_loc, x_glob = batch[0].to(device), batch[1].to(device)
            y_r, y_i = batch[7].to(device), batch[8].to(device)
            Y_true = torch.complex(y_r, y_i)
            
            optimizer_s3.zero_grad()
            poles, residues, d_term = model(x_loc, x_glob)
            S_pred = model.predict_frequency_response(poles, residues, d_term, frequencies_hz[:freq_limit])
            
            loss = sparam_combined_loss(S_pred, Y_true[:, :freq_limit], poles, residues, epoch=epoch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_s3.step()
            epoch_loss += loss.item() * x_loc.size(0)
        
        epoch_loss /= n_train
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"S3 Epoch {epoch:03d} [Val]", leave=False):
                x_loc, x_glob = batch[0].to(device), batch[1].to(device)
                y_r, y_i = batch[7].to(device), batch[8].to(device)
                Y_true = torch.complex(y_r, y_i)
                
                poles, residues, d_term = model(x_loc, x_glob)
                S_pred = model.predict_frequency_response(poles, residues, d_term, frequencies_hz[:freq_limit])
                loss = sparam_combined_loss(S_pred, Y_true[:, :freq_limit], poles, residues, epoch=epoch)
                val_loss += loss.item() * x_loc.size(0)
        
        val_loss /= n_val
        scheduler_s3.step(val_loss)
        
        current_lr = optimizer_s3.param_groups[0]['lr']
        history['stage3_train_loss'].append(epoch_loss)
        history['stage3_val_loss'].append(val_loss)
        history['stage3_lr'].append(current_lr)
        
        if epoch % 5 == 0 or epoch == 1:
            freq_pct = (0.5 + 0.5 * progress) * 100
            print(f"  S3 [{epoch:03d}/{STAGE3_EPOCHS}] | Train: {epoch_loss:.4e} | Val: {val_loss:.4e} | LR: {current_lr:.2e} | Freq: {freq_pct:.0f}%")
        
        if val_loss < best_s3_val:
            best_s3_val = val_loss
            s3_no_improve = 0
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'optimizer_state': optimizer_s3.state_dict(),
                'val_loss': best_s3_val, 'stage': 3,
                'num_poles': NUM_POLES, 'hidden_dim': HIDDEN_DIM,
                'num_local': num_local, 'num_global': num_global,
                'dataset_type': dataset_type,
            }, best_model_path)
        else:
            s3_no_improve += 1
            if s3_no_improve >= STAGE3_PATIENCE:
                print(f"  [INFO] Stage 3 early stopping at epoch {epoch}")
                break
    
    # Save history
    history_path = os.path.join(save_dir, f"training_history_2stage_{dataset_type}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f"\n[INFO] Training history saved to {history_path}")

    # ==========================================
    # Final Evaluation on Test Set
    # ==========================================
    print(f"\n{'='*70}")
    print(f"  Final Evaluation on Test Set")
    print(f"{'='*70}")
    
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print(f"  Loaded best Stage 3 checkpoint from epoch {checkpoint['epoch']} (val loss: {checkpoint['val_loss']:.4e})")
    
    test_loss = 0.0
    total_db_mae = 0.0
    total_phase_mae = 0.0
    all_poles = []
    all_residues = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            x_loc, x_glob = batch[0].to(device), batch[1].to(device)
            y_r, y_i = batch[7].to(device), batch[8].to(device)
            Y_true = torch.complex(y_r, y_i)
            
            poles, residues, d_term = model(x_loc, x_glob)
            S_pred = model.predict_frequency_response(poles, residues, d_term, frequencies_hz)
            
            loss = sparam_combined_loss(S_pred, Y_true, poles, residues, epoch=STAGE3_EPOCHS)
            test_loss += loss.item() * x_loc.size(0)
            
            eps = 1e-4
            db_err = torch.abs((20*torch.log10(torch.abs(S_pred)+eps)) - (20*torch.log10(torch.abs(Y_true)+eps)))
            phase_err = torch.abs(torch.angle(S_pred) - torch.angle(Y_true)) * (180.0 / torch.pi)
            
            total_db_mae += torch.mean(db_err).item() * x_loc.size(0)
            total_phase_mae += torch.mean(phase_err).item() * x_loc.size(0)
            
            all_poles.append(poles)
            all_residues.append(residues)
    
    test_loss /= n_test
    avg_db_mae = total_db_mae / n_test
    avg_phase_mae = total_phase_mae / n_test
    
    print(f"\n  Final Test Loss (Combined): {test_loss:.4e}")
    print(f"  Final Mean Absolute Error (dB): {avg_db_mae:.2f} dB")
    print(f"  Final Phase Error (Degrees): {avg_phase_mae:.2f}°")
    
    all_poles = torch.cat(all_poles, dim=0)
    all_residues = torch.cat(all_residues, dim=0)
    physics_checks = model.verify_physics_constraints(all_poles, all_residues)
    
    print(f"\n  Physics Constraint Verification:")
    print(f"    Causality Preserved: {'PASSED' if physics_checks['causality_preserved'] else 'FAILED'}")
    print(f"    Symmetry Preserved:  {'PASSED' if physics_checks['conjugate_symmetry_preserved'] else 'FAILED'}")
    print(f"    Passivity Preserved: {'PASSED' if physics_checks['passivity_preserved'] else 'FAILED'} (Min Eigenvalue: {physics_checks['min_residue_eigenvalue']:.2e})")

if __name__ == "__main__":
    for dtype in ['array', 'link']:
        train_twostage(dataset_type=dtype)