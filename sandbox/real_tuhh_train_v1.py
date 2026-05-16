"""
==========================================================================
COMPLETE TUHH PIPELINE: Raw Touchstone → Parsed Tensors → Trained Model
==========================================================================
Self-contained script. No external imports except standard libraries.

PHASE 1: Parse raw touchstone files
  - Reads parameter.csv + variation/*/touchstone files
  - Frequency interpolation to 0-100 GHz (401 points)
  - 4x4 core matrix extraction from N-port simulations
  - Reciprocity enforcement
  - Passivity check (eigenvalue-based, drops violating samples)
  - Mixed-mode conversion (single-ended → differential)
  - Log-scaling of material properties
  - Z-score normalization
  - Saves processed .pt file

PHASE 2: Train forward surrogate model
  - Direct frequency prediction MLP (ResNet backbone)
  - dB-space SmoothL1 + complex MSE loss
  - Saves best model checkpoint + publication plots

Run: python tuhh_complete_pipeline.py
==========================================================================
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime

try:
    import skrf as rf
    HAS_SKRF = True
except ImportError:
    HAS_SKRF = False
    print("WARNING: scikit-rf not installed. Cannot parse raw data.")
    print("Install with: pip install scikit-rf")


# =====================================================================
# PHASE 1: PHYSICS UTILITIES (self-contained, no external imports)
# =====================================================================

def enforce_reciprocity(s_matrix):
    """Force S_ij = S_ji to kill numerical noise. Input: (F, N, N) complex array."""
    return (s_matrix + np.transpose(s_matrix, (0, 2, 1))) / 2.0


def check_passivity(s_matrix, threshold=-1e-6):
    """
    Check if S-parameter matrix is passive at all frequencies.
    Passive iff Q = I - S^H @ S is positive semi-definite.
    Returns: (is_passive: bool, min_eigenvalue: float)
    """
    num_freqs = s_matrix.shape[0]
    num_ports = s_matrix.shape[1]
    min_eig = float('inf')
    
    for f in range(num_freqs):
        S = s_matrix[f]
        Q = np.eye(num_ports) - S.conj().T @ S
        eigs = np.linalg.eigvalsh(Q)
        min_eig = min(min_eig, eigs.min())
    
    return (min_eig >= threshold), min_eig


def convert_to_mixed_mode(s_se):
    """
    Convert single-ended 4x4 S-matrix to mixed-mode.
    S_mm = M @ S_se @ M^(-1)
    
    M = (1/sqrt(2)) * [[1,-1,0,0],[0,0,1,-1],[1,1,0,0],[0,0,1,1]]
    
    Result indices: Sdd11=[0,0], Sdd21=[1,0], Sdc=[0:2,2:4], etc.
    Input: (F, 4, 4) complex array
    """
    M = (1.0 / np.sqrt(2)) * np.array([
        [1, -1,  0,  0],
        [0,  0,  1, -1],
        [1,  1,  0,  0],
        [0,  0,  1,  1]
    ], dtype=np.complex128)
    
    M_inv = np.linalg.inv(M)
    
    num_freqs = s_se.shape[0]
    s_mm = np.zeros_like(s_se)
    for f in range(num_freqs):
        s_mm[f] = M @ s_se[f] @ M_inv
    
    return s_mm


# =====================================================================
# PHASE 1: PARSING PIPELINE
# =====================================================================

def parse_dataset(dataset_type, base_dir, output_dir):
    """
    Parse raw TUHH touchstone files into a processed .pt tensor file.
    
    Args:
        dataset_type: 'link' or 'array'
        base_dir: path to raw data (contains parameter.csv + variation/)
        output_dir: where to save the processed .pt file
    """
    if not HAS_SKRF:
        raise RuntimeError("scikit-rf required for parsing. pip install scikit-rf")
    
    print(f"\n{'='*70}")
    print(f"PARSING: TUHH {dataset_type.upper()} Dataset")
    print(f"{'='*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(base_dir, "parameter.csv")
    variation_dir = os.path.join(base_dir, "variation")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"parameter.csv not found at {csv_path}")
    if not os.path.exists(variation_dir):
        raise FileNotFoundError(f"variation directory not found at {variation_dir}")
    
    # Load parameter CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from parameter.csv")
    
    # Fix typo in Link dataset
    if 'LOSTANGENT' in df.columns:
        df.rename(columns={'LOSTANGENT': 'LOSSTANGENT'}, inplace=True)
        print("Fixed column name: LOSTANGENT → LOSSTANGENT")
    
    sim_ids = df['SIMULATION'].values
    features = df.drop(columns=['SIM_ID', 'SIMULATION'])
    feature_names = list(features.columns)
    
    # Log-scale material properties with extreme variance
    log_features = []
    if 'LOSSTANGENT' in features.columns:
        features['LOSSTANGENT'] = np.log10(features['LOSSTANGENT'].clip(lower=1e-6))
        log_features.append('LOSSTANGENT')
    if 'CONDUCTIVITY' in features.columns:
        features['CONDUCTIVITY'] = np.log10(features['CONDUCTIVITY'].clip(lower=1.0))
        log_features.append('CONDUCTIVITY')
    if dataset_type == 'link' and 'SL_WIDTH' in features.columns:
        features['SL_WIDTH'] = np.log10(features['SL_WIDTH'].clip(lower=1e-6))
        log_features.append('SL_WIDTH')
    
    print(f"Log-scaled features: {log_features}")
    
    X_raw_all = features.values.astype(np.float64)
    
    # Standardized frequency axis: 0 to 100 GHz, 401 points
    std_freqs_hz = np.linspace(0, 100e9, 401)
    std_rf_freq = rf.Frequency.from_f(std_freqs_hz, unit='Hz')
    
    # Process each simulation
    Y_real_list = []
    Y_imag_list = []
    valid_X_raw_list = []
    valid_sim_ids = []
    master_freqs = None
    passivity_violations = 0
    errors = 0
    
    print(f"\nProcessing {len(sim_ids)} simulations...")
    
    for idx in range(len(sim_ids)):
        sim_id = sim_ids[idx]
        sim_folder = os.path.join(variation_dir, str(sim_id))
        
        if not os.path.exists(sim_folder):
            continue
        
        files = glob.glob(os.path.join(sim_folder, "*.s*p"))
        if not files:
            continue
        
        try:
            # Load touchstone file
            network = rf.Network(files[0])
            
            # Interpolate to standard frequency axis (0-100 GHz, 401 pts)
            network.interpolate(std_rf_freq, bounds_error=False, fill_value="extrapolate")
            
            if master_freqs is None:
                master_freqs = network.f
            
            # Extract core 4x4 from N-port simulation
            num_ports = network.s.shape[1]
            half = num_ports // 2
            port_idx = [0, 1, half, half + 1]
            core_s = network.s[:, port_idx, :][:, :, port_idx]
            
            # Enforce reciprocity
            core_s = enforce_reciprocity(core_s)
            
            # Passivity check
            is_passive, min_eig = check_passivity(core_s, threshold=-1e-6)
            if not is_passive:
                passivity_violations += 1
                continue
            
            # Mixed-mode conversion
            mm_matrix = convert_to_mixed_mode(core_s)
            
            # Store as real/imag tensors
            Y_real_list.append(torch.tensor(np.real(mm_matrix), dtype=torch.float32))
            Y_imag_list.append(torch.tensor(np.imag(mm_matrix), dtype=torch.float32))
            
            # Append NUM_PORTS to feature vector
            current_x = np.append(X_raw_all[idx], num_ports)
            valid_X_raw_list.append(current_x)
            valid_sim_ids.append(sim_id)
            
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error in {sim_id}: {e}")
            continue
        
        # Progress
        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx+1}/{len(sim_ids)} "
                  f"(valid: {len(valid_sim_ids)}, dropped: {passivity_violations})")
    
    if not valid_X_raw_list:
        raise RuntimeError("No valid samples processed!")
    
    # Add NUM_PORTS to feature names
    feature_names.append('NUM_PORTS')
    
    # Z-score normalization
    X_raw_valid = np.stack(valid_X_raw_list)
    X_mean = X_raw_valid.mean(axis=0)
    X_std = X_raw_valid.std(axis=0)
    X_std_safe = np.where(X_std == 0, 1e-5, X_std)
    X_normalized = (X_raw_valid - X_mean) / X_std_safe
    
    X_final = torch.tensor(X_normalized, dtype=torch.float32)
    Y_real = torch.stack(Y_real_list)
    Y_imag = torch.stack(Y_imag_list)
    
    # Save
    filename = f"via_{dataset_type}_dataset.pt"
    save_path = os.path.join(output_dir, filename)
    
    torch.save({
        'X': X_final,
        'Y_real': Y_real,
        'Y_imag': Y_imag,
        'feature_names': feature_names,
        'frequencies': torch.tensor(master_freqs, dtype=torch.float32),
        'sim_ids': valid_sim_ids,
        'X_mean': torch.tensor(X_mean, dtype=torch.float32),
        'X_std': torch.tensor(X_std_safe, dtype=torch.float32),
        'log_features': log_features,
        'metadata': {
            'creation_date': datetime.datetime.now().isoformat(),
            'num_samples': len(valid_sim_ids),
            'passivity_violations': passivity_violations,
            'parse_errors': errors,
            'dataset_type': dataset_type,
        }
    }, save_path)
    
    print(f"\n{'='*70}")
    print(f"PARSING COMPLETE: {dataset_type.upper()}")
    print(f"  Valid samples:        {len(valid_sim_ids)}")
    print(f"  Passivity violations: {passivity_violations}")
    print(f"  Parse errors:         {errors}")
    print(f"  X shape:              {X_final.shape}")
    print(f"  Y_real shape:         {Y_real.shape}")
    print(f"  Features:             {feature_names}")
    print(f"  Saved to:             {save_path}")
    print(f"{'='*70}")
    
    return save_path


# =====================================================================
# PHASE 2: DATASET LOADER
# =====================================================================

class TUHHDataset(Dataset):
    def __init__(self, data_path, dataset_type='link'):
        data = torch.load(data_path, weights_only=False)
        
        self.feature_names = data['feature_names']
        self.freqs_ghz = data['frequencies'].numpy() / 1e9
        self.X_mean = data['X_mean']
        self.X_std = data['X_std']
        
        # Feature split
        local_features = [
            'VIA_RADIUS', 'PITCH', 'ANTIPAD_RADIUS', 'TMET', 'TDIEL',
            'CONDUCTIVITY', 'PERMITTIVITY', 'LOSSTANGENT'
        ]
        global_features = [
            'VIAS_X_AMOUNT', 'VIAS_Y_AMOUNT', 'SIGNAL_AMOUNT',
            'GROUND_AMOUNT', 'POWER_AMOUNT', 'LAYER_AMOUNT', 'NUM_PORTS'
        ]
        if dataset_type == 'link':
            local_features.append('SL_WIDTH')
            global_features.append('LENGTH')
        
        local_idx = [self.feature_names.index(f) for f in local_features]
        global_idx = [self.feature_names.index(f) for f in global_features]
        
        self.X = torch.cat([data['X'][:, local_idx], data['X'][:, global_idx]], dim=1)
        self.input_dim = self.X.shape[1]
        
        # Extract Sdd11 and Sdd21
        Y_real = data['Y_real']  # (N, 401, 4, 4)
        Y_imag = data['Y_imag']
        
        s11 = torch.complex(Y_real[:, :, 0, 0], Y_imag[:, :, 0, 0])
        s21 = torch.complex(Y_real[:, :, 1, 0], Y_imag[:, :, 1, 0])
        self.Y = torch.stack([s11, s21], dim=-1)  # (N, 401, 2)
        
        self.num_freqs = self.Y.shape[1]
        
        y_db = 20 * torch.log10(torch.abs(self.Y).clamp(min=1e-9))
        print(f"  Loaded: {len(self.X)} samples, {self.input_dim} features, "
              f"{self.num_freqs} freq points")
        print(f"  Sdd11: [{y_db[:,:,0].min():.1f}, {y_db[:,:,0].max():.1f}] dB")
        print(f"  Sdd21: [{y_db[:,:,1].min():.1f}, {y_db[:,:,1].max():.1f}] dB")
    
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]


# =====================================================================
# PHASE 2: MODEL
# =====================================================================

class ResBlock(nn.Module):
    def __init__(self, d, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d),
            nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
    def forward(self, x): return x + self.net(x)


class ForwardModel(nn.Module):
    def __init__(self, input_dim, num_freqs=401, num_targets=2, 
                 hidden_dim=512, num_blocks=5):
        super().__init__()
        self.num_freqs = num_freqs
        self.num_targets = num_targets
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
            *[ResBlock(hidden_dim) for _ in range(num_blocks)],
            nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, num_freqs * num_targets * 2),
        )
    
    def forward(self, x):
        out = self.net(x)
        out = out.view(-1, self.num_freqs, self.num_targets, 2)
        return torch.complex(out[..., 0], out[..., 1])


# =====================================================================
# PHASE 2: TRAINING
# =====================================================================

def train_model(data_path, dataset_type, results_dir, epochs=600):
    os.makedirs(results_dir, exist_ok=True)
    
    ds = TUHHDataset(data_path, dataset_type)
    
    n = len(ds)
    n_train, n_val = int(0.8*n), int(0.1*n)
    n_test = n - n_train - n_val
    gen = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(ds, [n_train, n_val, n_test], generator=gen)
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=32)
    
    print(f"  Split: {n_train}/{n_val}/{n_test}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForwardModel(input_dim=ds.input_dim, num_freqs=ds.num_freqs).to(device)
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params on {device}")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(train_loader),
        pct_start=0.05, anneal_strategy='cos', div_factor=10, final_div_factor=100)
    
    history = {'train': [], 'val_db': []}
    best_val = float('inf')
    
    print(f"\n  {'Ep':>4} | {'Loss':>9} | {'Val':>7} | {'Best':>7}")
    print(f"  {'-'*40}")
    
    for epoch in range(epochs):
        model.train()
        t_loss, nb = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            
            pm = torch.clamp(torch.abs(pred), min=1e-7)
            tm = torch.clamp(torch.abs(y), min=1e-7)
            pdb = torch.clamp(20*torch.log10(pm), -80, 10)
            tdb = torch.clamp(20*torch.log10(tm), -80, 10)
            loss_db = nn.functional.smooth_l1_loss(pdb, tdb)
            
            progress = min(epoch / (0.3 * epochs), 1.0)
            w_mse = 0.01 + 0.09 * progress
            loss_mse = nn.functional.mse_loss(pred.real, y.real) + nn.functional.mse_loss(pred.imag, y.imag)
            
            loss = loss_db + w_mse * loss_mse
            if torch.isnan(loss): optimizer.zero_grad(); continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            t_loss += loss.item(); nb += 1
        
        model.eval()
        v_err, nv = 0, 0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                pred = model(xv)
                pm = torch.clamp(torch.abs(pred), min=1e-7)
                tm = torch.clamp(torch.abs(yv), min=1e-7)
                pdb = torch.clamp(20*torch.log10(pm), -80, 10)
                tdb = torch.clamp(20*torch.log10(tm), -80, 10)
                v_err += torch.abs(pdb-tdb).mean().item(); nv += 1
        
        avg_t = t_loss/max(nb,1)
        avg_v = v_err/max(nv,1)
        history['train'].append(avg_t); history['val_db'].append(avg_v)
        
        if avg_v < best_val:
            best_val = avg_v
            torch.save({'model_state': model.state_dict(), 'input_dim': ds.input_dim,
                        'num_freqs': ds.num_freqs, 'best_val': best_val, 'epoch': epoch,
                        'dataset_type': dataset_type, 'feature_names': ds.feature_names,
                        'X_mean': ds.X_mean, 'X_std': ds.X_std,
                        'frequencies': torch.tensor(ds.freqs_ghz * 1e9),
                       }, os.path.join(results_dir, "best_forward_model.pth"))
        
        if (epoch+1) % 50 == 0 or epoch == 0:
            mk = " *" if avg_v <= best_val else ""
            print(f"  {epoch+1:4d} | {avg_t:9.4f} | {avg_v:7.2f} | {best_val:7.2f}{mk}")
    
    # TEST
    ckpt = torch.load(os.path.join(results_dir, "best_forward_model.pth"), weights_only=False)
    model.load_state_dict(ckpt['model_state']); model.eval()
    
    preds, tgts = [], []
    with torch.no_grad():
        for xv, yv in test_loader:
            xv, yv = xv.to(device), yv.to(device)
            preds.append(model(xv).cpu()); tgts.append(yv.cpu())
    all_p, all_t = torch.cat(preds), torch.cat(tgts)
    
    pdb = torch.clamp(20*torch.log10(torch.clamp(torch.abs(all_p),min=1e-7)),-80,10)
    tdb = torch.clamp(20*torch.log10(torch.clamp(torch.abs(all_t),min=1e-7)),-80,10)
    per_sample = torch.abs(pdb-tdb).mean(dim=(1,2))
    s11_mae = torch.abs(pdb[:,:,0]-tdb[:,:,0]).mean().item()
    s21_mae = torch.abs(pdb[:,:,1]-tdb[:,:,1]).mean().item()
    test_mae = per_sample.mean().item()
    
    print(f"\n  {'='*50}")
    print(f"  TEST: {dataset_type.upper()} ({len(all_t)} samples)")
    print(f"  Overall: {test_mae:.2f} dB | S11: {s11_mae:.2f} | S21: {s21_mae:.2f}")
    print(f"  Median: {torch.quantile(per_sample,.5):.2f} | 95th: {torch.quantile(per_sample,.95):.2f}")
    print(f"  {'='*50}")
    
    # PLOT
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(f"TUHH {dataset_type.upper()} Forward Model — Test MAE: {test_mae:.2f} dB",
                 fontsize=14, fontweight='bold')
    freqs = ds.freqs_ghz
    
    for col, (lb, fn) in enumerate([
        ("Best", lambda m: torch.argmin(m).item()),
        ("Median", lambda m: torch.argsort(m)[len(m)//2].item()),
        ("Worst", lambda m: torch.argmax(m).item()),
    ]):
        idx = fn(per_sample)
        ax = axes[0, col]
        td = 20*np.log10(np.abs(all_t[idx,:,1].numpy())+1e-12)
        pd = 20*np.log10(np.abs(all_p[idx,:,1].numpy())+1e-12)
        ax.plot(freqs, td, 'b-', lw=2, label='EM Sim')
        ax.plot(freqs, pd, 'r--', lw=2, label='Model')
        ax.set_title(f"{lb} Sdd21 ({per_sample[idx]:.2f} dB)")
        ax.set_xlabel("GHz"); ax.set_ylabel("dB"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    
    bi = torch.argmin(per_sample).item()
    ax = axes[0, 3]
    td = 20*np.log10(np.abs(all_t[bi,:,0].numpy())+1e-12)
    pd = 20*np.log10(np.abs(all_p[bi,:,0].numpy())+1e-12)
    ax.plot(freqs, td, 'b-', lw=2, label='EM Sim')
    ax.plot(freqs, pd, 'r--', lw=2, label='Model')
    ax.set_title(f"Best Sdd11 ({per_sample[bi]:.2f} dB)"); ax.set_xlabel("GHz")
    ax.set_ylabel("dB"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    
    ax = axes[1, 0]
    ax.plot(history['val_db'], 'b-', lw=1); ax.set_title("Convergence")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val MAE (dB)"); ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.hist(per_sample.numpy(), bins=25, color='steelblue', edgecolor='black', alpha=0.8)
    ax.axvline(x=test_mae, color='red', ls='-', lw=2, label=f'Mean: {test_mae:.2f}')
    ax.set_title("MAE Distribution"); ax.set_xlabel("MAE (dB)"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    ax = axes[1, 2]
    ferr = np.abs(pdb.numpy()-tdb.numpy()).mean(axis=0)
    ax.plot(freqs, ferr[:,0], 'b-', lw=1.5, label='Sdd11')
    ax.plot(freqs, ferr[:,1], 'r-', lw=1.5, label='Sdd21')
    ax.set_title("MAE vs Freq"); ax.set_xlabel("GHz"); ax.set_ylabel("|Error| dB")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    
    ax = axes[1, 3]
    ax.text(0.5, 0.5, f"TUHH {dataset_type.upper()}\n{'─'*25}\n"
            f"MAE: {test_mae:.2f} dB\nS11: {s11_mae:.2f} dB\nS21: {s21_mae:.2f} dB\n"
            f"Median: {torch.quantile(per_sample,.5):.2f}\n95th: {torch.quantile(per_sample,.95):.2f}\n"
            f"Best Val: {best_val:.2f} dB",
            transform=ax.transAxes, fontsize=13, va='center', ha='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax.axis('off'); ax.set_title("Summary")
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"tuhh_{dataset_type}_results.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Plots saved to {results_dir}/")
    
    return test_mae


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    PROJ = os.path.expanduser(
        "~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects")
    
    configs = [
        {
            'type': 'link',
            'raw_dir': os.path.join(PROJ, "data/raw/Universal-Diff-SI-Link"),
            'proc_dir': os.path.join(PROJ, "data/processed/Universal-Diff-SI-Link"),
            'pt_file': "via_link_dataset.pt",
        },
        {
            'type': 'array',
            'raw_dir': os.path.join(PROJ, "data/raw/Universal-Diff-SI-Array"),
            'proc_dir': os.path.join(PROJ, "data/processed/Universal-Diff-SI-Array"),
            'pt_file': "via_array_dataset.pt",
        },
    ]
    
    results = {}
    
    for cfg in configs:
        dt = cfg['type']
        pt_path = os.path.join(cfg['proc_dir'], cfg['pt_file'])
        
        # PHASE 1: Parse if needed
        if not os.path.exists(pt_path):
            if os.path.exists(cfg['raw_dir']):
                print(f"\nProcessed file not found. Parsing raw {dt} data...")
                pt_path = parse_dataset(dt, cfg['raw_dir'], cfg['proc_dir'])
            else:
                print(f"ERROR: Neither processed nor raw data found for {dt}")
                print(f"  Expected processed: {pt_path}")
                print(f"  Expected raw: {cfg['raw_dir']}")
                continue
        else:
            print(f"\nFound processed {dt} data at {pt_path}")
        
        # PHASE 2: Train
        res_dir = os.path.join(PROJ, f"results/forward_model_{dt}")
        mae = train_model(pt_path, dt, res_dir, epochs=600)
        results[dt] = mae
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    for dt, mae in results.items():
        print(f"  {dt.upper():>6}: {mae:.2f} dB MAE")
    print(f"{'='*60}")