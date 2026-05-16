import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skrf as rf
from tqdm import tqdm

# ==========================================
# 1. CRASH-PROOF RAW DATA PARSER
# ==========================================
def build_universal_dataset(raw_dir, output_pt_path):
    print("==================================================")
    print(" PHASE 1: RAW DATA PARSING (DIRECT S-PARAMETERS)")
    print("==================================================")
    
    csv_path = os.path.join(raw_dir, "parameter.csv")
    variation_dir = os.path.join(raw_dir, "variation")
    
    df = pd.read_csv(csv_path)
    if 'LOSTANGENT' in df.columns: df.rename(columns={'LOSTANGENT':'LOSSTANGENT'}, inplace=True)
    
    global_cols = ['LAYER_AMOUNT', 'PERMITTIVITY', 'LOSSTANGENT', 'TDIEL']
    local_cols = ['VIA_RADIUS', 'ANTIPAD_RADIUS', 'PITCH', 'SL_WIDTH', 'LENGTH']
    features_global = df[global_cols].copy()
    features_local = df[local_cols].copy()
    
    # Stabilize the loss tangent
    features_global['LOSSTANGENT'] = np.log10(features_global['LOSSTANGENT'].clip(lower=1e-6))
    
    sim_ids = df['SIMULATION'].values
    std_rf_freq = rf.Frequency(start=0.25, stop=100, npoints=401, unit='ghz')
    
    valid_indices = []
    Y_raw_list = []
    
    print(f"Extracting S-Parameters directly from {len(sim_ids)} simulations...")
    
    for idx, sim_id in tqdm(enumerate(sim_ids), total=len(sim_ids)):
        search_pattern = os.path.join(variation_dir, str(sim_id), "*.s*p")
        files = glob.glob(search_pattern)
        if not files: continue
        
        try:
            network = rf.Network(files[0]).interpolate(std_rf_freq, bounds_error=False, fill_value="extrapolate")
            num_ports = network.s.shape[1]
            if num_ports < 4: continue
            
            target_link = (num_ports // 4) // 2 
            tx_plus = target_link * 2
            tx_minus = tx_plus + 1
            rx_plus = (num_ports // 2) + tx_plus
            rx_minus = rx_plus + 1
            port_idx = [tx_plus, tx_minus, rx_plus, rx_minus]
            
            core_s_matrix = network.s[:, port_idx, :][:, :, port_idx]
            
            # Convert 4x4 Single-Ended to Mixed-Mode (Sdd11, Sdd21)
            Sdd11 = 0.5 * (core_s_matrix[:, 0, 0] - core_s_matrix[:, 0, 1] - core_s_matrix[:, 1, 0] + core_s_matrix[:, 1, 1])
            Sdd21 = 0.5 * (core_s_matrix[:, 2, 0] - core_s_matrix[:, 2, 1] - core_s_matrix[:, 3, 0] + core_s_matrix[:, 3, 1])
            
            if np.isnan(Sdd11).any() or np.isnan(Sdd21).any():
                continue
                
            Y_raw = torch.stack([torch.tensor(Sdd11, dtype=torch.complex64), 
                                 torch.tensor(Sdd21, dtype=torch.complex64)], dim=-1)
            
            Y_raw_list.append(Y_raw)
            valid_indices.append(idx)
            
        except Exception:
            continue # Silently skip unreadable files
            
    if not valid_indices:
        raise RuntimeError("CRITICAL ERROR: Failed to parse any files. Check directory paths.")

    # Standardize Features (Z-Score)
    X_global_raw = features_global.values[valid_indices]
    X_local_raw = features_local.values[valid_indices]
    
    num_ports_array = np.full((len(valid_indices), 1), 4.0) 
    X_global_raw = np.hstack((X_global_raw, num_ports_array))
    
    X_global_mean = X_global_raw.mean(axis=0)
    X_global_std = np.where(X_global_raw.std(axis=0) == 0, 1e-5, X_global_raw.std(axis=0))
    X_global_norm = (X_global_raw - X_global_mean) / X_global_std
    
    X_local_mean = X_local_raw.mean(axis=0)
    X_local_std = np.where(X_local_raw.std(axis=0) == 0, 1e-5, X_local_raw.std(axis=0))
    X_local_norm = (X_local_raw - X_local_mean) / X_local_std
    
    X_tensor = torch.cat([torch.tensor(X_global_norm, dtype=torch.float32), 
                          torch.tensor(X_local_norm, dtype=torch.float32)], dim=1)
    
    dataset_dict = {
        'X': X_tensor,
        'Y_raw': torch.stack(Y_raw_list),
        'frequencies': torch.tensor(std_rf_freq.f, dtype=torch.float32)
    }
    
    torch.save(dataset_dict, output_pt_path)
    print(f"\nParsing Complete! {len(valid_indices)} files processed. Saved to {output_pt_path}")
    return dataset_dict

# ==========================================
# 2. DATASET LOADER
# ==========================================
class UniversalDirectDataset(Dataset):
    def __init__(self, data_dict):
        self.X = data_dict['X']
        self.Y_raw = data_dict['Y_raw']
        self.freqs_ghz = data_dict['frequencies'].numpy() / 1e9

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y_raw[idx]

# ==========================================
# 3. UNIVERSAL DEEP RESNET ARCHITECTURE
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim),
            nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim)
        )
    def forward(self, x): return x + self.net(x)

class UniversalDirectNet(nn.Module):
    def __init__(self, input_dim=11, num_freqs=401, num_targets=2): # Input is 11 (10 geom + 1 port info)
        super().__init__()
        self.num_freqs = num_freqs
        self.num_targets = num_targets
        
        # 401 freqs * 2 targets (S11, S21) * 2 (Real, Imag) = 1604 outputs
        output_dim = num_freqs * num_targets * 2
        
        # Deep backbone to handle the variance of 4 to 48 layer boards
        hidden = 1024
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden),
            ResBlock(hidden), ResBlock(hidden), ResBlock(hidden), ResBlock(hidden),
            nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        batch = x.shape[0]
        raw_out = self.backbone(x)
        raw_out = raw_out.view(batch, self.num_freqs, self.num_targets, 2)
        
        H_s = torch.complex(raw_out[..., 0], raw_out[..., 1])
        return H_s

# ==========================================
# 4. MASTER TRAINING LOOP WITH DB LOSS FIX
# ==========================================
def main():
    results_dir = "universal_direct_corrected_results"
    os.makedirs(results_dir, exist_ok=True)
    
    raw_dir = os.path.expanduser("~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/data/raw/Universal-Diff-SI-Link")
    dataset_path = os.path.join(results_dir, "universal_direct_dataset.pt")
    
    # Parse Data (No Vector Fitting crashes)
    if not os.path.exists(dataset_path):
        data_dict = build_universal_dataset(raw_dir, dataset_path)
    else:
        print(f"Loading cached dataset from {dataset_path}")
        data_dict = torch.load(dataset_path)
        
    ds = UniversalDirectDataset(data_dict)
    input_dim = ds.X.shape[1] # Automatically detect 11 features
    
    train_set, val_set = random_split(ds, [int(0.85*len(ds)), len(ds)-int(0.85*len(ds))])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniversalDirectNet(input_dim=input_dim).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)

    def dual_objective_loss(pred, target):
        # 1. Complex MSE (Linear) - Anchors phase and main shape
        mse = nn.functional.mse_loss(pred.real, target.real) + \
              nn.functional.mse_loss(pred.imag, target.imag)
        
        # 2. THE FIX: Aggressive dB L1 Loss
        # Clamp to -100 dB (1e-5) so VNA thermal noise doesn't cause infinite gradients
        p_c = torch.clamp(torch.abs(pred), min=1e-5)
        t_c = torch.clamp(torch.abs(target), min=1e-5)
        p_db = 20 * torch.log10(p_c)
        t_db = 20 * torch.log10(t_c)
        
        # Using L1 Loss (MAE) heavily weighted (0.1 -> 1.0 depending on scaling)
        # This forces the optimizer to treat a 10 dB error at -60 dB as seriously as a 10 dB error at 0 dB.
        loss_db = nn.functional.l1_loss(p_db, t_db)
        
        return mse + (0.1 * loss_db)

    print("\n==================================================")
    print(f" PHASE 2: TRAINING CORRECTED RESNET ON {device}")
    print("==================================================")
    
    best_val_mae = float('inf')

    epochs = 500
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x) 
            loss = dual_objective_loss(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        model.eval()
        v_db = 0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                pred = model(xv)
                
                # Strict Evaluation MAE
                p_c = torch.clamp(torch.abs(pred), min=1e-5)
                t_c = torch.clamp(torch.abs(yv), min=1e-5)
                p_m = 20 * torch.log10(p_c)
                t_m = 20 * torch.log10(t_c)
                v_db += torch.abs(p_m - t_m).mean().item()
                
        avg_v_db = v_db / len(val_loader)
        scheduler.step()
        
        if avg_v_db < best_val_mae:
            best_val_mae = avg_v_db
            torch.save(model.state_dict(), os.path.join(results_dir, "best_universal_direct_corrected.pth"))

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d} | Val MAE: {avg_v_db:.3f} dB | Best: {best_val_mae:.3f} dB")

    # ==========================================
    # 5. Proof Plot
    # ==========================================
    model.load_state_dict(torch.load(os.path.join(results_dir, "best_universal_direct_corrected.pth")))
    model.eval()
    
    idx = np.random.randint(len(val_set))
    x_test, y_test = val_set[idx]
    
    with torch.no_grad():
        pred = model(x_test.unsqueeze(0).to(device)).cpu().squeeze(0)
    
    y_db = 20 * np.log10(torch.clamp(torch.abs(y_test[:, 1]), min=1e-5).numpy())
    p_db = 20 * np.log10(torch.clamp(torch.abs(pred[:, 1]), min=1e-5).numpy())

    plt.figure(figsize=(10, 6))
    plt.plot(ds.freqs_ghz, y_db, 'b', label='HFSS True Target (S21)', linewidth=2)
    plt.plot(ds.freqs_ghz, p_db, 'r--', label='Dual-Objective ResNet Prediction', linewidth=2)
    plt.title(f"Corrected Universal Data Verification (Best MAE: {best_val_mae:.3f} dB)")
    plt.ylabel("Magnitude (dB)"); plt.xlabel("Frequency (GHz)")
    plt.grid(True, alpha=0.3); plt.legend()
    
    save_path = os.path.join(results_dir, "universal_direct_corrected_proof.png")
    plt.savefig(save_path)
    print(f"Done! Corrected Universal Model created. Proof saved to {save_path}")

if __name__ == "__main__":
    main()