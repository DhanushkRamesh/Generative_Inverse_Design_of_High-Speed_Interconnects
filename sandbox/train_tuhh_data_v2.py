import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. Binned REAL Dataset Loader (Thin Boards)
# ==========================================
class BinnedViaDataset(Dataset):
    def __init__(self, pt_file_path, max_layers=16):
        data = torch.load(pt_file_path)
        X_full = torch.cat([data['X_global'], data['X_local']], dim=1)
        
        # S11 and S21
        s11 = torch.complex(data['Y_real'][:, :, 0, 0], data['Y_imag'][:, :, 0, 0])
        s21 = torch.complex(data['Y_real'][:, :, 1, 0], data['Y_imag'][:, :, 1, 0])
        Y_full = torch.stack([s11, s21], dim=-1)
        
        # Un-normalize LAYER_AMOUNT to filter
        layer_mean = data['X_global_mean'][0].item()
        layer_std = data['X_global_std'][0].item()
        unnormalized_layers = (X_full[:, 0] * layer_std) + layer_mean
        
        mask = torch.round(unnormalized_layers) <= max_layers
        
        self.X = X_full[mask]
        self.Y = Y_full[mask]
        self.freqs_ghz = data['frequencies'].numpy() / 1e9
        print(f"Binned Dataset: Kept {len(self.X)} out of {len(X_full)} samples (<= {max_layers} layers).")

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

# ==========================================
# 2. Pure Spread PRNet (20 Poles, NO Exponential Bias)
# ==========================================
class SpreadPRNet(nn.Module):
    def __init__(self, input_dim=10, num_poles=20, num_targets=2):
        super().__init__()
        self.num_targets = num_targets
        self.num_poles = num_poles
        
        self.params_per_target = (4 * num_poles) + 2
        total_out = self.params_per_target * num_targets
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, total_out)
        )
        
        # Spread 20 poles evenly
        target_f = torch.linspace(0.05, 0.95, num_poles)
        inv_sig = torch.log(target_f / (1.0 - target_f))
        
        with torch.no_grad():
            final_layer = self.mlp[-1]
            # S11 Init
            final_layer.bias[num_poles : 2*num_poles] = inv_sig 
            final_layer.bias[2*num_poles : 4*num_poles] = 1.0   
            # S21 Init
            offset = self.params_per_target
            final_layer.bias[offset + num_poles : offset + 2*num_poles] = inv_sig
            final_layer.bias[offset + 2*num_poles : offset + 4*num_poles] = 1.0

    def forward(self, x, s_tensor):
        batch = x.shape[0]
        freqs = s_tensor.shape[0]
        
        raw_out = self.mlp(x).view(batch, self.num_targets, self.params_per_target)
        
        alpha = - (torch.sigmoid(raw_out[:, :, 0:self.num_poles]) * 3.0 + 0.05)
        f_res = torch.sigmoid(raw_out[:, :, self.num_poles : 2*self.num_poles]) * 100.0
        beta = 2 * math.pi * f_res 
        c_re = (torch.sigmoid(raw_out[:, :, 2*self.num_poles : 3*self.num_poles]) - 0.5) * 80.0
        c_im = (torch.sigmoid(raw_out[:, :, 3*self.num_poles : 4*self.num_poles]) - 0.5) * 80.0
        
        d_re = raw_out[:, :, -2].unsqueeze(-1)
        d_im = raw_out[:, :, -1].unsqueeze(-1)
        
        p = torch.complex(alpha, beta).unsqueeze(-1)
        c = torch.complex(c_re, c_im).unsqueeze(-1)
        d = torch.complex(d_re, d_im)
        
        s_view = s_tensor.view(1, 1, 1, freqs)
        
        term1 = c / (s_view - p)
        term2 = torch.conj(c) / (s_view - torch.conj(p))
        
        H_s = torch.sum(term1 + term2, dim=2) + d
        return H_s.transpose(1, 2)

# ==========================================
# 3. Noise Floor Training Loop
# ==========================================
def main():
    results_dir = "forward_real_binned_spread_results"
    os.makedirs(results_dir, exist_ok=True)
    
    data_path = os.path.expanduser("~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/data/processed/Universal-Diff-SI-Link/via_link_dataset.pt")
    ds = BinnedViaDataset(data_path, max_layers=16)
    
    if len(ds) < 50:
        print("CRITICAL: Not enough samples after binning.")
        return

    train_set, val_set = random_split(ds, [int(0.85*len(ds)), len(ds)-int(0.85*len(ds))])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpreadPRNet(input_dim=10, num_poles=20, num_targets=2).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=600, eta_min=1e-6)

    f_ghz = torch.tensor(ds.freqs_ghz, dtype=torch.float32, device=device)
    s_tensor = 1j * 2 * math.pi * f_ghz

    def noise_floor_loss(pred, target):
        mse_c = nn.functional.mse_loss(pred.real, target.real) + \
                nn.functional.mse_loss(pred.imag, target.imag)
        
        # -100 dB clamp
        p_mag_clamped = torch.clamp(torch.abs(pred), min=1e-5)
        t_mag_clamped = torch.clamp(torch.abs(target), min=1e-5)
        
        p_db = 20 * torch.log10(p_mag_clamped)
        t_db = 20 * torch.log10(t_mag_clamped)
        
        db_smooth = nn.functional.smooth_l1_loss(p_db, t_db)
        return mse_c + (0.05 * db_smooth)

    print(f"Training PURE RATIONAL Spread PRNet on BINNED TUHH DATA ({device})...")
    history = {'train': [], 'val_db': []}
    best_val_mae = float('inf')

    epochs = 600
    for epoch in range(epochs):
        model.train()
        t_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, s_tensor) 
            loss = noise_floor_loss(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item()
        
        model.eval()
        v_db = 0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                pred = model(xv, s_tensor)
                
                p_clamped = torch.clamp(torch.abs(pred), min=1e-5)
                t_clamped = torch.clamp(torch.abs(yv), min=1e-5)
                
                p_m = 20 * torch.log10(p_clamped)
                t_m = 20 * torch.log10(t_clamped)
                v_db += torch.abs(p_m - t_m).mean().item()
        
        avg_loss = t_loss/len(train_loader)
        avg_v_db = v_db/len(val_loader)
        history['train'].append(avg_loss)
        history['val_db'].append(avg_v_db)
        scheduler.step()
        
        if avg_v_db < best_val_mae:
            best_val_mae = avg_v_db
            torch.save(model.state_dict(), os.path.join(results_dir, "best_binned_spread_prnet.pth"))

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.6f} | Val MAE: {avg_v_db:.2f} dB | Best: {best_val_mae:.2f} dB")

    # ==========================================
    # 4. Final Verification
    # ==========================================
    model.load_state_dict(torch.load(os.path.join(results_dir, "best_binned_spread_prnet.pth")))
    model.eval()
    
    idx = np.random.randint(len(val_set))
    x_test, y_test = val_set[idx]
    with torch.no_grad():
        pred = model(x_test.to(device).unsqueeze(0), s_tensor).cpu().squeeze()
    
    y_clamped = torch.clamp(torch.abs(y_test[:, 1]), min=1e-5).numpy()
    p_clamped = torch.clamp(torch.abs(pred[:, 1]), min=1e-5).numpy()
    
    y_db = 20 * np.log10(y_clamped)
    p_db = 20 * np.log10(p_clamped)

    plt.figure(figsize=(10, 6))
    plt.plot(ds.freqs_ghz, y_db, 'b', label='HFSS (Thin Board)', linewidth=2)
    plt.plot(ds.freqs_ghz, p_db, 'r--', label='Pure Rational PRNet (20 Poles)', linewidth=2)
    plt.title(f"Thin Board Verification (Best MAE: {best_val_mae:.2f} dB)")
    plt.ylabel("Magnitude (dB)"); plt.xlabel("Frequency (GHz)")
    plt.grid(True, alpha=0.3); plt.legend()
    
    save_path = os.path.join(results_dir, "binned_spread_proof.png")
    plt.savefig(save_path)
    print(f"Success! Best Model saved with MAE: {best_val_mae:.2f} dB. Proof saved to {save_path}")

if __name__ == "__main__":
    main()