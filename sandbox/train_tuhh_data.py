import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. REAL Dataset Loader (TUHH Data)
# ==========================================
class ViaLinkDataset(Dataset):
    def __init__(self, pt_file_path):
        data = torch.load(pt_file_path)
        self.X = torch.cat([data['X_global'], data['X_local']], dim=1)
        
        # Load S11 and S21 to match the 2-Target Architecture
        s11_r = data['Y_real'][:, :, 0, 0]
        s11_i = data['Y_imag'][:, :, 0, 0]
        s21_r = data['Y_real'][:, :, 1, 0]
        s21_i = data['Y_imag'][:, :, 1, 0]
        
        s11 = torch.complex(s11_r, s11_i)
        s21 = torch.complex(s21_r, s21_i)
        
        # Stack targets to shape (Samples, 401, 2)
        self.Y = torch.stack([s11, s21], dim=-1)
        
        self.freqs_ghz = data['frequencies'].numpy() / 1e9
        print(f"Loaded {len(self.X)} REAL TUHH Samples. Target points: {len(self.freqs_ghz)}")

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

# ==========================================
# 2. Proven Augmented PRNet (15 Poles)
# ==========================================
class AugmentedPRNet(nn.Module):
    def __init__(self, input_dim=10, num_poles=15, num_targets=2):
        super().__init__()
        self.num_targets = num_targets
        self.num_poles = num_poles
        
        self.params_per_target = (4 * num_poles) + 2
        total_out = (self.params_per_target * num_targets) + 1 
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, total_out)
        )
        
        target_f = torch.linspace(0.05, 0.95, num_poles)
        inv_sig = torch.log(target_f / (1.0 - target_f))
        
        with torch.no_grad():
            final_layer = self.mlp[-1]
            final_layer.bias[num_poles : 2*num_poles] = inv_sig 
            final_layer.bias[2*num_poles : 4*num_poles] = 1.0   
            offset = self.params_per_target
            final_layer.bias[offset + num_poles : offset + 2*num_poles] = inv_sig
            final_layer.bias[offset + 2*num_poles : offset + 4*num_poles] = 1.0
            final_layer.bias[-1] = 0.0

    def forward(self, x, s_tensor, freqs_ghz):
        batch = x.shape[0]
        num_freqs = s_tensor.shape[0]
        
        raw_out = self.mlp(x)
        pr_params = raw_out[:, :-1].view(batch, self.num_targets, self.params_per_target)
        gamma_raw = raw_out[:, -1]
        
        alpha = - (torch.sigmoid(pr_params[:, :, 0:self.num_poles]) * 3.0 + 0.05)
        f_res = torch.sigmoid(pr_params[:, :, self.num_poles : 2*self.num_poles]) * 100.0
        beta = 2 * math.pi * f_res 
        c_re = (torch.sigmoid(pr_params[:, :, 2*self.num_poles : 3*self.num_poles]) - 0.5) * 80.0
        c_im = (torch.sigmoid(pr_params[:, :, 3*self.num_poles : 4*self.num_poles]) - 0.5) * 80.0
        d_re = pr_params[:, :, -2].unsqueeze(-1)
        d_im = pr_params[:, :, -1].unsqueeze(-1)
        
        p = torch.complex(alpha, beta).unsqueeze(-1)
        c = torch.complex(c_re, c_im).unsqueeze(-1)
        d = torch.complex(d_re, d_im)
        s_view = s_tensor.view(1, 1, 1, num_freqs)
        
        term1 = c / (s_view - p)
        term2 = torch.conj(c) / (s_view - torch.conj(p))
        H_s = torch.sum(term1 + term2, dim=2) + d 
        
        gamma = torch.nn.functional.softplus(gamma_raw).unsqueeze(-1) 
        f_tensor = freqs_ghz.view(1, num_freqs) 
        exp_decay = torch.exp(-gamma * f_tensor).to(dtype=torch.complex64) 
        
        H_s11 = H_s[:, 0, :] 
        H_s21 = H_s[:, 1, :] * exp_decay 
        H_out = torch.stack([H_s11, H_s21], dim=1) 
        
        return H_out.transpose(1, 2)

# ==========================================
# 3. Stable Training Loop
# ==========================================
def main():
    results_dir = "forward_real_augmented_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Pointing to the REAL dataset
    data_path = os.path.expanduser("~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/data/processed/Universal-Diff-SI-Link/via_link_dataset.pt")
    ds = ViaLinkDataset(data_path)
    train_set, val_set = random_split(ds, [int(0.85*len(ds)), len(ds)-int(0.85*len(ds))])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AugmentedPRNet(input_dim=10, num_poles=15, num_targets=2).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=600, eta_min=1e-6)

    f_ghz = torch.tensor(ds.freqs_ghz, dtype=torch.float32, device=device)
    s_tensor = 1j * 2 * math.pi * f_ghz

    def stable_loss(pred, target):
        mse_c = nn.functional.mse_loss(pred.real, target.real) + \
                nn.functional.mse_loss(pred.imag, target.imag)
        p_db = 20 * torch.log10(torch.abs(pred) + 1e-9)
        t_db = 20 * torch.log10(torch.abs(target) + 1e-9)
        db_smooth = nn.functional.smooth_l1_loss(p_db, t_db)
        return mse_c + (0.05 * db_smooth)

    print(f"Training Proven PRNet on REAL TUHH DATA ({device})...")
    history = {'train': [], 'val_db': []}
    best_val_mae = float('inf')

    epochs = 600
    for epoch in range(epochs):
        model.train()
        t_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, s_tensor, f_ghz) 
            loss = stable_loss(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item()
        
        model.eval()
        v_db = 0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                pred = model(xv, s_tensor, f_ghz)
                p_m = 20 * torch.log10(torch.abs(pred) + 1e-9)
                t_m = 20 * torch.log10(torch.abs(yv) + 1e-9)
                v_db += torch.abs(p_m - t_m).mean().item()
        
        avg_loss = t_loss/len(train_loader)
        avg_v_db = v_db/len(val_loader)
        history['train'].append(avg_loss)
        history['val_db'].append(avg_v_db)
        scheduler.step()
        
        if avg_v_db < best_val_mae:
            best_val_mae = avg_v_db
            torch.save(model.state_dict(), os.path.join(results_dir, "best_real_prnet.pth"))

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.6f} | Val MAE: {avg_v_db:.2f} dB | Best: {best_val_mae:.2f} dB")

    # ==========================================
    # 4. Final Verification
    # ==========================================
    model.load_state_dict(torch.load(os.path.join(results_dir, "best_real_prnet.pth")))
    model.eval()
    
    idx = np.random.randint(len(val_set))
    x_test, y_test = val_set[idx]
    with torch.no_grad():
        pred = model(x_test.to(device).unsqueeze(0), s_tensor, f_ghz).cpu().squeeze()
    
    y_db = 20 * np.log10(torch.abs(y_test[:, 1]).numpy() + 1e-9)
    p_db = 20 * np.log10(torch.abs(pred[:, 1]).numpy() + 1e-9)

    plt.figure(figsize=(10, 6))
    plt.plot(ds.freqs_ghz, y_db, 'b', label='HFSS Real Target', linewidth=2)
    plt.plot(ds.freqs_ghz, p_db, 'r--', label='PRNet Prediction', linewidth=2)
    plt.title(f"Real Data Verification (Best MAE: {best_val_mae:.2f} dB)")
    plt.ylabel("Magnitude (dB)"); plt.xlabel("Frequency (GHz)")
    plt.grid(True, alpha=0.3); plt.legend()
    
    save_path = os.path.join(results_dir, "real_data_proof.png")
    plt.savefig(save_path)
    print(f"Success! Best Model saved with MAE: {best_val_mae:.2f} dB. Proof saved to {save_path}")

if __name__ == "__main__":
    main()