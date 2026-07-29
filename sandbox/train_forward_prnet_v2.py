import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. Dataset Loader
# ==========================================
class SyntheticPoleDataset(Dataset):
    def __init__(self, pt_file_path):
        data = torch.load(pt_file_path)
        self.X = torch.cat([data['X_global'], data['X_local']], dim=1)
        
        y_r = data['Y_real'][:, :, :, 0] if data['Y_real'].dim() == 4 else data['Y_real']
        y_i = data['Y_imag'][:, :, :, 0] if data['Y_imag'].dim() == 4 else data['Y_imag']
        self.Y = torch.complex(y_r, y_i)
        
        self.freqs_ghz = data['frequencies'].numpy() / 1e9
        print(f"Loaded {len(self.X)} Samples. Target points: {len(self.freqs_ghz)}")

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

# ==========================================
# 2. Master PRNet (High Capacity)
# ==========================================
class MasterPRNet(nn.Module):
    def __init__(self, input_dim=10, num_poles=15, num_targets=2): # UPGRADED TO 15 POLES
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
        
        # --- 15-POLE SPREADING INIT ---
        # Spreads 15 poles perfectly from ~5 GHz to ~95 GHz
        target_f = torch.linspace(0.05, 0.95, num_poles)
        inv_sig = torch.log(target_f / (1.0 - target_f))
        
        with torch.no_grad():
            final_layer = self.mlp[-1]
            
            # S11 Initialization
            final_layer.bias[num_poles : 2*num_poles] = inv_sig 
            final_layer.bias[2*num_poles : 4*num_poles] = 1.0   
            
            # S21 Initialization
            offset = self.params_per_target
            final_layer.bias[offset + num_poles : offset + 2*num_poles] = inv_sig
            final_layer.bias[offset + 2*num_poles : offset + 4*num_poles] = 1.0

    def forward(self, x, s_tensor):
        batch = x.shape[0]
        freqs = s_tensor.shape[0]
        
        raw_out = self.mlp(x).view(batch, self.num_targets, self.params_per_target)
        
        # 1. Alpha: Allow slightly sharper resonances (bounds [-3.01, -0.01])
        alpha = - (torch.sigmoid(raw_out[:, :, 0:self.num_poles]) * 3.0 + 0.01)
        
        # 2. F_res: Bounded [0, 100 GHz]
        f_res = torch.sigmoid(raw_out[:, :, self.num_poles : 2*self.num_poles]) * 100.0
        beta = 2 * math.pi * f_res 
        
        # 3. Residues: Expanded bounds to fit exponential background
        c_re = (torch.sigmoid(raw_out[:, :, 2*self.num_poles : 3*self.num_poles]) - 0.5) * 80.0
        c_im = (torch.sigmoid(raw_out[:, :, 3*self.num_poles : 4*self.num_poles]) - 0.5) * 80.0
        
        # 4. Direct Term
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
# 3. Precision Training Loop
# ==========================================
def main():
    results_dir = "forward_prnet_master_results"
    os.makedirs(results_dir, exist_ok=True)
    
    data_path = os.path.expanduser("~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/data/processed/Synthetic-Link/synthetic_poles_dataset.pt")
    ds = SyntheticPoleDataset(data_path)
    train_set, val_set = random_split(ds, [int(0.85*len(ds)), len(ds)-int(0.85*len(ds))])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MasterPRNet(input_dim=10, num_poles=15, num_targets=2).to(device)
    
    # Cosine Annealing ensures the learning rate drops smoothly to fine-tune the exact notch depths
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-6)

    f_ghz = torch.tensor(ds.freqs_ghz, dtype=torch.float32, device=device)
    s_tensor = 1j * 2 * math.pi * f_ghz

    # Strict L1 + Complex MSE
    def strict_master_loss(pred, target):
        # Lock in the phase and overall shape
        mse_c = nn.functional.mse_loss(pred.real, target.real) + \
                nn.functional.mse_loss(pred.imag, target.imag)
        
        # Mercilessly punish any dB deviation (strict L1)
        p_db = 20 * torch.log10(torch.abs(pred) + 1e-9)
        t_db = 20 * torch.log10(torch.abs(target) + 1e-9)
        db_l1 = nn.functional.l1_loss(p_db, t_db)
        
        return mse_c + (0.05 * db_l1)

    print(f"Training 15-Pole Master PRNet on {device}...")
    history = {'train': [], 'val_db': []}

    epochs = 400
    for epoch in range(epochs):
        model.train()
        t_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, s_tensor) 
            loss = strict_master_loss(out, y)
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
                p_m = 20 * torch.log10(torch.abs(pred) + 1e-9)
                t_m = 20 * torch.log10(torch.abs(yv) + 1e-9)
                v_db += torch.abs(p_m - t_m).mean().item()
        
        avg_loss = t_loss/len(train_loader)
        avg_v_db = v_db/len(val_loader)
        history['train'].append(avg_loss)
        history['val_db'].append(avg_v_db)
        scheduler.step()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d} | Strict Loss: {avg_loss:.6f} | Val MAE: {avg_v_db:.2f} dB")

    # ==========================================
    # 4. Final Verification
    # ==========================================
    model.eval()
    idx = np.random.randint(len(val_set))
    x_test, y_test = val_set[idx]
    with torch.no_grad():
        pred = model(x_test.to(device).unsqueeze(0), s_tensor).cpu().squeeze()
    
    y_db = 20 * np.log10(torch.abs(y_test[:, 1]).numpy() + 1e-9)
    p_db = 20 * np.log10(torch.abs(pred[:, 1]).numpy() + 1e-9)

    plt.figure(figsize=(10, 6))
    plt.plot(ds.freqs_ghz, y_db, 'b', label='Synthetic Target', linewidth=2)
    plt.plot(ds.freqs_ghz, p_db, 'r--', label='Master PRNet (15 Poles)', linewidth=2)
    plt.title(f"Master PRNet Convergence (Final MAE: {history['val_db'][-1]:.2f} dB)")
    plt.ylabel("Magnitude (dB)"); plt.xlabel("Frequency (GHz)")
    plt.grid(True, alpha=0.3); plt.legend()
    
    save_path = os.path.join(results_dir, "master_prnet_proof.png")
    plt.savefig(save_path)
    print(f"Success! Proof saved to {save_path}")

if __name__ == "__main__":
    main()