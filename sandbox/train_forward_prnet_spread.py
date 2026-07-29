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
# 2. PRNet with Spreading Initialization
# ==========================================
class SpreadPRNet(nn.Module):
    def __init__(self, input_dim=10, num_poles=5, num_targets=2):
        super().__init__()
        self.num_targets = num_targets
        self.num_poles = num_poles
        
        # 4 params per pole (alpha, f_res, c_re, c_im) + 2 direct terms
        self.params_per_target = (4 * num_poles) + 2
        total_out = self.params_per_target * num_targets
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, total_out)
        )
        
        # --- THE SECRET SAUCE: POLE SPREADING INIT ---
        # Force the network to start with 5 notches spread from 10 to 90 GHz
        # inverse_sigmoid(y) = log(y / (1-y)). We want y = [0.1, 0.3, 0.5, 0.7, 0.9]
        target_f = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        inv_sig = torch.log(target_f / (1.0 - target_f))
        
        with torch.no_grad():
            final_layer = self.mlp[-1]
            
            # S11 Target (Indices 0 to 21)
            final_layer.bias[num_poles : 2*num_poles] = inv_sig # Set f_res
            final_layer.bias[2*num_poles : 4*num_poles] = 1.0   # Force non-zero residues
            
            # S21 Target (Indices 22 to 43)
            offset = self.params_per_target
            final_layer.bias[offset + num_poles : offset + 2*num_poles] = inv_sig
            final_layer.bias[offset + 2*num_poles : offset + 4*num_poles] = 1.0

    def forward(self, x, s_tensor):
        batch = x.shape[0]
        freqs = s_tensor.shape[0]
        
        raw_out = self.mlp(x).view(batch, self.num_targets, self.params_per_target)
        
        # 1. Alpha (Damping): Bounded [-3.05, -0.05]
        alpha = - (torch.sigmoid(raw_out[:, :, 0:self.num_poles]) * 3.0 + 0.05)
        
        # 2. f_res: Bounded [0, 100 GHz]
        f_res = torch.sigmoid(raw_out[:, :, self.num_poles : 2*self.num_poles]) * 100.0
        beta = 2 * math.pi * f_res 
        
        # 3. Residues: Bounded [-20, 20]
        c_re = (torch.sigmoid(raw_out[:, :, 2*self.num_poles : 3*self.num_poles]) - 0.5) * 40.0
        c_im = (torch.sigmoid(raw_out[:, :, 3*self.num_poles : 4*self.num_poles]) - 0.5) * 40.0
        
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
# 3. dB-Space Training Loop
# ==========================================
def main():
    results_dir = "forward_prnet_spread_results"
    os.makedirs(results_dir, exist_ok=True)
    
    data_path = os.path.expanduser("~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/data/processed/Synthetic-Link/synthetic_poles_dataset.pt")
    ds = SyntheticPoleDataset(data_path)
    train_set, val_set = random_split(ds, [int(0.8*len(ds)), len(ds)-int(0.8*len(ds))])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpreadPRNet(input_dim=10, num_poles=5, num_targets=2).to(device)
    
    # Lower learning rate so the poles don't "jump" out of position
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)

    f_ghz = torch.tensor(ds.freqs_ghz, dtype=torch.float32, device=device)
    s_tensor = 1j * 2 * math.pi * f_ghz

    # Pure Magnitude-dB Loss + Phase Loss
    def pure_db_loss(pred, target):
        p_mag_db = 20 * torch.log10(torch.abs(pred) + 1e-9)
        t_mag_db = 20 * torch.log10(torch.abs(target) + 1e-9)
        
        # We use SmoothL1Loss (Huber) because it prevents gradients from exploding 
        # when a pole is slightly misaligned.
        loss_db = nn.functional.smooth_l1_loss(p_mag_db, t_mag_db)
        return loss_db

    print(f"Training Spread PRNet on {device}...")
    history = {'train': [], 'val_db': []}

    epochs = 400
    for epoch in range(epochs):
        model.train()
        t_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, s_tensor) 
            loss = pure_db_loss(out, y)
            loss.backward()
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
        scheduler.step(avg_loss)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d} | dB Huber Loss: {avg_loss:.6f} | Val MAE: {avg_v_db:.2f} dB")

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
    plt.plot(ds.freqs_ghz, p_db, 'r--', label='Spread PRNet', linewidth=2)
    plt.title(f"Pole Spreading Initialization (Final MAE: {history['val_db'][-1]:.2f} dB)")
    plt.ylabel("Magnitude (dB)"); plt.xlabel("Frequency (GHz)")
    plt.grid(True, alpha=0.3); plt.legend()
    
    save_path = os.path.join(results_dir, "spread_prnet_proof.png")
    plt.savefig(save_path)
    print(f"Success! Proof saved to {save_path}")

if __name__ == "__main__":
    main()