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
# 2. Propagation-Augmented PRNet
# ==========================================
class AugmentedPRNet(nn.Module):
    def __init__(self, input_dim=10, num_poles=10, num_targets=2):
        super().__init__()
        self.num_targets = num_targets
        self.num_poles = num_poles
        
        self.params_per_target = (4 * num_poles) + 2
        # TOTAL OUTPUT = (Params per target * 2 targets) + 1 Exponential Decay parameter (Gamma)
        total_out = (self.params_per_target * num_targets) + 1 
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, total_out)
        )
        
        # --- POLE SPREADING INIT ---
        target_f = torch.linspace(0.1, 0.9, num_poles)
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
            # Initialize Gamma close to 0
            final_layer.bias[-1] = 0.0

    def forward(self, x, s_tensor, freqs_ghz):
        batch = x.shape[0]
        num_freqs = s_tensor.shape[0]
        
        raw_out = self.mlp(x)
        
        # Split Rational Parameters from the Exponential Parameter
        pr_params = raw_out[:, :-1].view(batch, self.num_targets, self.params_per_target)
        gamma_raw = raw_out[:, -1]
        
        # 1. Physics Bounding for Poles
        alpha = - (torch.sigmoid(pr_params[:, :, 0:self.num_poles]) * 3.0 + 0.05)
        f_res = torch.sigmoid(pr_params[:, :, self.num_poles : 2*self.num_poles]) * 100.0
        beta = 2 * math.pi * f_res 
        c_re = (torch.sigmoid(pr_params[:, :, 2*self.num_poles : 3*self.num_poles]) - 0.5) * 40.0
        c_im = (torch.sigmoid(pr_params[:, :, 3*self.num_poles : 4*self.num_poles]) - 0.5) * 40.0
        d_re = pr_params[:, :, -2].unsqueeze(-1)
        d_im = pr_params[:, :, -1].unsqueeze(-1)
        
        p = torch.complex(alpha, beta).unsqueeze(-1)
        c = torch.complex(c_re, c_im).unsqueeze(-1)
        d = torch.complex(d_re, d_im)
        s_view = s_tensor.view(1, 1, 1, num_freqs)
        
        # Calculate standard Rational Function
        term1 = c / (s_view - p)
        term2 = torch.conj(c) / (s_view - torch.conj(p))
        H_s = torch.sum(term1 + term2, dim=2) + d # Shape: (Batch, Targets, Freqs)
        
        # --- THE EXPONENTIAL AUGMENTATION (TRANSMISSION LINE PHYSICS) ---
        gamma = torch.nn.functional.softplus(gamma_raw).unsqueeze(-1) # Shape: (Batch, 1)
        f_tensor = freqs_ghz.view(1, num_freqs) # Shape: (1, Freqs)
        
        exp_decay = torch.exp(-gamma * f_tensor).to(dtype=torch.complex64) # Shape: (Batch, Freqs)
        
        # --- THE BUG FIX: OUT-OF-PLACE STACKING ---
        H_s11 = H_s[:, 0, :] # Original S11
        H_s21 = H_s[:, 1, :] * exp_decay # Exponentially decayed S21
        
        # Stack them into a brand new tensor so autograd doesn't crash
        H_out = torch.stack([H_s11, H_s21], dim=1) # Shape: (Batch, Targets, Freqs)
        
        return H_out.transpose(1, 2)

# ==========================================
# 3. Precision Training Loop
# ==========================================
def main():
    results_dir = "forward_prnet_augmented_results"
    os.makedirs(results_dir, exist_ok=True)
    
    data_path = os.path.expanduser("~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/data/processed/Synthetic-Link/synthetic_poles_dataset.pt")
    ds = SyntheticPoleDataset(data_path)
    train_set, val_set = random_split(ds, [int(0.85*len(ds)), len(ds)-int(0.85*len(ds))])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AugmentedPRNet(input_dim=10, num_poles=10, num_targets=2).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-6)

    f_ghz = torch.tensor(ds.freqs_ghz, dtype=torch.float32, device=device)
    s_tensor = 1j * 2 * math.pi * f_ghz

    def strict_loss(pred, target):
        mse_c = nn.functional.mse_loss(pred.real, target.real) + \
                nn.functional.mse_loss(pred.imag, target.imag)
        p_db = 20 * torch.log10(torch.abs(pred) + 1e-9)
        t_db = 20 * torch.log10(torch.abs(target) + 1e-9)
        db_l1 = nn.functional.l1_loss(p_db, t_db)
        return mse_c + (0.05 * db_l1)

    print(f"Training Augmented PRNet on {device}...")
    history = {'train': [], 'val_db': []}

    epochs = 400
    for epoch in range(epochs):
        model.train()
        t_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, s_tensor, f_ghz) 
            loss = strict_loss(out, y)
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

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d} | Strict Loss: {avg_loss:.6f} | Val MAE: {avg_v_db:.2f} dB")

    # ==========================================
    # 4. Final Verification
    # ==========================================
    model.eval()
    idx = np.random.randint(len(val_set))
    x_test, y_test = val_set[idx]
    with torch.no_grad():
        pred = model(x_test.to(device).unsqueeze(0), s_tensor, f_ghz).cpu().squeeze()
    
    y_db = 20 * np.log10(torch.abs(y_test[:, 1]).numpy() + 1e-9)
    p_db = 20 * np.log10(torch.abs(pred[:, 1]).numpy() + 1e-9)

    plt.figure(figsize=(10, 6))
    plt.plot(ds.freqs_ghz, y_db, 'b', label='Synthetic Target', linewidth=2)
    plt.plot(ds.freqs_ghz, p_db, 'r--', label='Augmented PRNet', linewidth=2)
    plt.title(f"Augmented PRNet Verification (Final MAE: {history['val_db'][-1]:.2f} dB)")
    plt.ylabel("Magnitude (dB)"); plt.xlabel("Frequency (GHz)")
    plt.grid(True, alpha=0.3); plt.legend()
    
    save_path = os.path.join(results_dir, "augmented_prnet_proof.png")
    plt.savefig(save_path)
    print(f"Success! Proof saved to {save_path}")

if __name__ == "__main__":
    main()