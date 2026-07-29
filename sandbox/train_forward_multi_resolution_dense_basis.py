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
# 2. Multi-Resolution Dense Basis (MRDB) Net
# ==========================================
class MRDBRationalModel(nn.Module):
    def __init__(self, input_dim=10, num_targets=2):
        super().__init__()
        self.num_targets = num_targets
        
        # --- THE SOTA UPGRADE: MULTI-RESOLUTION GRID ---
        # 100 frequencies spread across the band
        freqs = np.linspace(0.5, 100, 100)
        # 4 different Q-factors (from very sharp to very wide)
        alphas = [-0.05, -0.5, -1.5, -3.0] 
        
        poles = []
        for a in alphas:
            for f in freqs:
                poles.append(complex(a, 2 * np.pi * f))
                
        self.num_poles = len(poles) # 400 fixed poles
        
        fixed_p = torch.tensor(poles, dtype=torch.complex64)
        self.register_buffer('fixed_poles', fixed_p)
        self.register_buffer('fixed_poles_conj', torch.conj(fixed_p))
        
        # Output: c_re, c_im for 400 poles (800) + direct terms d_re, d_im (2)
        self.params_per_target = (2 * self.num_poles) + 2
        total_output_dim = self.params_per_target * num_targets
        
        # High-capacity MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, total_output_dim)
        )

    def forward(self, x, s_tensor):
        batch = x.shape[0]
        freqs = s_tensor.shape[0]
        
        raw_out = self.mlp(x).view(batch, self.num_targets, self.params_per_target)
        
        # Extract Residues (We scale them down initially for stability)
        c_re = raw_out[:, :, 0 : self.num_poles] * 0.1
        c_im = raw_out[:, :, self.num_poles : 2*self.num_poles] * 0.1
        d_re = raw_out[:, :, -2].unsqueeze(-1)
        d_im = raw_out[:, :, -1].unsqueeze(-1)
        
        c = torch.complex(c_re, c_im).unsqueeze(-1)
        d = torch.complex(d_re, d_im)
        c_conj = torch.conj(c)
        
        p = self.fixed_poles.view(1, 1, self.num_poles, 1)
        p_conj = self.fixed_poles_conj.view(1, 1, self.num_poles, 1)
        
        s_view = s_tensor.view(1, 1, 1, freqs)
        
        # Vectorized Rational Function
        term1 = c / (s_view - p)
        term2 = c_conj / (s_view - p_conj)
        
        H_s = torch.sum(term1 + term2, dim=2) + d
        return H_s.transpose(1, 2)

# ==========================================
# 3. Log-Aware Training Loop
# ==========================================
def main():
    results_dir = "forward_mrdb_results"
    os.makedirs(results_dir, exist_ok=True)
    
    data_path = os.path.expanduser("~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/data/processed/Synthetic-Link/synthetic_poles_dataset.pt")
    ds = SyntheticPoleDataset(data_path)
    train_set, val_set = random_split(ds, [int(0.8*len(ds)), len(ds)-int(0.8*len(ds))])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MRDBRationalModel(input_dim=10, num_targets=2).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)

    f_ghz = torch.tensor(ds.freqs_ghz, dtype=torch.float32, device=device)
    s_tensor = 1j * 2 * math.pi * f_ghz

    # The SOTA Loss Function
    def log_aware_loss(pred, target):
        # 1. Standard Complex MSE (keeps general phase/shape correct)
        mse = nn.functional.mse_loss(pred.real, target.real) + \
              nn.functional.mse_loss(pred.imag, target.imag)
        
        # 2. Log-Magnitude MSE (Forces exact matching of deep notches)
        log_pred = torch.log10(torch.abs(pred) + 1e-12)
        log_target = torch.log10(torch.abs(target) + 1e-12)
        log_mse = nn.functional.mse_loss(log_pred, log_target)
        
        # Blend them. The log_mse ensures the notches drop down completely.
        return mse + (0.5 * log_mse)

    print(f"Training MRDB Rational Model on {device}...")
    history = {'train': [], 'val_db': []}

    epochs = 400
    for epoch in range(epochs):
        model.train()
        t_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, s_tensor) 
            loss = log_aware_loss(out, y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        
        model.eval()
        v_db = 0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                pred = model(xv, s_tensor)
                p_m = torch.sqrt(pred.real**2 + pred.imag**2 + 1e-12)
                t_m = torch.sqrt(yv.real**2 + yv.imag**2 + 1e-12)
                v_db += torch.abs(20*torch.log10(p_m) - 20*torch.log10(t_m)).mean().item()
        
        avg_loss = t_loss/len(train_loader)
        avg_v_db = v_db/len(val_loader)
        history['train'].append(avg_loss)
        history['val_db'].append(avg_v_db)
        scheduler.step(avg_loss)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d} | Log-Aware Loss: {avg_loss:.6f} | Val MAE: {avg_v_db:.2f} dB")

    # ==========================================
    # 4. Final Verification
    # ==========================================
    model.eval()
    idx = np.random.randint(len(val_set))
    x_test, y_test = val_set[idx]
    with torch.no_grad():
        pred = model(x_test.to(device).unsqueeze(0), s_tensor).cpu().squeeze()
    
    y_db = 20 * np.log10(torch.abs(y_test[:, 1]).numpy() + 1e-12)
    p_db = 20 * np.log10(torch.abs(pred[:, 1]).numpy() + 1e-12)

    plt.figure(figsize=(10, 6))
    plt.plot(ds.freqs_ghz, y_db, 'b', label='Synthetic Target', linewidth=2)
    plt.plot(ds.freqs_ghz, p_db, 'r--', label='MRDB Rational Net', linewidth=2)
    plt.title(f"SOTA MRDB Verification (Final MAE: {history['val_db'][-1]:.2f} dB)")
    plt.ylabel("Magnitude (dB)"); plt.xlabel("Frequency (GHz)")
    plt.grid(True, alpha=0.3); plt.legend()
    
    save_path = os.path.join(results_dir, "mrdb_proof.png")
    plt.savefig(save_path)
    print(f"Success! Proof saved to {save_path}")

if __name__ == "__main__":
    main()