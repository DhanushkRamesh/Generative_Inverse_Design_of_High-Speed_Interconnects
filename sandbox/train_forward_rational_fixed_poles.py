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
# 2. Hybrid Fixed-Pole Rational Layer
# ==========================================
class HybridRationalModel(nn.Module):
    def __init__(self, input_dim=10, num_targets=2):
        super().__init__()
        self.num_targets = num_targets
        
        # --- THE HYBRID UPGRADE: FIXED BASIS POLES ---
        # We place 40 poles evenly across the 0 to 100 GHz band.
        num_poles = 40
        f_centers = np.linspace(1, 100, num_poles)
        
        # Beta = 2*pi*f (Imaginary part: resonance frequency)
        beta = 2 * np.pi * f_centers
        # Alpha = Negative real part (Damping). 
        # We make them wide enough to overlap and form a continuous basis.
        alpha = -2.0 * np.ones(num_poles) 
        
        # Register as a buffer so it moves to GPU but is NOT updated by the optimizer
        fixed_p = torch.tensor(alpha + 1j * beta, dtype=torch.complex64)
        self.register_buffer('fixed_poles', fixed_p)
        self.register_buffer('fixed_poles_conj', torch.conj(fixed_p))
        
        # The AI only predicts: c_re, c_im (2 per pole) + d_re, d_im (2 total)
        self.params_per_target = (2 * num_poles) + 2
        total_output_dim = self.params_per_target * num_targets
        self.num_poles = num_poles
        
        # High-capacity MLP to map geometry strictly to residues
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, total_output_dim)
        )

    def forward(self, x, s_tensor):
        batch_size = x.shape[0]
        num_freqs = s_tensor.shape[0]
        
        # 1. Predict Residues and Direct Term
        raw_out = self.mlp(x)
        raw_out = raw_out.view(batch_size, self.num_targets, self.params_per_target)
        
        c_re = raw_out[:, :, 0 : self.num_poles]
        c_im = raw_out[:, :, self.num_poles : 2*self.num_poles]
        d_re = raw_out[:, :, -2].unsqueeze(-1)
        d_im = raw_out[:, :, -1].unsqueeze(-1)
        
        # 2. Construct Complex Tensors
        c = torch.complex(c_re, c_im)
        d = torch.complex(d_re, d_im)
        c_conj = torch.conj(c)
        
        # 3. Align shapes for broadcasting
        # c: (Batch, Targets, Poles, 1)
        c = c.unsqueeze(-1)
        c_conj = c_conj.unsqueeze(-1)
        
        # fixed_poles: (1, 1, Poles, 1)
        p = self.fixed_poles.view(1, 1, self.num_poles, 1)
        p_conj = self.fixed_poles_conj.view(1, 1, self.num_poles, 1)
        
        # s_tensor: (1, 1, 1, Freqs)
        s_view = s_tensor.view(1, 1, 1, num_freqs)
        
        # 4. Compute Transfer Function with Fixed Denominators
        term1 = c / (s_view - p)
        term2 = c_conj / (s_view - p_conj)
        
        # Sum across poles and add direct term
        H_s = torch.sum(term1 + term2, dim=2) + d
        return H_s.transpose(1, 2)

# ==========================================
# 3. Training Logic
# ==========================================
def main():
    results_dir = "forward_hybrid_results"
    os.makedirs(results_dir, exist_ok=True)
    
    data_path = os.path.expanduser("~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/data/processed/Synthetic-Link/synthetic_poles_dataset.pt")
    ds = SyntheticPoleDataset(data_path)
    train_set, val_set = random_split(ds, [int(0.8*len(ds)), len(ds)-int(0.8*len(ds))])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridRationalModel(input_dim=10, num_targets=2).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)

    f_ghz = torch.tensor(ds.freqs_ghz, dtype=torch.float32, device=device)
    s_tensor = 1j * 2 * math.pi * f_ghz

    def hybrid_loss(pred, target):
        mse_real = nn.functional.mse_loss(pred.real, target.real)
        mse_imag = nn.functional.mse_loss(pred.imag, target.imag)
        
        p_mag = torch.sqrt(pred.real**2 + pred.imag**2 + 1e-12)
        t_mag = torch.sqrt(target.real**2 + target.imag**2 + 1e-12)
        db_mae = torch.abs(20*torch.log10(p_mag) - 20*torch.log10(t_mag)).mean()
        
        return mse_real + mse_imag + (0.05 * db_mae) # Increased dB weight slightly

    print(f"Training Hybrid Fixed-Pole Model on {device}...")
    history = {'train': [], 'val_db': []}

    epochs = 400
    for epoch in range(epochs):
        model.train()
        t_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, s_tensor) 
            loss = hybrid_loss(out, y)
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
            print(f"Epoch {epoch+1:03d} | Train Loss: {avg_loss:.6f} | Val MAE: {avg_v_db:.2f} dB")

    # ==========================================
    # 4. Verification Plot
    # ==========================================
    model.eval()
    idx = np.random.randint(len(val_set))
    x_test, y_test = val_set[idx]
    with torch.no_grad():
        pred = model(x_test.to(device).unsqueeze(0), s_tensor).cpu().squeeze()
    
    y_db = 20 * np.log10(torch.abs(y_test[:, 1]).numpy() + 1e-12)
    p_db = 20 * np.log10(torch.abs(pred[:, 1]).numpy() + 1e-12)

    plt.figure(figsize=(10, 6))
    plt.plot(ds.freqs_ghz, y_db, 'b', label='Synthetic Truth', linewidth=2)
    plt.plot(ds.freqs_ghz, p_db, 'r--', label='Hybrid Fixed-Pole Predict', linewidth=2)
    plt.title(f"Hybrid Rational Layer Verification (Final MAE: {history['val_db'][-1]:.2f} dB)")
    plt.ylabel("Magnitude (dB)"); plt.xlabel("Frequency (GHz)")
    plt.grid(True, alpha=0.3); plt.legend()
    
    save_path = os.path.join(results_dir, "hybrid_synthetic_proof.png")
    plt.savefig(save_path)
    print(f"Proof saved to {save_path}")

if __name__ == "__main__":
    main()