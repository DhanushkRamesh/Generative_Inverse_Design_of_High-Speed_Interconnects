import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. Dataset Loader (Synthetic Data)
# ==========================================
class SyntheticPoleDataset(Dataset):
    def __init__(self, pt_file_path):
        data = torch.load(pt_file_path)
        self.X = torch.cat([data['X_global'], data['X_local']], dim=1)
        
        # Slicing the dummy dimensions to get back to (Samples, 401, 2)
        y_r = data['Y_real'][:, :, :, 0] if data['Y_real'].dim() == 4 else data['Y_real']
        y_i = data['Y_imag'][:, :, :, 0] if data['Y_imag'].dim() == 4 else data['Y_imag']
        
        # Targets as complex tensors
        self.Y = torch.complex(y_r, y_i) # Shape: (Samples, 401, 2)
        
        self.freqs_ghz = data['frequencies'].numpy() / 1e9
        print(f"Loaded {len(self.X)} Synthetic Samples. Freqs: {len(self.freqs_ghz)}")

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

# ==========================================
# 2. The Physics-Informed Rational Layer
# ==========================================
class RationalForwardModel(nn.Module):
    def __init__(self, input_dim=10, num_poles=10, num_targets=2):
        super().__init__()
        self.num_poles = num_poles
        self.num_targets = num_targets 
        
        # alpha, beta, c_re, c_im (4 parameters per pole)
        # d_re, d_im (2 parameters for direct term)
        self.params_per_target = (4 * num_poles) + 2
        total_output_dim = self.params_per_target * num_targets
        
        # The MLP "Brain"
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
        
        # 1. Predict Raw Parameters
        raw_out = self.mlp(x)
        raw_out = raw_out.view(batch_size, self.num_targets, self.params_per_target)
        
        # 2. Extract and Enforce Physics Constraints
        # Softplus ensures positive, negative sign flips it to left-half plane (stable)
        alpha = -nn.functional.softplus(raw_out[:, :, 0:self.num_poles]) - 1e-4
        
        beta = raw_out[:, :, self.num_poles : 2*self.num_poles]
        c_re = raw_out[:, :, 2*self.num_poles : 3*self.num_poles]
        c_im = raw_out[:, :, 3*self.num_poles : 4*self.num_poles]
        
        # Shape: (Batch, Targets, 1)
        d_re = raw_out[:, :, -2].unsqueeze(-1)
        d_im = raw_out[:, :, -1].unsqueeze(-1)
        
        # 3. Construct Complex Tensors
        p = torch.complex(alpha, beta)       # Shape: (Batch, Targets, Poles)
        c = torch.complex(c_re, c_im)        # Shape: (Batch, Targets, Poles)
        d = torch.complex(d_re, d_im)        # Shape: (Batch, Targets, 1)
        
        p_conj = torch.conj(p)
        c_conj = torch.conj(c)
        
        # 4. The Mathematical "Rational Layer"
        # Unsqueeze to (Batch, Targets, Poles, 1) to broadcast with Frequencies
        p = p.unsqueeze(-1)
        c = c.unsqueeze(-1)
        p_conj = p_conj.unsqueeze(-1)
        c_conj = c_conj.unsqueeze(-1)
        
        s_view = s_tensor.view(1, 1, 1, num_freqs)
        
        # H(s) = Sum [ c/(s-p) + c*/(s-p*) ] + d
        term1 = c / (s_view - p)
        term2 = c_conj / (s_view - p_conj)
        
        # Sum across the 'Poles' dimension (dim=2). 
        # term1+term2 shape: (Batch, Targets, Poles, Freqs) -> sum -> (Batch, Targets, Freqs)
        # d shape: (Batch, Targets, 1) -> Broadcasts perfectly to (Batch, Targets, Freqs)
        H_s = torch.sum(term1 + term2, dim=2) + d
        
        # Transpose to match Target shape: (Batch, Freqs, Targets)
        return H_s.transpose(1, 2)

# ==========================================
# 3. Training Logic
# ==========================================
def main():
    results_dir = "forward_rational_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Path to the synthetic dataset
    data_path = os.path.expanduser("~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/data/processed/Synthetic-Link/synthetic_poles_dataset.pt")
    
    ds = SyntheticPoleDataset(data_path)
    train_set, val_set = random_split(ds, [int(0.8*len(ds)), len(ds)-int(0.8*len(ds))])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = RationalForwardModel(input_dim=10, num_poles=10, num_targets=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)

    f_ghz = torch.tensor(ds.freqs_ghz, dtype=torch.float32, device=device)
    s_tensor = 1j * 2 * math.pi * f_ghz

    def rational_loss(pred, target):
        mse_real = nn.functional.mse_loss(pred.real, target.real)
        mse_imag = nn.functional.mse_loss(pred.imag, target.imag)
        
        pred_mag = torch.sqrt(pred.real**2 + pred.imag**2 + 1e-12)
        target_mag = torch.sqrt(target.real**2 + target.imag**2 + 1e-12)
        db_mae = torch.abs(20*torch.log10(pred_mag) - 20*torch.log10(target_mag)).mean()
        
        return mse_real + mse_imag + (0.01 * db_mae)

    print(f"Training Physics-Informed Rational Model on {device}...")
    history = {'train': [], 'val_db': []}

    epochs = 400
    for epoch in range(epochs):
        model.train()
        t_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            out = model(x, s_tensor) 
            loss = rational_loss(out, y)
            loss.backward()
            
            # Critical: Prevents gradient explosion from complex division
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.6f} | Val MAE: {avg_v_db:.2f} dB")

    # ==========================================
    # 4. Final Validation Plot
    # ==========================================
    model.eval()
    idx = np.random.randint(len(val_set))
    x_test, y_test = val_set[idx]
    with torch.no_grad():
        pred = model(x_test.to(device).unsqueeze(0), s_tensor).cpu().squeeze()
    
    y_db = 20 * np.log10(torch.abs(y_test[:, 1]).numpy() + 1e-12)
    p_db = 20 * np.log10(torch.abs(pred[:, 1]).numpy() + 1e-12)

    plt.figure(figsize=(10, 6))
    plt.plot(ds.freqs_ghz, y_db, 'b', label='Synthetic Ground Truth', linewidth=2)
    plt.plot(ds.freqs_ghz, p_db, 'r--', label='Rational NN Prediction', linewidth=2)
    plt.title(f"Rational Layer Validation (Final Error: {history['val_db'][-1]:.2f} dB)")
    plt.ylabel("Magnitude (dB)"); plt.xlabel("Frequency (GHz)")
    plt.grid(True, alpha=0.3); plt.legend()
    
    save_path = os.path.join(results_dir, "rational_synthetic_proof.png")
    plt.savefig(save_path)
    print(f"Proof saved to {save_path}")

if __name__ == "__main__":
    main()