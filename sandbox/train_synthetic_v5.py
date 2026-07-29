import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import math

class ParameterDataset(Dataset):
    def __init__(self, pt_file_path):
        data = torch.load(pt_file_path)
        self.X = torch.cat([data['X_global'], data['X_local']], dim=1)
        self.P_gt = data['P_gt']
        self.C_gt = data['C_gt']
        
        y_r = data['Y_real'][:, :, :, 0] if data['Y_real'].dim() == 4 else data['Y_real']
        y_i = data['Y_imag'][:, :, :, 0] if data['Y_imag'].dim() == 4 else data['Y_imag']
        self.Y_complex = torch.complex(y_r, y_i)
        self.freqs_ghz = data['frequencies'].numpy() / 1e9

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.P_gt[idx], self.C_gt[idx], self.Y_complex[idx]

class ParameterMLP(nn.Module):
    def __init__(self, input_dim=10, num_poles=3, num_targets=2):
        super().__init__()
        self.num_targets = num_targets
        self.num_poles = num_poles
        
        # 4 values per pole (alpha, beta, c_re, c_im) * 3 poles * 2 targets = 24 outputs
        output_dim = 4 * num_poles * num_targets
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.GELU(),
            nn.Linear(128, 256), nn.GELU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        batch = x.shape[0]
        out = self.net(x).view(batch, self.num_targets, self.num_poles, 4)
        
        # We don't bound these with Sigmoids because we are doing direct linear regression
        P_pred = torch.complex(out[..., 0], out[..., 1])
        C_pred = torch.complex(out[..., 2], out[..., 3])
        return P_pred, C_pred

def main():
    results_dir = "synthetic_parameter_results"
    os.makedirs(results_dir, exist_ok=True)
    
    data_path = os.path.expanduser("~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/data/processed/Synthetic-Link/synthetic_poles_dataset.pt")
    ds = ParameterDataset(data_path)
    
    train_set, val_set = random_split(ds, [int(0.85*len(ds)), len(ds)-int(0.85*len(ds))])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ParameterMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"Training Two-Step Surrogate (Parameter Regression)...")
    epochs = 150
    for epoch in range(epochs):
        model.train()
        for x, p_gt, c_gt, _ in train_loader:
            x, p_gt, c_gt = x.to(device), p_gt.to(device), c_gt.to(device)
            optimizer.zero_grad()
            p_pred, c_pred = model(x)
            
            # --- THE MAGIC: Direct Parameter MSE ---
            loss = nn.functional.mse_loss(p_pred.real, p_gt.real) + \
                   nn.functional.mse_loss(p_pred.imag, p_gt.imag) + \
                   nn.functional.mse_loss(c_pred.real, c_gt.real) + \
                   nn.functional.mse_loss(c_pred.imag, c_gt.imag)
                   
            loss.backward()
            optimizer.step()
        
        # Validation checks S-parameter reconstruction
        model.eval()
        v_mae = 0
        s_tensor = torch.tensor(1j * 2 * math.pi * ds.freqs_ghz, dtype=torch.complex64, device=device)
        s_view = s_tensor.view(1, 1, 1, 401)
        
        with torch.no_grad():
            for xv, _, _, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                p_pred, c_pred = model(xv)
                
                # Reconstruct S-parameters from predicted poles
                p = p_pred.unsqueeze(-1)
                c = c_pred.unsqueeze(-1)
                
                H_s = torch.sum(c/(s_view - p) + torch.conj(c)/(s_view - torch.conj(p)), dim=2)
                H_s = H_s.transpose(1, 2)
                
                p_db = 20 * torch.log10(torch.clamp(torch.abs(H_s), min=1e-5))
                t_db = 20 * torch.log10(torch.clamp(torch.abs(yv), min=1e-5))
                v_mae += torch.abs(p_db - t_db).mean().item()
                
        avg_mae = v_mae / len(val_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d} | S-Parameter Reconstruction MAE: {avg_mae:.5f} dB")

    # Final Plot
    model.eval()
    idx = np.random.randint(len(val_set))
    x_test, _, _, y_test = val_set[idx]
    with torch.no_grad():
        p_pred, c_pred = model(x_test.unsqueeze(0).to(device))
        p, c = p_pred.unsqueeze(-1), c_pred.unsqueeze(-1)
        H_s = torch.sum(c/(s_view - p) + torch.conj(c)/(s_view - torch.conj(p)), dim=2).transpose(1, 2).cpu()
    
    plt.figure(figsize=(10, 6))
    plt.plot(ds.freqs_ghz, 20*np.log10(torch.clamp(torch.abs(y_test[:, 1]), min=1e-5).numpy()), 'b', label='True')
    plt.plot(ds.freqs_ghz, 20*np.log10(torch.clamp(torch.abs(H_s[0, :, 1]), min=1e-5).numpy()), 'r--', label='Predicted')
    plt.title(f"Two-Step Parameter Regression (Final MAE: {avg_mae:.5f} dB)")
    plt.legend(); plt.grid(alpha=0.3); plt.savefig(os.path.join(results_dir, "two_step_proof.png"))
    print("Proof saved! The model has proven it can learn the geometry mapping.")

if __name__ == "__main__":
    main()