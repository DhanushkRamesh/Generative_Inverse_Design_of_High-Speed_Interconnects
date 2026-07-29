import os
import sys
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# HARDWARE CONFIGURATION
# =============================================================================
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim, dim),
            nn.LayerNorm(dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim, dim)
        )
    def forward(self, x):
        return x + self.net(x)

class ForwardRationalSurrogate(nn.Module):
    def __init__(self, in_features=21, n_poles=40, n_ports=4, f_scale=100e9):
        super().__init__()
        self.n_poles = n_poles
        self.n_ports = n_ports
        self.w_scale = 2.0 * math.pi * f_scale
        self.sym_elements = (n_ports * (n_ports + 1)) // 2 
        
        out_features = (n_poles * self.sym_elements * 2) + self.sym_elements
        
        hidden = 256
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            ResBlock(hidden), ResBlock(hidden), ResBlock(hidden),
            nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, out_features)
        )
        
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        
        self.register_buffer("r_idx", torch.triu_indices(n_ports, n_ports)[0])
        self.register_buffer("c_idx", torch.triu_indices(n_ports, n_ports)[1])

    def _vec_to_sym_matrix(self, vec, batch_size):
        mat = torch.zeros((batch_size, self.n_ports, self.n_ports), dtype=vec.dtype, device=vec.device)
        mat[:, self.r_idx, self.c_idx] = vec
        mat[:, self.c_idx, self.r_idx] = vec
        return mat

    def forward(self, x, freqs_hz, poles, return_raw=False):
        batch = x.shape[0]
        F = freqs_hz.shape[0]
        
        raw_out = self.mlp(x)
        out = torch.tanh(raw_out) * 2.0 
        
        idx = 0
        res_len = self.n_poles * self.sym_elements
        
        R_re_vec = out[:, idx : idx + res_len].view(batch, self.n_poles, self.sym_elements)
        idx += res_len
        R_im_vec = out[:, idx : idx + res_len].view(batch, self.n_poles, self.sym_elements)
        idx += res_len
        D_vec = out[:, idx : idx + self.sym_elements]
        
        R_re = torch.stack([self._vec_to_sym_matrix(R_re_vec[:, p, :], batch) for p in range(self.n_poles)], dim=1)
        R_im = torch.stack([self._vec_to_sym_matrix(R_im_vec[:, p, :], batch) for p in range(self.n_poles)], dim=1)
        
        R_hat = torch.complex(R_re.to(torch.float64), R_im.to(torch.float64))
        D = self._vec_to_sym_matrix(D_vec.to(torch.float64), batch)

        s_hat = 1j * (freqs_hz.to(torch.float64) / (self.w_scale / (2*math.pi)))
        poles_hat = poles.to(torch.complex128) / self.w_scale
        
        s_view = s_hat.view(1, F, 1, 1, 1)
        p_view = poles_hat.view(1, 1, self.n_poles, 1, 1)
        R_view = R_hat.unsqueeze(1) 
        
        term_pos = R_view / (s_view - p_view)
        term_neg = torch.conj(R_view) / (s_view - torch.conj(p_view))
        
        S = torch.sum(term_pos + term_neg, dim=2) + D.unsqueeze(1)
        
        if return_raw:
            return S, raw_out
        return S

class InterconnectDataset(Dataset):
    def __init__(self, data_dict):
        self.X = torch.cat([data_dict["X_local"], data_dict["X_global"], data_dict["X_context"]], dim=1)
        self.Y_real = data_dict["Y_real"]
        self.Y_imag = data_dict["Y_imag"]
        self.sim_ids = np.array(data_dict["sim_ids"])
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): 
        return self.X[idx], torch.complex(self.Y_real[idx], self.Y_imag[idx])

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def draw_progress_bar(epoch, total_epochs, metrics_str, bar_len=20):
    progress = epoch / total_epochs
    block = int(round(bar_len * progress))
    bar = "=" * block + "-" * (bar_len - block)
    sys.stdout.write(f"\rEpoch {epoch:03d}/{total_epochs} [{bar}] {progress*100:04.1f}% | {metrics_str}")
    sys.stdout.flush()

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================
def main():
    current_dir = Path(__file__).resolve().parent
    sandbox_root = current_dir if current_dir.name == "sandbox_v1" else current_dir.parent
    project_root = sandbox_root.parent
    
    dataset_path = project_root / "data" / "processed" / "Universal-Diff-SI-Array" / "diff_pair_dataset.pt"
    poles_path = sandbox_root / "data" / "universal_pole_basis.pt"
    if not poles_path.exists(): 
        poles_path = project_root / "data" / "universal_pole_basis.pt"
        
    results_dir = sandbox_root / "results"
    model_dir = results_dir / "models"
    fig_dir = results_dir / "figures"
    model_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading datasets...")
    data_dict = torch.load(dataset_path, weights_only=False)
    ds = InterconnectDataset(data_dict)
    freqs_hz = data_dict["frequencies"]
    poles = torch.load(poles_path, weights_only=False)
    
    gss = GroupShuffleSplit(n_splits=1, train_size=0.85, random_state=42)
    train_idx, val_idx = next(gss.split(ds.X, groups=ds.sim_ids))
    
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=128, num_workers=4, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForwardRationalSurrogate(in_features=21, n_poles=len(poles)).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    
    freqs_dev = freqs_hz.to(device)
    poles_dev = poles.to(device)
    
    best_linear_mae = float('inf')
    epochs_without_improvement = 0
    patience_limit = 35  
    total_epochs = 500
    
    history_train_loss = []
    history_val_loss = []
    history_val_lin = []
    history_val_db = []
    
    model_save_path = model_dir / "best_forward_surrogate.pth"
    
    print("Starting Training Pipeline...")
    for epoch in range(1, total_epochs + 1):
        model.train()
        train_loss_sum = 0
        train_lin_err_sum = 0
        
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, dtype=torch.complex128, non_blocking=True)
            optimizer.zero_grad()
            
            y_pred, raw_out = model(x, freqs_dev, poles_dev, return_raw=True)
            
            loss_cplx = nn.functional.l1_loss(y_pred.real, y.real) + nn.functional.l1_loss(y_pred.imag, y.imag)
            loss_reg = torch.mean(raw_out ** 2) 
            
            loss = loss_cplx + (0.01 * loss_reg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss_sum += loss.item()
            with torch.no_grad():
                train_lin_err_sum += torch.abs(torch.abs(y_pred[:, :, 1, 0]) - torch.abs(y[:, :, 1, 0])).mean().item()
            
        model.eval()
        val_loss_sum = 0
        val_linear_err = 0
        val_s11_err = 0
        val_s21_err = 0
        
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(device, non_blocking=True), yv.to(device, dtype=torch.complex128, non_blocking=True)
                y_pred = model(xv, freqs_dev, poles_dev)
                
                v_loss_cplx = nn.functional.l1_loss(y_pred.real, yv.real) + nn.functional.l1_loss(y_pred.imag, yv.imag)
                val_loss_sum += v_loss_cplx.item()
                
                val_linear_err += torch.abs(torch.abs(y_pred[:, :, 1, 0]) - torch.abs(yv[:, :, 1, 0])).mean().item()
                
                p_db = 20 * torch.log10(torch.clamp(torch.abs(y_pred), min=1e-5))
                t_db = 20 * torch.log10(torch.clamp(torch.abs(yv), min=1e-5))
                
                val_s11_err += torch.abs(p_db[:, :, 0, 0] - t_db[:, :, 0, 0]).mean().item()
                val_s21_err += torch.abs(p_db[:, :, 1, 0] - t_db[:, :, 1, 0]).mean().item()
                
        avg_train_loss = train_loss_sum / len(train_loader)
        avg_train_lin = train_lin_err_sum / len(train_loader)
        
        avg_val_loss = val_loss_sum / len(val_loader)
        avg_lin = val_linear_err / len(val_loader)
        avg_s11 = val_s11_err / len(val_loader)
        avg_s21 = val_s21_err / len(val_loader)
        avg_db = (avg_s11 + avg_s21) / 2.0
        
        history_train_loss.append(avg_train_loss)
        history_val_loss.append(avg_val_loss)
        history_val_lin.append(avg_lin)
        history_val_db.append(avg_s21)
        
        scheduler.step(avg_lin)
        
        marker = ""
        if avg_lin < best_linear_mae:
            best_linear_mae = avg_lin
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_save_path)
            marker = "[NEW BEST]"
        else:
            epochs_without_improvement += 1
            
        current_lr = optimizer.param_groups[0]['lr']
        
        metrics_str = (
            f"LR: {current_lr:.1e} | "
            f"TLoss: {avg_train_loss:.3f} | VLoss: {avg_val_loss:.3f} | "
            f"T_Lin: {avg_train_lin:.3f} | V_Lin: {avg_lin:.3f} | "
            f"S11: {avg_s11:.2f}dB | S21: {avg_s21:.2f}dB | Avg: {avg_db:.2f}dB {marker}"
        )
        
        draw_progress_bar(epoch, total_epochs, metrics_str)
        sys.stdout.write("\n")
                  
        if epochs_without_improvement >= patience_limit:
            sys.stdout.write(f"\n[EARLY STOP] Triggered at Epoch {epoch}. Convergence reached.\n")
            break

    # =========================================================================
    # POST-TRAINING PLOTTING
    # =========================================================================
    sys.stdout.write("\nGenerating validation plots...\n")
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(history_train_loss, label='Train Loss (L1+Reg)', color='blue')
    ax1.plot(history_val_loss, label='Val Loss (L1)', color='orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Complex L1 Loss', color='black')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    ax2 = ax1.twinx()
    ax2.plot(history_val_db, label='Val S21 dB MAE', color='red', linestyle='--')
    ax2.set_ylabel('S21 Error (dB)', color='red')
    ax2.legend(loc='lower right')
    
    loss_plot_path = fig_dir / "training_loss_curves.png"
    fig.tight_layout()
    fig.savefig(loss_plot_path, dpi=150)
    
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    model.eval()
    
    X_all = torch.cat([data_dict["X_local"], data_dict["X_global"], data_dict["X_context"]], dim=1)
    Y_real, Y_imag = data_dict["Y_real"], data_dict["Y_imag"]
    
    rng = np.random.default_rng(99)
    indices = rng.choice(len(X_all), size=3, replace=False)
    f_ghz = freqs_hz.numpy() / 1e9
    
    fig2, axes = plt.subplots(3, 2, figsize=(15, 12))
    with torch.no_grad():
        for i, idx in enumerate(indices):
            x_test = X_all[idx:idx+1].to(device)
            y_true = torch.complex(Y_real[idx:idx+1], Y_imag[idx:idx+1]).to(device)
            
            y_pred = model(x_test, freqs_dev, poles_dev)
            
            p_db = 20 * torch.log10(torch.clamp(torch.abs(y_pred[0]), min=1e-5)).cpu().numpy()
            t_db = 20 * torch.log10(torch.clamp(torch.abs(y_true[0]), min=1e-5)).cpu().numpy()
            
            p_lin = torch.abs(y_pred[0]).cpu().numpy()
            t_lin = torch.abs(y_true[0]).cpu().numpy()
            s21_lin_err = np.mean(np.abs(p_lin[:, 1, 0] - t_lin[:, 1, 0]))
            
            axes[i, 0].plot(f_ghz, t_db[:, 0, 0], 'b-', lw=2, label="HFSS Ground Truth")
            axes[i, 0].plot(f_ghz, p_db[:, 0, 0], 'r--', lw=1.5, label="Neural Surrogate")
            axes[i, 0].set_title(f"Sample {idx} - Sdd11 (Return Loss)")
            axes[i, 0].set_ylabel("Magnitude (dB)")
            axes[i, 0].grid(alpha=0.3)
            axes[i, 0].set_ylim([-60, 5])
            
            axes[i, 1].plot(f_ghz, t_db[:, 1, 0], 'b-', lw=2, label="HFSS Ground Truth")
            axes[i, 1].plot(f_ghz, p_db[:, 1, 0], 'r--', lw=1.5, label=f"Surrogate (Lin Err: {s21_lin_err:.4f})")
            axes[i, 1].set_title(f"Sample {idx} - Sdd21 (Insertion Loss)")
            axes[i, 1].grid(alpha=0.3)
            axes[i, 1].set_ylim([-80, 5])
            
            if i == 0:
                axes[i, 0].legend()
                axes[i, 1].legend()

    axes[2, 0].set_xlabel("Frequency (GHz)")
    axes[2, 1].set_xlabel("Frequency (GHz)")
    
    fig2.tight_layout()
    val_plot_path = fig_dir / "surrogate_validation_check.png"
    fig2.savefig(val_plot_path, dpi=150)
    
    sys.stdout.write(f"S-Parameter validation plot saved to: {val_plot_path}\n")
    sys.stdout.write("Pipeline execution finished.\n")

if __name__ == "__main__":
    main()