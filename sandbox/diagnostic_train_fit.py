"""
DIAGNOSTIC: Can the PRNet overfit a SINGLE sample?

If YES → The architecture has enough capacity, the problem is
         generalization (MLP can't learn the X→params mapping)
If NO  → The optimization landscape is too hard for gradient descent
         to navigate (coupled poles create local minima)

This test removes the MLP entirely and directly optimizes the 69
pole/residue parameters to fit one target curve.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import os

def load_single_sample(data_path, sample_idx=0):
    data = torch.load(data_path)
    X = torch.cat([data['X_global'], data['X_local']], dim=1)
    y_r = data['Y_real'][:, :, :, 0] if data['Y_real'].dim() == 4 else data['Y_real']
    y_i = data['Y_imag'][:, :, :, 0] if data['Y_imag'].dim() == 4 else data['Y_imag']
    Y = torch.complex(y_r, y_i)
    freqs_ghz = data['frequencies'].numpy() / 1e9
    return X[sample_idx], Y[sample_idx], freqs_ghz


def pole_residue_forward(params_raw, s_tensor, f_ghz_tensor, num_poles=8):
    """Same forward pass as PRNet but with raw parameter tensor instead of MLP."""
    num_targets = 2
    ppt = 4 * num_poles + 2
    
    pr = params_raw[:ppt*num_targets].view(num_targets, ppt)
    gamma_raw = params_raw[-1]
    
    alpha = -(torch.sigmoid(pr[:, :num_poles]) * 2.85 + 0.15)
    f_res = torch.sigmoid(pr[:, num_poles:2*num_poles]) * 100.0
    beta = 2 * math.pi * f_res
    c_re = (torch.sigmoid(pr[:, 2*num_poles:3*num_poles]) - 0.5) * 100.0
    c_im = (torch.sigmoid(pr[:, 3*num_poles:4*num_poles]) - 0.5) * 100.0
    d_re = pr[:, -2].unsqueeze(-1)
    d_im = pr[:, -1].unsqueeze(-1)
    
    num_freqs = s_tensor.shape[0]
    p = torch.complex(alpha, beta).unsqueeze(-1)       # (2, P, 1)
    c = torch.complex(c_re, c_im).unsqueeze(-1)         # (2, P, 1)
    d = torch.complex(d_re, d_im)                       # (2, 1)
    s_view = s_tensor.view(1, 1, num_freqs)              # (1, 1, F)
    
    denom1 = s_view - p                                  # (2, P, F)
    denom2 = s_view - torch.conj(p)
    eps = 1e-4
    safe_d1 = torch.complex(denom1.real + eps*(denom1.real.abs()<eps).float(), denom1.imag)
    safe_d2 = torch.complex(denom2.real + eps*(denom2.real.abs()<eps).float(), denom2.imag)
    
    term1 = c / safe_d1                                  # (2, P, F)
    term2 = torch.conj(c) / safe_d2
    H_s = torch.sum(term1 + term2, dim=1) + d            # (2, F) + (2, 1) -> (2, F)
    H_s = torch.complex(H_s.real.clamp(-200,200), H_s.imag.clamp(-200,200))
    
    gamma = torch.nn.functional.softplus(gamma_raw)
    exp_decay = torch.exp(-gamma * f_ghz_tensor).to(torch.complex64)
    
    H_s11 = H_s[0, :]
    H_s21 = H_s[1, :] * exp_decay
    
    return torch.stack([H_s11, H_s21], dim=-1)  # (F, 2)


def run_diagnostic(data_path, results_dir="diagnostic_results"):
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test on 5 different samples
    sample_indices = [0, 100, 500, 1000, 1500]
    
    fig, axes = plt.subplots(2, 5, figsize=(25, 8))
    fig.suptitle("DIAGNOSTIC: Direct Parameter Optimization (No MLP) — Can we fit single samples?", 
                 fontsize=14, fontweight='bold')
    
    results = []
    
    for col, si in enumerate(sample_indices):
        x_sample, y_sample, f_ghz = load_single_sample(data_path, si)
        y_target = y_sample.to(device)
        
        f_ghz_t = torch.tensor(f_ghz, dtype=torch.float32, device=device)
        omega = 2 * math.pi * f_ghz_t
        s_tensor = torch.complex(torch.zeros_like(omega), omega)
        
        # Initialize raw parameters — DIRECTLY optimizable, no MLP
        num_poles = 8
        ppt = 4 * num_poles + 2
        total_params = ppt * 2 + 1  # 69
        
        params = torch.randn(total_params, device=device, requires_grad=True)
        
        # Spreading init for f_res
        target_f = torch.linspace(0.05, 0.95, num_poles)
        inv_sig = torch.log(target_f / (1.0 - target_f))
        with torch.no_grad():
            for t in range(2):
                off = t * ppt
                params[off + num_poles : off + 2*num_poles] = inv_sig.to(device)
                params[off + 2*num_poles : off + 4*num_poles] = 0.5
        
        optimizer = optim.Adam([params], lr=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-5)
        
        history = []
        
        for step in range(2000):
            optimizer.zero_grad()
            pred = pole_residue_forward(params, s_tensor, f_ghz_t, num_poles)
            
            pm = torch.clamp(torch.abs(pred), min=1e-7)
            tm = torch.clamp(torch.abs(y_target), min=1e-7)
            pdb = torch.clamp(20*torch.log10(pm), -100, 40)
            tdb = torch.clamp(20*torch.log10(tm), -100, 40)
            loss = nn.functional.smooth_l1_loss(pdb, tdb)
            
            if torch.isnan(loss):
                break
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_([params], 1.0)
            optimizer.step()
            scheduler.step()
            history.append(loss.item())
        
        final_mae = history[-1] if history else float('inf')
        results.append(final_mae)
        
        # Plot Sdd21
        with torch.no_grad():
            pred_final = pole_residue_forward(params, s_tensor, f_ghz_t, num_poles)
        
        ax = axes[0, col]
        td = 20*np.log10(np.abs(y_target.cpu()[:, 1].numpy())+1e-12)
        pd = 20*np.log10(np.abs(pred_final.cpu()[:, 1].numpy())+1e-12)
        ax.plot(f_ghz, td, 'b-', lw=2, label='Target')
        ax.plot(f_ghz, pd, 'r--', lw=2, label=f'Fit ({final_mae:.2f} dB)')
        ax.set_title(f"Sample {si} — Sdd21", fontsize=10)
        ax.set_xlabel("GHz"); ax.set_ylabel("dB")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        
        # Plot convergence
        ax = axes[1, col]
        ax.plot(history, 'b-', lw=1)
        ax.set_title(f"Convergence → {final_mae:.2f} dB", fontsize=10)
        ax.set_xlabel("Step"); ax.set_ylabel("Loss (dB)")
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        print(f"Sample {si}: Final MAE = {final_mae:.2f} dB")
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "single_sample_diagnostic.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    avg = np.mean(results)
    print(f"\n{'='*50}")
    print(f"DIAGNOSTIC RESULT")
    print(f"{'='*50}")
    print(f"Average single-sample fit: {avg:.2f} dB")
    if avg < 0.5:
        print("VERDICT: Architecture CAN represent the data.")
        print("         Problem is MLP generalization, not optimization landscape.")
        print("         → Need better MLP or curriculum learning.")
    elif avg < 2.0:
        print("VERDICT: Architecture PARTIALLY represents the data.")
        print("         Optimization landscape has local minima but can be escaped.")
        print("         → Need multi-start init or curriculum on pole count.")
    else:
        print("VERDICT: Even direct optimization fails.")
        print("         The pole-residue formula + bounded activations cannot fit the data.")
        print("         → Need to change the architecture (more poles, unbounded activations, etc).")
    print(f"{'='*50}")
    
    return results


if __name__ == "__main__":
    data_path = os.path.expanduser(
        "~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects"
        "/data/processed/Synthetic-Link/synthetic_poles_dataset.pt"
    )
    run_diagnostic(data_path)