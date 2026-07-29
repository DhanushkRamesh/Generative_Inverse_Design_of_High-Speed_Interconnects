"""
day1_sanity_check.py
--------------------
Day-1 question:  Does the rational layer correctly represent the rational form
                 S(s) = sum_n  R_n / (s - p_n) + D ?

We answer this two ways:

  TEST A  (gradient training):
    Random-initialise residues+D, train with Adam to fit a known target,
    confirm the loss collapses to a small value.  This proves gradient flow
    through the layer is well-behaved.

  TEST B  (closed-form least squares):
    With poles fixed, the (residues, D) fit is LINEAR in the parameters.
    Solve it in closed form via np.linalg.lstsq and verify the residual
    is at machine epsilon.  This proves the rational form itself is
    correctly implemented (independent of any optimiser quirks).

If TEST A reaches < 1e-3 and TEST B reaches < 1e-10  ->  layer is sound.

Important takeaway for your real forward model:
  Because residues+D are linear-in-parameters, in Day 2-3 you will use the
  MLP only to learn the *geometry -> residues* mapping.  Within each
  evaluation, residues come from the MLP and are then plugged into the
  exact rational form; you never need gradient descent on the residues
  themselves once you have a working MLP.
"""

import math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from synthetic_data import generate_synthetic_sample, s_to_db
from rational_forward_v1 import RationalLayer, complex_mse_loss


def test_A_gradient_fit(target, n_iters=2000, lr=5e-3):
    """Random-init residues, Adam-train to match the target."""
    true_poles = torch.from_numpy(target['poles']).to(torch.complex128)
    freqs      = torch.from_numpy(target['freqs']).to(torch.float64)
    S_target   = torch.from_numpy(target['S']).to(torch.complex128)

    torch.manual_seed(0)
    layer = RationalLayer(true_poles, n_ports=4, f_scale=100e9,
                          init_scale=0.01, dtype=torch.complex128)
    optim = torch.optim.Adam(layer.parameters(), lr=lr)

    history = []
    for it in range(n_iters):
        optim.zero_grad()
        loss = complex_mse_loss(layer(freqs), S_target)
        loss.backward()
        optim.step()
        history.append(loss.item())
        if it % 200 == 0 or it == n_iters - 1:
            print(f"  Adam  iter {it:5d}   MSE = {loss.item():.4e}")

    final_S = layer(freqs).detach().numpy()
    return history[-1], final_S, history


def test_B_closed_form(target):
    """Solve residues+D in closed form via LSQ."""
    poles  = target['poles']
    freqs  = target['freqs']
    S      = target['S']
    w_scale = 2 * math.pi * 100e9

    s_hat = 1j * freqs / 100e9                # (F,)
    p_hat = poles / w_scale                    # (N,)
    N = len(poles)
    F = len(freqs)

    Phi = np.zeros((F, N + 1), dtype=complex)
    Phi[:, :N] = 1.0 / (s_hat[:, None] - p_hat[None, :])
    Phi[:, N] = 1.0

    S_pred = np.zeros_like(S)
    max_err = 0.0
    for i in range(4):
        for j in range(4):
            x, *_ = np.linalg.lstsq(Phi, S[:, i, j], rcond=None)
            S_pred[:, i, j] = Phi @ x
            max_err = max(max_err, np.max(np.abs(Phi @ x - S[:, i, j])))
    return max_err, S_pred


def plot_results(target, S_adam, S_lsq, adam_history, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    f_GHz = target['freqs'] / 1e9
    pairs = [(0, 0, "S11 reflection"),
             (1, 0, "S21 forward path"),
             (2, 0, "S31 NE cross-talk"),
             (3, 1, "S42"),
             (1, 1, "S22"),
             (2, 2, "S33")]

    for ax, (i, j, title) in zip(axes.flatten(), pairs):
        ax.plot(f_GHz, s_to_db(target['S'][:, i, j]), 'b-', lw=2, label='Target')
        ax.plot(f_GHz, s_to_db(S_adam[:, i, j]),     'r--', lw=1.2, label='Adam')
        ax.plot(f_GHz, s_to_db(S_lsq[:, i, j]),      'g:',  lw=1.5, label='LSQ')
        ax.set_title(title)
        ax.set_xlabel("f (GHz)")
        ax.set_ylabel("|S| (dB)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Day-1 sanity check: rational layer fits known target", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"\nSaved plot -> {save_path}")

    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(adam_history)
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel("Complex MSE (log scale)")
    ax.set_title("Adam loss curve (random init, fixed poles)")
    ax.grid(alpha=0.3, which='both')
    fig2.tight_layout()
    fig2.savefig(save_path.replace('.png', '_loss.png'), dpi=120, bbox_inches='tight')


def main():
    print("=" * 70)
    print("Day-1 sanity check: rational layer vs. known synthetic target")
    print("=" * 70)

    target = generate_synthetic_sample(seed=42)
    print(f"\nGenerated target:")
    print(f"  poles:    {target['poles'].shape}  (20 conjugate pairs)")
    print(f"  S:        {target['S'].shape}")
    print(f"  |S| range: {np.abs(target['S']).min():.4f} ... {np.abs(target['S']).max():.4f}")
    print(f"  freqs:    {target['freqs'].min()/1e9:.2f} ... {target['freqs'].max()/1e9:.1f} GHz")

    print("\n--- TEST A: Adam gradient fit ---")
    final_mse_adam, S_adam, history = test_A_gradient_fit(target, n_iters=2000)

    print("\n--- TEST B: closed-form LSQ ---")
    max_err_lsq, S_lsq = test_B_closed_form(target)
    print(f"  LSQ max |S_pred - S_true| = {max_err_lsq:.4e}")

    plot_results(target, S_adam, S_lsq, history,
                 "results/rational_forward_fit.png")

    print("\n" + "=" * 70)
    print(" VERDICT")
    print("=" * 70)
    adam_pass = final_mse_adam < 1e-3
    lsq_pass  = max_err_lsq < 1e-10

    print(f"  TEST A (Adam):  MSE = {final_mse_adam:.4e}   "
          f"{'PASS' if adam_pass else 'FAIL'}")
    print(f"  TEST B (LSQ):   max err = {max_err_lsq:.4e}   "
          f"{'PASS' if lsq_pass else 'FAIL'}")

    if adam_pass and lsq_pass:
        print(
            "\n  Architecture is sound. Day-2 actions:\n"
            "    (a) Run scikit-rf VFIT on ~30 real samples from your\n"
            "        Universal-Diff-SI-Array dataset.\n"
            "    (b) K-means the pooled poles to a shared 40-pole basis.\n"
            "    (c) Wrap the layer with a Fourier-feature MLP that maps\n"
            "        X_local -> residues, and train end-to-end.\n"
        )
    else:
        print("\n  Investigation needed before Day 2.\n")
    print("=" * 70)
    return adam_pass and lsq_pass


if __name__ == "__main__":
    main()