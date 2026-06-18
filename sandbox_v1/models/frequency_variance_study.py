"""
04_frequency_importance_eda.py
================================================================================
Frequency Importance Analysis for the Universal-Diff-SI-Array dataset.

Goal:
    Identify which regions of the 0-100 GHz band are MOST important to fit
    accurately for downstream inverse design, and which can be deprioritized or
    ignored. The analysis is anchored to 112G PAM4 SerDes specifications
    (Nyquist 28 GHz, 3 dB channel bandwidth ~42 GHz, harmonic content out to
    ~56 GHz). The output is a locked set of frequency-band weights consumed by
    the inverse model training and Test-Time Optimization.

Structure:
    Section 1:  Load dataset, splits, and trained forward model checkpoint.
    Section 2:  Per-frequency dataset magnitude statistics
                (mean / median / std / percentile envelopes for Sdd11, Sdd21).
    Section 3:  Rare-feature analysis (deep nulls and their frequency clustering).
    Section 4:  Per-frequency forward model error on validation set (MAE in dB).
    Section 5:  Noise floor verification: where does the dataset run out of
                meaningful dynamic range, justifying or revising the -45 dB choice.
    Section 6:  112G PAM4 industry band overlay on all results.
    Section 7:  Cross-band variance and correlation analysis
                (do different bands carry independent information?).
    Section 8:  Final band weight recommendation and JSON output for downstream use.

This is a Jupytext-style Python script: it runs as a normal script and also
opens cleanly cell-by-cell in Jupyter / VS Code if you open it as a notebook
(cells delimited by '# %%').

References:
    IEEE 802.3ck (2022): 112G PAM4 electrical specifications
    OIF-CEI-112G-PAM4: chip-to-chip and chip-to-module 112G PAM4 standards
    Synopsys / Cadence application notes on 112G channel design
"""

# %% Section 0: Imports and configuration
import json
import sys
from pathlib import Path
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")  # safe for headless terminals; comment out for interactive
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Project paths (matches the rest of your sandbox_v1 conventions)
PROJECT_ROOT = Path.home() / "mece_project_inverse_model" / "Generative_Inverse_Design_of_High-Speed_Interconnects"
DATA_PT = PROJECT_ROOT / "data" / "processed" / "Universal-Diff-SI-Array" / "diff_pair_dataset.pt"

# USER: UPDATE THIS PATH to point to your best forward model checkpoint
FORWARD_MODEL_PATH = (
    PROJECT_ROOT / "sandbox_v1" / "models" / "forward_runs"
    / "run_2026-06-01_213554_direct_resnet_array" / "checkpoint_best.pt"
)

# Output directory for plots and the band weights JSON
EDA_OUT_DIR = PROJECT_ROOT / "sandbox_v1" / "data" / "frequency_eda"
EDA_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Noise floor currently used by the forward-model training (justified or revised below)
NOISE_FLOOR_DB = -45.0

# Plot styling: thesis-friendly defaults
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# 112G PAM4 reference frequencies (locked by industry spec)
NYQUIST_112G_GHZ = 28.0          # IEEE 802.3ck / OIF-CEI-112G Nyquist
HALF_NYQUIST_GHZ = 14.0          # sub-Nyquist baseband
CHANNEL_BW_GHZ = 42.0            # 0.75 x symbol rate (PAM4 rule of thumb)
TWICE_NYQUIST_GHZ = 56.0         # harmonic / FEXT relevance ceiling

print(f"Project root: {PROJECT_ROOT}")
print(f"Data file:    {DATA_PT}")
print(f"Output dir:   {EDA_OUT_DIR}")
print(f"Forward checkpoint: {FORWARD_MODEL_PATH}")
print(f"Noise floor in use: {NOISE_FLOOR_DB} dB")


# %% Section 1: Load dataset, splits, and inspect contents
print("\n=== Section 1: Dataset loading ===")
payload = torch.load(DATA_PT, weights_only=False, map_location="cpu")

freqs_hz = payload["frequencies"].numpy()   # (F,) in Hz
freqs_ghz = freqs_hz / 1e9
F_LEN = len(freqs_ghz)
print(f"Total pairs:        {payload['X_local'].shape[0]:,}")
print(f"Frequency points:   {F_LEN}")
print(f"Frequency range:    {freqs_ghz.min():.3f} - {freqs_ghz.max():.3f} GHz")
print(f"Frequency spacing:  {freqs_ghz[1] - freqs_ghz[0]:.4f} GHz "
      f"(uniform: {np.allclose(np.diff(freqs_ghz), freqs_ghz[1] - freqs_ghz[0])})")

# Reconstruct the train/val split exactly as used during forward training
# (sim-level split, 85/15, deterministic with seed 42)
sim_ids = np.array(payload["sim_ids"])
unique_sims = np.unique(sim_ids)
rng = np.random.default_rng(42)
rng.shuffle(unique_sims)
n_train_sims = int(0.85 * len(unique_sims))
train_sims = set(unique_sims[:n_train_sims])

train_idx = np.array([i for i, sid in enumerate(sim_ids) if sid in train_sims])
val_idx = np.array([i for i, sid in enumerate(sim_ids) if sid not in train_sims])
print(f"Train pairs: {len(train_idx):,}  ({len(train_idx)/len(sim_ids)*100:.1f}%)")
print(f"Val pairs:   {len(val_idx):,}  ({len(val_idx)/len(sim_ids)*100:.1f}%)")

# Extract complex S-parameters for analysis (use validation set for honest stats;
# training set is also analyzed in Section 2 for distribution context)
Y_complex_val = torch.complex(
    payload["Y_real"][val_idx].to(torch.float64),
    payload["Y_imag"][val_idx].to(torch.float64),
).numpy()  # (N_val, F, 4, 4)
Y_complex_train = torch.complex(
    payload["Y_real"][train_idx].to(torch.float64),
    payload["Y_imag"][train_idx].to(torch.float64),
).numpy()
print(f"Complex S arrays:  train {Y_complex_train.shape}, val {Y_complex_val.shape}")


# %% Section 2: Per-frequency magnitude statistics
# What we learn: where the signal envelope lives, where the variation lives,
# what the model is being asked to fit at each frequency.

print("\n=== Section 2: Per-frequency magnitude statistics ===")
EPS = 1e-12


def magnitude_db(S_complex, i, j):
    """20 log10 |S_ij| for a (N, F, 4, 4) array, returning (N, F)."""
    return 20.0 * np.log10(np.abs(S_complex[..., i, j]) + EPS)


# Compute the dB magnitude grids once
Sdd11_db_train = magnitude_db(Y_complex_train, 0, 0)  # (N_train, F)
Sdd21_db_train = magnitude_db(Y_complex_train, 1, 0)
Sdd11_db_val = magnitude_db(Y_complex_val, 0, 0)
Sdd21_db_val = magnitude_db(Y_complex_val, 1, 0)


def per_freq_stats(arr_db):
    """Returns dict of per-frequency statistics (mean/median/std/percentiles) in dB."""
    return {
        "mean":  arr_db.mean(axis=0),
        "median": np.median(arr_db, axis=0),
        "std":    arr_db.std(axis=0),
        "p05":  np.percentile(arr_db, 5, axis=0),
        "p25":  np.percentile(arr_db, 25, axis=0),
        "p75":  np.percentile(arr_db, 75, axis=0),
        "p95":  np.percentile(arr_db, 95, axis=0),
        "min":   arr_db.min(axis=0),
        "max":   arr_db.max(axis=0),
    }


stats_Sdd11_train = per_freq_stats(Sdd11_db_train)
stats_Sdd21_train = per_freq_stats(Sdd21_db_train)
stats_Sdd11_val = per_freq_stats(Sdd11_db_val)
stats_Sdd21_val = per_freq_stats(Sdd21_db_val)

# Plot 1: Per-frequency mean +/- std and percentile envelopes for Sdd11 and Sdd21
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

for ax, label, stats in zip(
    axes, ["|Sdd11|", "|Sdd21|"],
    [stats_Sdd11_train, stats_Sdd21_train],
):
    ax.fill_between(freqs_ghz, stats["p05"], stats["p95"],
                     alpha=0.20, color="steelblue", label="5-95 percentile")
    ax.fill_between(freqs_ghz, stats["p25"], stats["p75"],
                     alpha=0.40, color="steelblue", label="25-75 percentile")
    ax.plot(freqs_ghz, stats["median"], color="navy", lw=1.5, label="median")
    ax.plot(freqs_ghz, stats["mean"], color="darkorange", lw=1.0, ls="--",
             label="mean")
    ax.axhline(NOISE_FLOOR_DB, color="gray", ls=":", lw=1.0,
                label=f"noise floor {NOISE_FLOOR_DB} dB")
    ax.set_ylabel(f"{label} [dB]")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"Per-frequency {label} distribution across {len(train_idx)} "
                  f"training pairs")
    ax.set_ylim(bottom=-80)

axes[-1].set_xlabel("Frequency [GHz]")
plt.tight_layout()
plt.savefig(EDA_OUT_DIR / "01_per_freq_magnitude_distribution.png",
             bbox_inches="tight")
plt.close()
print(f"Saved: 01_per_freq_magnitude_distribution.png")

# Plot 2: Per-frequency standard deviation, showing where the dataset has
# meaningful variability (= where the model must work hard to discriminate samples)
fig, ax = plt.subplots(1, 1, figsize=(12, 4.5))
ax.plot(freqs_ghz, stats_Sdd11_train["std"], color="C0", lw=1.5,
         label="|Sdd11| std (dB)")
ax.plot(freqs_ghz, stats_Sdd21_train["std"], color="C1", lw=1.5,
         label="|Sdd21| std (dB)")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Per-frequency std [dB]")
ax.set_title("Where the dataset has variability: higher std = more diverse "
              "responses to fit")
ax.legend()
plt.tight_layout()
plt.savefig(EDA_OUT_DIR / "02_per_freq_std.png", bbox_inches="tight")
plt.close()
print(f"Saved: 02_per_freq_std.png")


# %% Section 3: Rare-feature analysis (deep nulls)
# Where do sharp resonant features cluster? These are the targets that the cVAE
# struggles to reproduce, so we need to know if they live in the critical band.

print("\n=== Section 3: Deep-null frequency clustering ===")


def count_below_threshold(arr_db, threshold_db):
    """For each frequency, count samples whose |S| < threshold."""
    return (arr_db < threshold_db).sum(axis=0)


thresholds_db = [-20, -30, -40, -50]
null_counts_Sdd11 = {t: count_below_threshold(Sdd11_db_train, t)
                      for t in thresholds_db}
null_counts_Sdd21 = {t: count_below_threshold(Sdd21_db_train, t)
                      for t in thresholds_db}

# Plot 3: Count of "deep-null" samples per frequency
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for ax, label, counts in zip(
    axes, ["|Sdd11|", "|Sdd21|"],
    [null_counts_Sdd11, null_counts_Sdd21],
):
    for t in thresholds_db:
        ax.plot(freqs_ghz, counts[t], lw=1.2,
                 label=f"samples with {label} < {t} dB")
    ax.set_ylabel(f"Number of {label} samples")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Distribution of deep features in {label} across frequency")

axes[-1].set_xlabel("Frequency [GHz]")
plt.tight_layout()
plt.savefig(EDA_OUT_DIR / "03_deep_null_distribution.png", bbox_inches="tight")
plt.close()
print(f"Saved: 03_deep_null_distribution.png")

# Quantify: what fraction of validation samples have at least one null below -30 dB
# in each band? This is what the cVAE is failing on.
def per_sample_min_in_band(arr_db, freq_mask):
    return arr_db[:, freq_mask].min(axis=1)


band_definitions_for_null = OrderedDict([
    ("0-14 GHz", (freqs_ghz >= 0) & (freqs_ghz < 14)),
    ("14-28 GHz", (freqs_ghz >= 14) & (freqs_ghz < 28)),
    ("28-42 GHz", (freqs_ghz >= 28) & (freqs_ghz < 42)),
    ("42-56 GHz", (freqs_ghz >= 42) & (freqs_ghz < 56)),
    ("56-100 GHz", (freqs_ghz >= 56) & (freqs_ghz <= 100)),
])

print("\nFraction of TRAINING samples with at least one null below -30 dB in band:")
print(f"  {'Band':12s}  {'Sdd11':>10s}  {'Sdd21':>10s}")
for band_name, mask in band_definitions_for_null.items():
    f_min_S11 = (per_sample_min_in_band(Sdd11_db_train, mask) < -30).mean()
    f_min_S21 = (per_sample_min_in_band(Sdd21_db_train, mask) < -30).mean()
    print(f"  {band_name:12s}  {f_min_S11*100:>9.1f}%  {f_min_S21*100:>9.1f}%")


# %% Section 4: Forward model architecture (needed to load checkpoint)
# Reproducing the exact architecture from your training script so the saved
# state_dict loads without surprises.

class Conv1DResBlock(nn.Module):
    def __init__(self, channels, dropout=0.10):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2,
                                padding_mode="replicate")
        self.norm1 = nn.GroupNorm(4, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2,
                                padding_mode="replicate")
        self.norm2 = nn.GroupNorm(4, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        h = F.silu(self.conv1(h))
        h = self.drop(h)
        h = self.norm2(h)
        h = self.conv2(h)
        return x + h


class DirectSequenceResNet(nn.Module):
    UPPER_R = (0, 0, 0, 0, 1, 1, 1, 2, 2, 3)
    UPPER_C = (0, 1, 2, 3, 1, 2, 3, 2, 3, 3)

    def __init__(self, d_local=8, d_global=6, d_context=7,
                 F_len=401, hidden_dim=384, n_blocks=8):
        super().__init__()
        self.F_len = F_len
        self.is_link_dataset = (d_global == 7)
        in_dim = d_local + d_global + d_context

        self.geom_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
        )
        self.learnable_velocity = nn.Parameter(torch.tensor(10.0))
        self.proj_in = nn.Conv1d(hidden_dim + 3, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [Conv1DResBlock(hidden_dim, dropout=0.10) for _ in range(n_blocks)]
        )
        self.proj_out = nn.Conv1d(hidden_dim, 20, kernel_size=1)
        self.register_buffer("upper_r",
                              torch.tensor(self.UPPER_R, dtype=torch.long))
        self.register_buffer("upper_c",
                              torch.tensor(self.UPPER_C, dtype=torch.long))

    def _scatter_symmetric(self, vec_real, vec_imag, B):
        mat = torch.zeros((B, 4, 4, self.F_len), dtype=torch.float64,
                           device=vec_real.device)
        mat_i = torch.zeros((B, 4, 4, self.F_len), dtype=torch.float64,
                             device=vec_real.device)
        mat[:, self.upper_r, self.upper_c, :] = vec_real.to(torch.float64)
        mat[:, self.upper_c, self.upper_r, :] = vec_real.to(torch.float64)
        mat_i[:, self.upper_r, self.upper_c, :] = vec_imag.to(torch.float64)
        mat_i[:, self.upper_c, self.upper_r, :] = vec_imag.to(torch.float64)
        S = torch.complex(mat, mat_i)
        return S.permute(0, 3, 1, 2)

    def forward(self, x_local, x_global, x_context, freqs_hz_t):
        B = x_local.shape[0]
        x = torch.cat([x_local, x_global, x_context], dim=-1)
        h_geom = self.geom_mlp(x)
        h_seq = h_geom.unsqueeze(-1).expand(-1, -1, self.F_len)
        f_norm = (freqs_hz_t / freqs_hz_t.max()).view(1, 1, self.F_len)
        f_norm = f_norm.expand(B, 1, -1).to(h_seq.dtype)
        length_feature = (
            x_global[:, -1].view(B, 1, 1) if self.is_link_dataset
            else torch.zeros((B, 1, 1), device=x.device, dtype=h_seq.dtype)
        )
        phase = self.learnable_velocity * f_norm * length_feature
        h = torch.cat([h_seq, f_norm, torch.sin(phase), torch.cos(phase)], dim=1)
        h = self.proj_in(h)
        for block in self.blocks:
            h = block(h)
        out = self.proj_out(h)
        return self._scatter_symmetric(out[:, :10, :], out[:, 10:, :], B)


# %% Section 5: Forward model per-frequency MAE on validation set
# What we learn: where the model is already good (low error => low band weight
# needed because there's nothing more to fix), and where it struggles (high
# error => high band weight justified to push improvement).

print("\n=== Section 5: Per-frequency forward-model error ===")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model_available = False
try:
    model = DirectSequenceResNet(
        d_local=payload["X_local"].shape[1],
        d_global=payload["X_global"].shape[1],
        d_context=payload["X_context"].shape[1],
        F_len=F_LEN,
        hidden_dim=384,
        n_blocks=8,
    ).to(device)
    state = torch.load(FORWARD_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model_available = True
    print(f"Forward model loaded: {sum(p.numel() for p in model.parameters()):,}"
          f" parameters")
except Exception as e:
    print(f"Could not load forward model ({e}); skipping per-freq error analysis.")

if model_available:
    # Run inference on the full validation set in chunks (memory-safe)
    BATCH = 64
    freqs_hz_t = torch.from_numpy(freqs_hz).to(device).to(torch.float64)
    xl_v = payload["X_local"][val_idx]
    xg_v = payload["X_global"][val_idx]
    xc_v = payload["X_context"][val_idx]
    yr_v = payload["Y_real"][val_idx]
    yi_v = payload["Y_imag"][val_idx]

    abs_err_Sdd11 = []  # per-pair, per-frequency dB error magnitude
    abs_err_Sdd21 = []
    pred_db_Sdd11_all = []  # for noise-floor analysis below
    pred_db_Sdd21_all = []

    with torch.no_grad():
        for i in range(0, len(val_idx), BATCH):
            sl = slice(i, i + BATCH)
            S_pred = model(
                xl_v[sl].to(device),
                xg_v[sl].to(device),
                xc_v[sl].to(device),
                freqs_hz_t,
            ).cpu()
            S_tgt = torch.complex(yr_v[sl].to(torch.float64),
                                   yi_v[sl].to(torch.float64))
            # dB magnitudes
            p_db = 20.0 * torch.log10(S_pred.abs().clamp_min(EPS))
            t_db = 20.0 * torch.log10(S_tgt.abs().clamp_min(EPS))
            # Mask: only count errors where the TRUE response is above noise floor
            # (matches the metric used during training)
            mask11 = (t_db[..., 0, 0] > NOISE_FLOOR_DB).to(torch.float64)
            mask21 = (t_db[..., 1, 0] > NOISE_FLOOR_DB).to(torch.float64)
            err11 = (p_db[..., 0, 0] - t_db[..., 0, 0]).abs() * mask11
            err21 = (p_db[..., 1, 0] - t_db[..., 1, 0]).abs() * mask21
            # We sum errors and count valid points so we can compute the masked mean
            abs_err_Sdd11.append((err11.numpy(), mask11.numpy()))
            abs_err_Sdd21.append((err21.numpy(), mask21.numpy()))
            pred_db_Sdd11_all.append(p_db[..., 0, 0].numpy())
            pred_db_Sdd21_all.append(p_db[..., 1, 0].numpy())

    # Per-frequency masked MAE
    e11_arr = np.concatenate([x for x, _ in abs_err_Sdd11], axis=0)
    m11_arr = np.concatenate([m for _, m in abs_err_Sdd11], axis=0)
    e21_arr = np.concatenate([x for x, _ in abs_err_Sdd21], axis=0)
    m21_arr = np.concatenate([m for _, m in abs_err_Sdd21], axis=0)
    pred_S11_db_arr = np.concatenate(pred_db_Sdd11_all, axis=0)
    pred_S21_db_arr = np.concatenate(pred_db_Sdd21_all, axis=0)

    # Per-frequency MAE: sum-of-errors / count-of-unmasked-points
    eps_counts = 1e-6
    mae_Sdd11_per_freq = e11_arr.sum(axis=0) / (m11_arr.sum(axis=0) + eps_counts)
    mae_Sdd21_per_freq = e21_arr.sum(axis=0) / (m21_arr.sum(axis=0) + eps_counts)
    valid_frac_Sdd11 = m11_arr.mean(axis=0)
    valid_frac_Sdd21 = m21_arr.mean(axis=0)

    # Plot 4: Per-frequency forward-model error
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(freqs_ghz, mae_Sdd11_per_freq, color="C0", lw=1.5,
                  label="|Sdd11| MAE")
    axes[0].plot(freqs_ghz, mae_Sdd21_per_freq, color="C1", lw=1.5,
                  label="|Sdd21| MAE")
    axes[0].axhline(2.0, color="g", ls=":", lw=0.8, label="2 dB target")
    axes[0].set_ylabel("Per-frequency MAE [dB]")
    axes[0].set_title("Forward-model error vs frequency (validation set, "
                       "masked to passband)")
    axes[0].legend()

    axes[1].plot(freqs_ghz, valid_frac_Sdd11 * 100, color="C0", lw=1.0,
                  label="fraction of |Sdd11| samples above noise floor")
    axes[1].plot(freqs_ghz, valid_frac_Sdd21 * 100, color="C1", lw=1.0,
                  label="fraction of |Sdd21| samples above noise floor")
    axes[1].set_ylabel("Valid sample fraction [%]")
    axes[1].set_xlabel("Frequency [GHz]")
    axes[1].set_title("How much of the dataset contributes to the MAE at each "
                       "frequency (passband validity)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(EDA_OUT_DIR / "04_forward_model_per_freq_mae.png",
                 bbox_inches="tight")
    plt.close()
    print(f"Saved: 04_forward_model_per_freq_mae.png")

    # Plot 5: Model error vs data std (irreducible error proxy)
    # If MAE >> std, the model is failing relative to natural variation.
    # If MAE ~ std/N, the model is near the ceiling for what's learnable.
    fig, ax = plt.subplots(1, 1, figsize=(12, 4.5))
    ratio_S11 = mae_Sdd11_per_freq / (stats_Sdd11_val["std"] + 1e-6)
    ratio_S21 = mae_Sdd21_per_freq / (stats_Sdd21_val["std"] + 1e-6)
    ax.plot(freqs_ghz, ratio_S11, color="C0", lw=1.5,
             label="Sdd11: MAE / data std")
    ax.plot(freqs_ghz, ratio_S21, color="C1", lw=1.5,
             label="Sdd21: MAE / data std")
    ax.axhline(1.0, color="k", ls="--", lw=0.6,
                label="MAE = data std (model no better than mean)")
    ax.axhline(0.1, color="g", ls=":", lw=0.6,
                label="MAE = 10% of std (very strong fit)")
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Error / variability ratio")
    ax.set_title("Model error normalized to dataset variability per frequency "
                  "(low = good)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(EDA_OUT_DIR / "05_model_error_vs_data_variability.png",
                 bbox_inches="tight")
    plt.close()
    print(f"Saved: 05_model_error_vs_data_variability.png")
else:
    # placeholders so later sections can run
    mae_Sdd11_per_freq = None
    mae_Sdd21_per_freq = None
    valid_frac_Sdd11 = None
    valid_frac_Sdd21 = None


# %% Section 6: Noise floor verification
# Currently using -45 dB. Let's see if the data justifies it or suggests revision.

print("\n=== Section 6: Noise floor verification ===")
sample_freqs_ghz = [5, 14, 28, 42, 56, 75, 95]
sample_indices = [int(np.argmin(np.abs(freqs_ghz - f))) for f in sample_freqs_ghz]

fig, axes = plt.subplots(2, len(sample_freqs_ghz), figsize=(3.0 * len(sample_freqs_ghz), 6),
                          sharex=False, sharey=True)
for col, (fghz, fidx) in enumerate(zip(sample_freqs_ghz, sample_indices)):
    # Sdd11 distribution at this freq
    ax = axes[0, col]
    ax.hist(Sdd11_db_train[:, fidx], bins=60, color="C0", alpha=0.85)
    ax.axvline(NOISE_FLOOR_DB, color="r", ls=":", lw=1.0)
    ax.set_title(f"f = {fghz} GHz", fontsize=10)
    if col == 0:
        ax.set_ylabel("|Sdd11| samples")
    ax.set_xlabel("|Sdd11| [dB]")

    # Sdd21 distribution at this freq
    ax = axes[1, col]
    ax.hist(Sdd21_db_train[:, fidx], bins=60, color="C1", alpha=0.85)
    ax.axvline(NOISE_FLOOR_DB, color="r", ls=":", lw=1.0,
                label=f"noise floor {NOISE_FLOOR_DB} dB" if col == 0 else None)
    if col == 0:
        ax.set_ylabel("|Sdd21| samples")
        ax.legend(fontsize=8)
    ax.set_xlabel("|Sdd21| [dB]")

fig.suptitle("Distribution of |S| at representative frequencies (red dotted = "
              "current noise floor)", fontsize=11)
plt.tight_layout()
plt.savefig(EDA_OUT_DIR / "06_noise_floor_histograms.png", bbox_inches="tight")
plt.close()
print(f"Saved: 06_noise_floor_histograms.png")

# Quantitative noise-floor diagnosis: at each frequency, what fraction of samples
# fall below -40, -45, -50 dB? If a band has many samples piling below the floor,
# the floor is meaningful; if not, the floor is too low (we're losing valid signal).
for thresh in [-40, -45, -50, -55]:
    below11 = (Sdd11_db_train < thresh).mean(axis=0)
    below21 = (Sdd21_db_train < thresh).mean(axis=0)
    print(f"  Samples with |S| < {thresh} dB: "
          f"|Sdd11| {below11.mean()*100:.2f}% (peak {below11.max()*100:.2f}%)  "
          f"|Sdd21| {below21.mean()*100:.2f}% (peak {below21.max()*100:.2f}%)")


# %% Section 7: 112G PAM4 band definitions and overlay
# Now we anchor everything to the industry standard.

print("\n=== Section 7: 112G PAM4 band overlay and weighting proposal ===")

# Industry-aligned 5-band scheme for 112G PAM4
PAM4_BANDS = OrderedDict([
    ("baseband",   {"f_lo": 0.0,   "f_hi": 14.0,  "weight": 3.0,
                     "label": "0-14 GHz (baseband)",
                     "rationale": "Sub-Nyquist; DC return loss and low-freq matching"}),
    ("nyquist",    {"f_lo": 14.0,  "f_hi": 28.0,  "weight": 5.0,
                     "label": "14-28 GHz (Nyquist)",
                     "rationale": "112G PAM4 Nyquist; dominant PAM4 eye energy"}),
    ("channel_bw", {"f_lo": 28.0,  "f_hi": 42.0,  "weight": 3.0,
                     "label": "28-42 GHz (channel BW)",
                     "rationale": "0.75 x symbol rate; first-harmonic / ISI region"}),
    ("harmonics",  {"f_lo": 42.0,  "f_hi": 56.0,  "weight": 1.5,
                     "label": "42-56 GHz (harmonics)",
                     "rationale": "2 x Nyquist; secondary jitter / FEXT contribution"}),
    ("out_of_band",{"f_lo": 56.0,  "f_hi": 100.0, "weight": 0.5,
                     "label": "56-100 GHz (out of band)",
                     "rationale": "Beyond 112G PAM4 spec; train weakly, do not "
                                   "optimize against"}),
])


def freq_mask(f_lo, f_hi):
    return (freqs_ghz >= f_lo) & (freqs_ghz < f_hi)


def freq_weights_vector(band_dict):
    w = np.zeros(F_LEN, dtype=np.float64)
    for b in band_dict.values():
        w[freq_mask(b["f_lo"], b["f_hi"])] = b["weight"]
    # Edge: last point goes to the highest band
    w[freqs_ghz >= max(b["f_hi"] for b in band_dict.values())] = list(band_dict.values())[-1]["weight"]
    return w


band_weights = freq_weights_vector(PAM4_BANDS)
print("Proposed 112G PAM4 band scheme (starting point; data may revise these):")
for name, b in PAM4_BANDS.items():
    print(f"  {b['label']:30s}  weight={b['weight']:.1f}  -- {b['rationale']}")

# Plot 7: Comprehensive overlay - magnitude envelope + model error + bands
fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
band_colors = ["#FFE6CC", "#FFB347", "#FFD96A", "#A6E22E", "#7FDBFF"]


def draw_bands(ax):
    for (name, b), col in zip(PAM4_BANDS.items(), band_colors):
        ax.axvspan(b["f_lo"], b["f_hi"], alpha=0.30, color=col,
                    label=f"{b['label']} w={b['weight']:.1f}")


# Panel 1: median Sdd11 / Sdd21 with bands
ax = axes[0]
draw_bands(ax)
ax.plot(freqs_ghz, stats_Sdd11_train["median"], color="navy", lw=1.5,
         label="median |Sdd11|")
ax.plot(freqs_ghz, stats_Sdd21_train["median"], color="darkred", lw=1.5,
         label="median |Sdd21|")
ax.axhline(NOISE_FLOOR_DB, color="gray", ls=":", lw=0.8)
ax.set_ylabel("Median |S| [dB]")
ax.set_title("Where the signal lives, vs 112G PAM4 bands")
ax.legend(loc="lower left", fontsize=7, ncol=2)
ax.set_ylim(bottom=-60)

# Panel 2: deep-null density with bands
ax = axes[1]
draw_bands(ax)
ax.plot(freqs_ghz, null_counts_Sdd11[-30], color="navy", lw=1.2,
         label="|Sdd11| < -30 dB count")
ax.plot(freqs_ghz, null_counts_Sdd21[-30], color="darkred", lw=1.2,
         label="|Sdd21| < -30 dB count")
ax.set_ylabel("Deep-null sample count")
ax.set_title("Where the rare features (deep nulls) cluster")
ax.legend(loc="upper left", fontsize=7)

# Panel 3: per-frequency MAE if available; otherwise per-frequency std
ax = axes[2]
draw_bands(ax)
if mae_Sdd11_per_freq is not None:
    ax.plot(freqs_ghz, mae_Sdd11_per_freq, color="navy", lw=1.5,
             label="Sdd11 MAE")
    ax.plot(freqs_ghz, mae_Sdd21_per_freq, color="darkred", lw=1.5,
             label="Sdd21 MAE")
    ax.axhline(2.0, color="g", ls=":", lw=0.8, label="2 dB target")
    ax.set_ylabel("Forward model MAE [dB]")
    ax.set_title("Where the forward model errs the most")
else:
    ax.plot(freqs_ghz, stats_Sdd11_train["std"], color="navy", lw=1.5,
             label="Sdd11 std")
    ax.plot(freqs_ghz, stats_Sdd21_train["std"], color="darkred", lw=1.5,
             label="Sdd21 std")
    ax.set_ylabel("Per-frequency std [dB]")
    ax.set_title("Where the dataset has variability (model not loaded)")
ax.legend(loc="upper right", fontsize=7)
ax.set_xlabel("Frequency [GHz]")
plt.tight_layout()
plt.savefig(EDA_OUT_DIR / "07_industry_band_overlay.png", bbox_inches="tight")
plt.close()
print(f"Saved: 07_industry_band_overlay.png")


# %% Section 8: Final band weighting recommendation
# Combine the data analysis with industry priors to produce a defensible weighting.

print("\n=== Section 8: Final band weight recommendation ===")
print("\nPer-band averaged statistics (training set unless noted):")
print(f"{'Band':30s}  {'med S11':>9s}  {'med S21':>9s}  {'std S11':>9s}"
      f"  {'std S21':>9s}  {'model MAE S11':>14s}  {'model MAE S21':>14s}")
print("-" * 110)

band_stats_summary = OrderedDict()
for name, b in PAM4_BANDS.items():
    mask = freq_mask(b["f_lo"], b["f_hi"])
    med_S11 = stats_Sdd11_train["median"][mask].mean()
    med_S21 = stats_Sdd21_train["median"][mask].mean()
    std_S11 = stats_Sdd11_train["std"][mask].mean()
    std_S21 = stats_Sdd21_train["std"][mask].mean()
    mae_S11 = (mae_Sdd11_per_freq[mask].mean() if mae_Sdd11_per_freq is not None
                else float("nan"))
    mae_S21 = (mae_Sdd21_per_freq[mask].mean() if mae_Sdd21_per_freq is not None
                else float("nan"))
    band_stats_summary[name] = {
        "label": b["label"],
        "weight_proposed": b["weight"],
        "median_Sdd11_db": float(med_S11),
        "median_Sdd21_db": float(med_S21),
        "std_Sdd11_db": float(std_S11),
        "std_Sdd21_db": float(std_S21),
        "model_mae_Sdd11_db": float(mae_S11) if not np.isnan(mae_S11) else None,
        "model_mae_Sdd21_db": float(mae_S21) if not np.isnan(mae_S21) else None,
        "rationale": b["rationale"],
        "f_lo": b["f_lo"], "f_hi": b["f_hi"],
    }
    print(f"  {b['label']:30s}  {med_S11:>9.2f}  {med_S21:>9.2f}  {std_S11:>9.2f}"
          f"  {std_S21:>9.2f}  {mae_S11:>14.2f}  {mae_S21:>14.2f}")

# Save the band scheme and the per-frequency weight vector as JSON + .npy
band_output = {
    "noise_floor_db": NOISE_FLOOR_DB,
    "n_freq_points": int(F_LEN),
    "freq_ghz_min": float(freqs_ghz.min()),
    "freq_ghz_max": float(freqs_ghz.max()),
    "target_standard": "112G PAM4 (IEEE 802.3ck / OIF-CEI-112G-PAM4)",
    "nyquist_ghz": NYQUIST_112G_GHZ,
    "channel_bw_ghz": CHANNEL_BW_GHZ,
    "bands": band_stats_summary,
    "notes": (
        "Weights initialized from 112G PAM4 spec, then sanity-checked against "
        "per-band model error and dataset variability. Bands are mutually "
        "exclusive and cover the full 0-100 GHz range. Reuse this scheme for "
        "the Link dataset and for the inverse / TTO band-weighted loss."
    ),
}

with open(EDA_OUT_DIR / "band_weights.json", "w") as f:
    json.dump(band_output, f, indent=2)
np.save(EDA_OUT_DIR / "band_weights_per_freq.npy", band_weights)
print(f"\nSaved: {EDA_OUT_DIR / 'band_weights.json'}")
print(f"Saved: {EDA_OUT_DIR / 'band_weights_per_freq.npy'}")

# Plot 8: final summary figure
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
ax = axes[0]
draw_bands(ax)
ax.plot(freqs_ghz, band_weights, color="black", lw=2.5,
         label="band weight w(f)")
ax.set_ylim(0, max(b["weight"] for b in PAM4_BANDS.values()) * 1.2)
ax.set_ylabel("Per-frequency weight w(f)")
ax.set_xlabel("Frequency [GHz]")
ax.set_title("Final 112G-aligned weight vector (used in inverse loss / TTO)")
ax.legend(loc="upper right", fontsize=7, ncol=2)

# Table of band stats
ax = axes[1]
ax.axis("off")
table_data = []
for name, st in band_stats_summary.items():
    table_data.append([
        st["label"],
        f"{st['weight_proposed']:.1f}",
        f"{st['median_Sdd11_db']:.1f}",
        f"{st['median_Sdd21_db']:.1f}",
        f"{st['model_mae_Sdd11_db']:.2f}" if st["model_mae_Sdd11_db"] else "n/a",
        f"{st['model_mae_Sdd21_db']:.2f}" if st["model_mae_Sdd21_db"] else "n/a",
    ])
columns = ["Band", "w", "med S11 (dB)", "med S21 (dB)",
            "MAE S11 (dB)", "MAE S21 (dB)"]
tbl = ax.table(cellText=table_data, colLabels=columns,
                loc="center", cellLoc="center", colLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.0, 1.5)
ax.set_title("Per-band summary")

plt.tight_layout()
plt.savefig(EDA_OUT_DIR / "08_final_band_recommendation.png",
             bbox_inches="tight")
plt.close()
print(f"Saved: 08_final_band_recommendation.png")

print("\n=== EDA complete ===")
print(f"All outputs in: {EDA_OUT_DIR}")
print("Files:")
for p in sorted(EDA_OUT_DIR.iterdir()):
    print(f"   {p.name}")
