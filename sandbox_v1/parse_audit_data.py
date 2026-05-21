import json
import pathlib

CELLS: list[tuple[str, str]] = []

def md(text: str) -> None:
    CELLS.append(("markdown", text))

def code(text: str) -> None:
    CELLS.append(("code", text))

# =============================================================================
md("""# Parsed-data audit: `diff_pair_dataset.pt`

This notebook validates the per-differential-pair dataset produced by `parse_diff_pairs.py`. We perform a series of physical and mathematical sanity checks to ensure the data is ready for the Neural Network.

### Audit Plan:
1. **Schema & Integrity**: Verify tensor shapes and check for NaNs/Infs.
2. **Within-Sim Variance**: Prove that diff-pairs on the same board are distinct via Context Vectors.
3. **Cross-Sim Diversity**: Overlay traces from different geometries to ensure dataset breadth.
4. **Aggregate Distributions**: Check S-parameter statistics at DC, 25, and 50 GHz.
5. **Denormalization Check**: Physically verify $X \\rightarrow \\text{Norm} \\rightarrow \\text{Denorm} \\rightarrow X$ matches the raw CSV.
""")

# =============================================================================
code('''# --- Imports and paths -------------------------------------------------------
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# >>> CHANGE TO "Link" OR "Array" AS NEEDED <<<
DATASET = "Array"

# Setup project root and relative paths
# This allows the notebook to find the utils folder if you moved it to the root
PROJ_ROOT = Path.cwd().parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

PT_PATH   = PROJ_ROOT / "data" / "processed" / f"Universal-Diff-SI-{DATASET}" / "diff_pair_dataset.pt"
RAW_DIR   = PROJ_ROOT / "data" / "raw"       / f"Universal-Diff-SI-{DATASET}"
SKIP_CSV  = PROJ_ROOT / "data" / "processed" / f"Universal-Diff-SI-{DATASET}" / "skipped_pairs.csv"
FIG_DIR   = PROJ_ROOT / "results" / "figures" / "audit"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Project Root:      {PROJ_ROOT}")
print(f"Targeting Dataset: {DATASET}")
print(f"Loading .pt from:  {PT_PATH.relative_to(PROJ_ROOT)}")
''')

# =============================================================================
md("""## 1. Schema Inspection & Data Integrity
Before looking at the physics, we must ensure the tensors are complete and free of numerical corruption.""")

code('''# --- Load .pt and run NaN/Inf Audit ------------------------------------------
data = torch.load(PT_PATH, weights_only=False)

print(f"Metadata Summary:")
print(f"  Dataset Type: {data['dataset_type']}")
print(f"  Total Pairs:  {data['X_local'].shape[0]}")
print(f"  Unique Sims:  {len(set(data['sim_ids']))}\\n")

print(f"Numerical Integrity Audit:")
for k in ("X_local", "X_global", "X_context", "Y_real", "Y_imag"):
    t = data[k]
    nans = torch.isnan(t).sum().item()
    infs = torch.isinf(t).sum().item()
    status = "PASS" if nans == 0 and infs == 0 else "!!! FAIL !!!"
    print(f"  {k:12s} | NaNs: {nans:5d} | Infs: {infs:5d} | Status: {status}")
''')

# =============================================================================
md("""## 2. Within-Sim Variance & Context Vector Proof
We select 3 random simulations. For each, we plot all its diff-pairs in a 3x2 grid. We then print the variance of the context features to prove that our augmentation strategy is capturing spatial boundary conditions.""")

code('''# --- Within-Sim Comparison (3x2 Grid) -----------------------------------------
sim_ids_arr = np.array(data["sim_ids"])
pair_ids    = data["pair_ids"].numpy()
unique_sims, counts = np.unique(sim_ids_arr, return_counts=True)
big_sims = unique_sims[counts >= 4]

rng = np.random.default_rng(42)
chosen = rng.choice(big_sims, size=min(3, len(big_sims)), replace=False)
freqs = data["frequencies"].numpy() / 1e9

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

for i, sim_id in enumerate(chosen):
    mask = sim_ids_arr == sim_id
    S = data["Y_real"][mask].numpy() + 1j * data["Y_imag"][mask].numpy()
    X_ctx = data["X_context"][mask].numpy()
    n_pairs = S.shape[0]
    
    # 1. Print Variance Proof
    ctx_var = X_ctx.var(axis=0)
    print(f"Sim {sim_id} [n={n_pairs}]: Context Variance detected in:")
    for f_idx, feat in enumerate(data["context_features"]):
        if ctx_var[f_idx] > 1e-5:
            print(f"  - {feat:28s} Var: {ctx_var[f_idx]:.4f}")

    # 2. Plot Return vs Insertion Loss
    for k in range(n_pairs):
        p_id = pair_ids[mask][k]
        axes[i, 0].plot(freqs, 20 * np.log10(np.abs(S[k, :, 0, 0]) + 1e-12), alpha=0.6, label=f"P{p_id}")
        axes[i, 1].plot(freqs, 20 * np.log10(np.abs(S[k, :, 1, 0]) + 1e-12), alpha=0.6, label=f"P{p_id}")
        
    axes[i, 0].set_title(f"Sim {sim_id}: Sdd11 (Return Loss)"); axes[i, 0].grid(alpha=0.3)
    axes[i, 1].set_title(f"Sim {sim_id}: Sdd21 (Insertion Loss)"); axes[i, 1].grid(alpha=0.3)
    axes[i, 0].legend(fontsize=7, loc='lower right', ncol=2)

fig.tight_layout()
fig.savefig(FIG_DIR / f"{DATASET}_within_sim_audit.png", dpi=150)
plt.show()
''')

# =============================================================================
md("""## 3. Aggregate Distributions (DC, 25 GHz, 50 GHz)
We check the statistical health of every pair in the dataset using a 2x3 histogram grid.""")

code('''# --- Aggregate Distributions (2x3 Grid) ---------------------------------------
target_fs = [0.25, 25.0, 50.0]
f_idxs = [int(np.argmin(np.abs(freqs - f))) for f in target_fs]

Sdd21_all = (data["Y_real"][:, :, 1, 0] + 1j * data["Y_imag"][:, :, 1, 0]).numpy()
Sdd11_all = (data["Y_real"][:, :, 0, 0] + 1j * data["Y_imag"][:, :, 0, 0]).numpy()

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for j, (f, fi) in enumerate(zip(target_fs, f_idxs)):
    s21_db = 20 * np.log10(np.abs(Sdd21_all[:, fi]) + 1e-12)
    s11_db = 20 * np.log10(np.abs(Sdd11_all[:, fi]) + 1e-12)
    
    axes[0, j].hist(s21_db, bins=60, color="C0", alpha=0.7)
    axes[0, j].set_title(f"Sdd21 at {f} GHz\\nMean: {s21_db.mean():.2f} dB"); axes[0, j].grid(alpha=0.3)
    axes[0, j].axvline(0, color="red", linestyle="--", alpha=0.5)
    
    axes[1, j].hist(s11_db, bins=60, color="C3", alpha=0.7)
    axes[1, j].set_title(f"Sdd11 at {f} GHz\\nMean: {s11_db.mean():.2f} dB"); axes[1, j].grid(alpha=0.3)
    axes[1, j].axvline(0, color="red", linestyle="--", alpha=0.5)

fig.suptitle(f"Aggregate Magnitude Distributions: {DATASET} Dataset", fontsize=14)
fig.tight_layout()
plt.show()
''')

# =============================================================================
md("""## 4. De-normalisation Sanity Check
We pick Sample 0, denormalize it, and compare it against the raw `parameter.csv`.""")

code('''# --- Denorm Verification -----------------------------------------------------
sample_idx = 0
sim_id = data["sim_ids"][sample_idx]
log_feats = set(data["log_features"])

def denorm(vec, m, s, names):
    phys = vec * s + m
    for i, name in enumerate(names):
        if name in log_feats: phys[i] = 10**phys[i]
    return phys

x_loc_phys = denorm(data["X_local"][sample_idx].numpy(), data["X_local_mean"].numpy(), data["X_local_std"].numpy(), data["local_features"])
x_glo_phys = denorm(data["X_global"][sample_idx].numpy(), data["X_global_mean"].numpy(), data["X_global_std"].numpy(), data["global_features"])

# Load raw CSV for ground truth
df_raw = pd.read_csv(RAW_DIR / "parameter.csv")
if "LOSTANGENT" in df_raw.columns: df_raw = df_raw.rename(columns={"LOSTANGENT": "LOSSTANGENT"})
row_gt = df_raw[df_raw["SIMULATION"] == sim_id].iloc[0]

print(f"{'Feature':<22} | {'Recovered':>12} | {'CSV Ground Truth':>16} | {'Match'}")
print("-" * 75)
for i, name in enumerate(data["local_features"]):
    match = "OK" if np.isclose(x_loc_phys[i], row_gt[name], rtol=1e-3) else "FAIL"
    print(f"{name:<22} | {x_loc_phys[i]:12.6g} | {row_gt[name]:16.6g} | {match}")
for i, name in enumerate(data["global_features"]):
    match = "OK" if np.isclose(x_glo_phys[i], row_gt[name], rtol=1e-3) else "FAIL"
    print(f"{name:<22} | {x_glo_phys[i]:12.6g} | {row_gt[name]:16.6g} | {match}")
''')

# =============================================================================
def to_cell(cell_type: str, source: str) -> dict:
    return {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True), **({"execution_count": None, "outputs": []} if cell_type == "code" else {})}

notebook = {
    "cells": [to_cell(t, s) for t, s in CELLS],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}},
    "nbformat": 4, "nbformat_minor": 5,
}

sandbox_root = pathlib.Path(__file__).resolve().parent
project_root = sandbox_root.parent
out_path = project_root / "notebooks" / "03_parsed_data_audit.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(notebook, indent=1))
print(f"Wrote {out_path} ({len(CELLS)} cells)")