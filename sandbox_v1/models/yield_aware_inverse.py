"""
tto_yield_inverse.py
--------------------------------------------------
YIELD-AWARE latent-space TTO with a RANKED design portfolio.
The final methodological stage of the inverse-design pipeline (B3).

WHAT THIS ADDS ON TOP OF tto_latent_inverse.py
  1. A manufacturing TOLERANCE MODEL: per-parameter perturbation widths
     sigma_i, expressed in normalized feature units (features are z-scored,
     so sigma_norm = tol_frac means "tol_frac x the population spread of that
     parameter"). The script prints the PHYSICAL meaning of each sigma using
     the dataset's X_local_std and log_features, so the thesis can report
     e.g. "via radius +/- 0.4 mil (1 sigma)".
  2. A DIFFERENTIABLE SENSITIVITY PENALTY (the yield proxy):
         sens(x) = sum_i  || W * ( S(x + sigma_i e_i) - S(x) ) ||^2
     i.e. displace each parameter by exactly one tolerance sigma and penalize
     the spec-weighted response deviation. To first order this equals
     sum_i sigma_i^2 ||dS_w/dx_i||^2 -- the tolerance-scaled Jacobian norm
     (Hoffman et al. 2019, arXiv:1908.02729, transplanted from classifier
     robustness to manufacturing robustness) -- but it needs NO double
     backward: all 9 geometries (nominal + 8 displaced) go through the
     surrogate in ONE batched forward pass, and ordinary autograd
     differentiates it w.r.t. the latent z. It also captures curvature that
     the pure first-order Jacobian misses (honest bonus, note in thesis).
  3. MONTE-CARLO YIELD ESTIMATION through the surrogate:
         yield(x) = P_delta[ spec( S(x + delta) ) ],  delta ~ N(0, diag(sigma^2))
     with N samples batched on GPU and a Wilson confidence interval.
     Spec (eye band 0-28 GHz, both must hold at every eye-band frequency):
        insertion loss   : Sdd21_dB >= Sdd21_target_dB - spec_il_margin
        return loss      : Sdd11_dB <= spec_rl_max
     The spec is target-relative for IL (a design should not lose more than
     `spec_il_margin` dB relative to what it nominally promises) and absolute
     for RL. Both are CLI-configurable and must be frozen before the sweep.
  4. THE RANKED PORTFOLIO (the one-to-many property as a feature):
     every restart's converged design is KEPT as a candidate. Candidates are
     first GATED on nominal fit (eye-band mean |dB error| vs target <=
     fit_gate; a robustly-wrong design is useless), then RANKED top-to-bottom
     by MC yield. Output: ranked CSV, a fit-vs-yield Pareto scatter, one npz
     per candidate (stage-07 compatible), and the top-ranked design saved
     separately for immediate OpenEMS verification.

REFERENCES
  latent manifold search . Gomez-Bombarelli et al. 2018, arXiv:1610.02415
  z prior reliability .... Notin et al., NeurIPS 2021, arXiv:2107.00096
  OOD boundary loss ...... Ren, Padilla & Malof, NeurIPS 2020, arXiv:2009.12919
  TTO framing ............ LaBash et al. 2025, arXiv:2505.18188
  Jacobian robustness .... Hoffman et al. 2019, arXiv:1908.02729
  robust-by-perturbation . Wang, Lazarov & Sigmund 2011 (Struct Multidisc Optim)
  curriculum ............. Bengio et al., ICML 2009

USAGE
  # yield-aware portfolio for one target (fast; surrogate only):
  python3 tto_yield_inverse.py --samples 0 --restarts 12 --lambda-j 0.1

  # lambda sweep for the Pareto front (run repeatedly, results are tagged):
  for L in 0 0.01 0.1 1.0; do
      python3 tto_yield_inverse.py --samples 0 --lambda-j $L
  done

  # then stage 07 on the top-ranked design:
  validate_model_generated.py --designs .../design_sample_0_yield_L0.1_top.npz --check-only
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# PATHS (identical to tto_latent_inverse.py)
# =============================================================================
PROJECT_ROOT = Path.home() / "mece_project_inverse_model" / "Generative_Inverse_Design_of_High-Speed_Interconnects"
DATA_PT = PROJECT_ROOT / "data" / "processed" / "Universal-Diff-SI-Array" / "diff_pair_dataset.pt"
WEIGHT_TENSOR_PATH = PROJECT_ROOT / "sandbox_v1" / "data" / "frequency_eda" / "weights_element_aware_per_freq.npy"
FORWARD_MODEL_PATH = PROJECT_ROOT / "sandbox_v1" / "models" / "forward_runs" / "run_2026-06-01_213554_direct_resnet_array" / "checkpoint_best.pt"
INVERSE_RUNS_DIR = PROJECT_ROOT / "sandbox_v1" / "models" / "inverse_runs"
OUT_DIR = PROJECT_ROOT / "sandbox_v1" / "models" / "evaluation_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
_THIS_DIR = Path(__file__).resolve().parent


def _find_cvae_checkpoint():
    runs = sorted(INVERSE_RUNS_DIR.glob("run_*_inverse_element_aware_*"))
    if not runs:
        raise FileNotFoundError(f"no cVAE run under {INVERSE_RUNS_DIR}")
    return runs[-1] / "cvae_element_aware_best.pt"


# =============================================================================
# ARCHITECTURES -- identical attribute names to the trained checkpoints
# =============================================================================
class Conv1DResBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.10):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2, padding_mode='replicate')
        self.norm1 = nn.GroupNorm(4, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2, padding_mode='replicate')
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

    def __init__(self, d_local=8, d_global=6, d_context=7, F_len=401, hidden_dim=384, n_blocks=8):
        super().__init__()
        self.F_len = F_len
        self.is_link_dataset = (d_global == 7)
        in_dim = d_local + d_global + d_context
        self.geom_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.learnable_velocity = nn.Parameter(torch.tensor(10.0))
        self.proj_in = nn.Conv1d(hidden_dim + 3, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([Conv1DResBlock(hidden_dim, dropout=0.10) for _ in range(n_blocks)])
        self.proj_out = nn.Conv1d(hidden_dim, 20, kernel_size=1)
        self.register_buffer("upper_r", torch.tensor(self.UPPER_R, dtype=torch.long))
        self.register_buffer("upper_c", torch.tensor(self.UPPER_C, dtype=torch.long))

    def _scatter_symmetric(self, vec_real, vec_imag, B):
        mat = torch.zeros((B, 4, 4, self.F_len), dtype=torch.float64, device=vec_real.device)
        mat_i = torch.zeros((B, 4, 4, self.F_len), dtype=torch.float64, device=vec_real.device)
        mat[:, self.upper_r, self.upper_c, :] = vec_real.to(torch.float64)
        mat[:, self.upper_c, self.upper_r, :] = vec_real.to(torch.float64)
        mat_i[:, self.upper_r, self.upper_c, :] = vec_imag.to(torch.float64)
        mat_i[:, self.upper_c, self.upper_r, :] = vec_imag.to(torch.float64)
        S = torch.complex(mat, mat_i)
        return S.permute(0, 3, 1, 2)

    def forward(self, x_local, x_global, x_context, freqs_hz):
        B = x_local.shape[0]
        x = torch.cat([x_local, x_global, x_context], dim=-1)
        h_geom = self.geom_mlp(x)
        h_seq = h_geom.unsqueeze(-1).expand(-1, -1, self.F_len)
        f_norm = (freqs_hz / freqs_hz.max()).view(1, 1, self.F_len).expand(B, 1, -1).to(h_seq.dtype)
        length_feature = (x_global[:, -1].view(B, 1, 1) if self.is_link_dataset
                          else torch.zeros((B, 1, 1), device=x.device, dtype=h_seq.dtype))
        phase = self.learnable_velocity * f_norm * length_feature
        h = torch.cat([h_seq, f_norm, torch.sin(phase), torch.cos(phase)], dim=1)
        h = self.proj_in(h)
        for block in self.blocks:
            h = block(h)
        out = self.proj_out(h)
        return self._scatter_symmetric(out[:, :10, :], out[:, 10:, :], B)


class SConditionEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(20, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(128, out_dim)

    def forward(self, S_complex):
        B = S_complex.shape[0]
        s_real = S_complex[:, :, DirectSequenceResNet.UPPER_R, DirectSequenceResNet.UPPER_C].real.permute(0, 2, 1).float()
        s_imag = S_complex[:, :, DirectSequenceResNet.UPPER_R, DirectSequenceResNet.UPPER_C].imag.permute(0, 2, 1).float()
        return self.fc(self.conv(torch.cat([s_real, s_imag], dim=1)).view(B, 128))


class Tandem_cVAE(nn.Module):
    def __init__(self, d_local=8, d_global=6, d_context=7, latent_dim=16, cond_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.s_encoder = SConditionEncoder(out_dim=cond_dim)
        total_cond_dim = cond_dim + d_global + d_context
        self.enc_mlp = nn.Sequential(nn.Linear(d_local + total_cond_dim, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU())
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.dec_mlp = nn.Sequential(
            nn.Linear(latent_dim + total_cond_dim, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, d_local))

    def forward(self, x_local, x_global, x_context, S_target):
        cond = torch.cat([self.s_encoder(S_target), x_global, x_context], dim=1)
        h = self.enc_mlp(torch.cat([x_local, cond], dim=1))
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return self.dec_mlp(torch.cat([z, cond], dim=1)), mu, logvar

    def build_condition(self, S_target, x_global, x_context):
        return torch.cat([self.s_encoder(S_target), x_global, x_context], dim=1)

    def decode(self, z, cond):
        return self.dec_mlp(torch.cat([z, cond], dim=1))


# =============================================================================
# LOSS PIECES
# =============================================================================
def weighted_s_loss(S_pred, S_tgt, w_bcast):
    dr = (S_pred.real.float() - S_tgt.real.float()).pow(2)
    di = (S_pred.imag.float() - S_tgt.imag.float()).pow(2)
    return (dr * w_bcast).mean() + (di * w_bcast).mean()


def boundary_loss(x_norm, margin=2.5):
    return F.relu(x_norm.abs() - margin).pow(2).sum()


def curriculum_mask(freqs_hz, step, total_steps, device):
    frac = step / max(total_steps - 1, 1)
    f_hi = 28e9 if frac < 1 / 3 else (56e9 if frac < 2 / 3 else float("inf"))
    return (freqs_hz <= f_hi).to(torch.float32).view(1, -1, 1, 1).to(device)


def sensitivity_penalty(fwd, x, xg, xc, freqs_hz, sigma, w_bcast, S_nom=None):
    """Tolerance-displaced sensitivity = differentiable yield proxy.

    Builds a batch [x, x + sigma_1 e_1, ..., x + sigma_8 e_8] (9 designs),
    runs ONE forward pass, and returns
        sum_i weighted_MSE( S(x + sigma_i e_i), S(x) ).
    First-order equal to sum_i sigma_i^2 ||dS_w/dx_i||^2 (Jacobian proxy),
    differentiable by ordinary backprop (no double backward).
    """
    d = x.shape[1]
    X = x.repeat(d + 1, 1)
    for i in range(d):
        X[i + 1, i] = X[i + 1, i] + sigma[i]
    S_all = fwd(X, xg.repeat(d + 1, 1), xc.repeat(d + 1, 1), freqs_hz)
    S0 = S_all[0:1]
    pen = 0.0
    for i in range(d):
        pen = pen + weighted_s_loss(S_all[i + 1:i + 2], S0, w_bcast)
    return pen


# =============================================================================
# MONTE-CARLO YIELD (evaluation metric -- through the surrogate)
# =============================================================================
@torch.no_grad()
def mc_yield(fwd, x, xg, xc, freqs_hz, sigma, S_tgt, eye_mask,
             spec_il_margin=1.0, spec_rl_max=-8.0, n_mc=2000, batch=256,
             device="cpu"):
    """yield = fraction of tolerance-perturbed designs meeting the eye-band
    spec. Spec per MC sample (must hold at EVERY eye-band frequency):
        Sdd21_dB >= Sdd21_target_dB - spec_il_margin   (target-relative IL)
        Sdd11_dB <= spec_rl_max                        (absolute RL ceiling)
    Returns (yield, wilson_lo, wilson_hi)."""
    tgt21 = 20 * torch.log10(S_tgt[0, :, 1, 0].abs() + 1e-12)  # (F,)
    passes = 0
    done = 0
    while done < n_mc:
        b = min(batch, n_mc - done)
        delta = torch.randn(b, x.shape[1], device=device) * sigma.view(1, -1)
        Xp = x.repeat(b, 1) + delta
        S = fwd(Xp, xg.repeat(b, 1), xc.repeat(b, 1), freqs_hz)
        s21 = 20 * torch.log10(S[:, :, 1, 0].abs() + 1e-12)
        s11 = 20 * torch.log10(S[:, :, 0, 0].abs() + 1e-12)
        ok21 = (s21[:, eye_mask] >= (tgt21[eye_mask].view(1, -1) - spec_il_margin)).all(dim=1)
        ok11 = (s11[:, eye_mask] <= spec_rl_max).all(dim=1)
        passes += (ok21 & ok11).sum().item()
        done += b
    p = passes / n_mc
    # Wilson 95% interval
    zc = 1.96
    den = 1 + zc * zc / n_mc
    ctr = (p + zc * zc / (2 * n_mc)) / den
    hw = zc * np.sqrt(p * (1 - p) / n_mc + zc * zc / (4 * n_mc * n_mc)) / den
    return p, max(0.0, ctr - hw), min(1.0, ctr + hw)


# =============================================================================
# MAIN PER-TARGET ROUTINE
# =============================================================================
def run_yield_tto(sample_idx, tto_steps=150, lr=0.05, restarts=12,
                  lambda_prior=1e-3, lambda_bnd=1e-2, lambda_j=0.1,
                  curriculum=False, tol_frac=0.10, n_mc=2000,
                  spec_il_margin=1.0, spec_rl_max=-8.0, fit_gate=3.0,
                  seed=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    payload = torch.load(DATA_PT, map_location=device, weights_only=False)
    freqs_hz = payload["frequencies"].to(device)
    freqs_ghz = freqs_hz.cpu().numpy() / 1e9
    eye_mask = (freqs_hz <= 28e9)

    w = torch.from_numpy(np.load(WEIGHT_TENSOR_PATH)).float().to(device)
    w_bcast = w.permute(2, 0, 1).unsqueeze(0)

    xg = payload["X_global"][sample_idx:sample_idx + 1].to(device)
    xc = payload["X_context"][sample_idx:sample_idx + 1].to(device)
    xl_true = payload["X_local"][sample_idx:sample_idx + 1].to(device)
    S_tgt = torch.complex(payload["Y_real"][sample_idx:sample_idx + 1].to(torch.float64),
                          payload["Y_imag"][sample_idx:sample_idx + 1].to(torch.float64)).to(device)

    # ---- tolerance model: sigma in NORMALIZED units + printed physical map --
    d_local = xl_true.shape[1]
    sigma = torch.full((d_local,), tol_frac, device=device)
    local_names = list(payload.get("local_features", [f"x{i}" for i in range(d_local)]))
    log_feats = set(payload.get("log_features", []))
    Xstd = payload["X_local_std"]
    Xstd = Xstd.cpu().numpy() if hasattr(Xstd, "cpu") else np.asarray(Xstd)
    print("\n  TOLERANCE MODEL (1-sigma perturbations, normalized units = "
          f"{tol_frac:.2f} x population std):")
    for i, nm in enumerate(local_names):
        phys = tol_frac * float(Xstd[i])
        unit = "decades (log10 feature -> relative)" if nm in log_feats else "physical units"
        print(f"    {nm:>16s} : sigma_norm={tol_frac:.2f} -> sigma_phys={phys:.4g} {unit}")
    print("  (edit --tol-frac globally now; per-feature IPC-based sigmas are a"
          " one-dict change for the thesis)")

    # ---- frozen models -------------------------------------------------------
    fwd = DirectSequenceResNet(d_local=d_local, d_global=xg.shape[1],
                               d_context=xc.shape[1], F_len=len(freqs_hz)).to(device)
    fwd.load_state_dict(torch.load(FORWARD_MODEL_PATH, map_location=device, weights_only=True))
    fwd.eval()
    for p in fwd.parameters():
        p.requires_grad = False
    cvae = Tandem_cVAE(d_local=d_local, d_global=xg.shape[1], d_context=xc.shape[1]).to(device)
    cvae.load_state_dict(torch.load(_find_cvae_checkpoint(), map_location=device, weights_only=True))
    cvae.eval()
    for p in cvae.parameters():
        p.requires_grad = False

    xl_guess, _, _ = cvae(torch.zeros_like(xl_true), xg, xc, S_tgt)
    cond = cvae.build_condition(S_tgt, xg, xc).detach()

    tgt21_db = 20 * torch.log10(S_tgt[0, :, 1, 0].abs() + 1e-12)
    tgt11_db = 20 * torch.log10(S_tgt[0, :, 0, 0].abs() + 1e-12)

    tag = f"yield_L{lambda_j:g}" + ("_curr" if curriculum else "")
    print(f"\n=== YIELD TTO [{tag}] sample {sample_idx}: {restarts} restarts x "
          f"{tto_steps} steps, lambda_j={lambda_j} ===")

    # =========================================================================
    # RESTART LOOP -- every converged restart is a PORTFOLIO CANDIDATE
    # =========================================================================
    candidates = []
    for r in range(restarts):
        z = torch.randn(1, cvae.latent_dim, device=device, requires_grad=True)
        opt = torch.optim.Adam([z], lr=lr)
        for step in range(tto_steps):
            opt.zero_grad()
            x_dec = cvae.decode(z, cond)
            wm = w_bcast * curriculum_mask(freqs_hz, step, tto_steps, device) \
                 if curriculum else w_bcast
            loss = (weighted_s_loss(fwd(x_dec, xg, xc, freqs_hz), S_tgt, wm)
                    + lambda_prior * z.pow(2).sum()
                    + lambda_bnd * boundary_loss(x_dec))
            if lambda_j > 0:
                loss = loss + lambda_j * sensitivity_penalty(
                    fwd, x_dec, xg, xc, freqs_hz, sigma, w_bcast)
            loss.backward()
            opt.step()

        # ---- evaluate this candidate (no grad) ------------------------------
        with torch.no_grad():
            x_c = cvae.decode(z, cond)
            S_c = fwd(x_c, xg, xc, freqs_hz)
            fit_full = weighted_s_loss(S_c, S_tgt, w_bcast).item()
            c21 = 20 * torch.log10(S_c[0, :, 1, 0].abs() + 1e-12)
            c11 = 20 * torch.log10(S_c[0, :, 0, 0].abs() + 1e-12)
            fit_eye_db = 0.5 * ((c21 - tgt21_db).abs()[eye_mask].mean()
                                + (c11 - tgt11_db).abs()[eye_mask].mean()).item()
            sens = sensitivity_penalty(fwd, x_c, xg, xc, freqs_hz, sigma, w_bcast).item()
            y, ylo, yhi = mc_yield(fwd, x_c, xg, xc, freqs_hz, sigma, S_tgt,
                                   eye_mask, spec_il_margin, spec_rl_max,
                                   n_mc=n_mc, device=device)
        candidates.append({"restart": r, "x": x_c.cpu(), "S": S_c.cpu(),
                           "fit_full": fit_full, "fit_eye_db": fit_eye_db,
                           "sens": sens, "yield": y, "yield_lo": ylo,
                           "yield_hi": yhi,
                           "maxabs": x_c.abs().max().item()})
        print(f"  restart {r:02d} | eye-fit {fit_eye_db:5.2f} dB | "
              f"sens {sens:8.5f} | yield {100*y:5.1f}% "
              f"[{100*ylo:.1f},{100*yhi:.1f}] | max|x| {x_c.abs().max():.2f}")

    # =========================================================================
    # GATE (fit) -> RANK (yield)
    # =========================================================================
    passing = [c for c in candidates if c["fit_eye_db"] <= fit_gate]
    print(f"\n  fit gate ({fit_gate:.1f} dB eye-band): "
          f"{len(passing)}/{len(candidates)} candidates pass")
    if not passing:
        print("  [warn] nothing passed the gate -- ranking ALL candidates by "
              "yield anyway; consider raising --fit-gate or lowering lambda_j")
        passing = candidates
    ranked = sorted(passing, key=lambda c: c["yield"], reverse=True)

    # portfolio diversity (is one-to-many real here?)
    if len(ranked) > 1:
        X = torch.cat([c["x"] for c in ranked], dim=0)
        dist = torch.cdist(X, X)
        off = dist[~torch.eye(len(ranked), dtype=bool)]
        print(f"  portfolio diversity: pairwise |dx| mean {off.mean():.3f}, "
              f"max {off.max():.3f} (normalized units; ~0 would mean all "
              f"restarts found the same design)")

    # =========================================================================
    # EXPORT: ranked CSV + Pareto scatter + per-candidate npz (stage-07 keys)
    # =========================================================================
    design_dir = OUT_DIR / "generated_designs"
    design_dir.mkdir(parents=True, exist_ok=True)
    sim_ids_raw = payload["sim_ids"]
    if hasattr(sim_ids_raw, "cpu"):
        sim_ids_raw = sim_ids_raw.cpu()
    sim_ids = list(sim_ids_raw)
    pids = payload["pair_ids"]
    pids = pids.cpu().numpy() if hasattr(pids, "cpu") else np.asarray(pids)
    template_sim = str(sim_ids[sample_idx])
    pair_id = int(pids[sample_idx])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for rank, c in enumerate(ranked, start=1):
        suffix = f"{tag}_r{c['restart']:02d}"
        npz_path = design_dir / f"design_sample_{sample_idx}_{suffix}.npz"
        S_c = torch.complex(torch.as_tensor(np.real(c["S"].numpy())),
                            torch.as_tensor(np.imag(c["S"].numpy())))
        np.savez(npz_path,
                 x_local_norm=c["x"].numpy(),
                 x_local_cvae=xl_guess.detach().cpu().numpy(),
                 x_local_true=xl_true.detach().cpu().numpy(),
                 x_global_norm=xg.detach().cpu().numpy(),
                 x_context_norm=xc.detach().cpu().numpy(),
                 target_real=S_tgt.real.detach().cpu().numpy(),
                 target_imag=S_tgt.imag.detach().cpu().numpy(),
                 pred_real=np.real(c["S"].numpy()),
                 pred_imag=np.imag(c["S"].numpy()),
                 template_sim_id=np.array([template_sim]),
                 pair_id=np.array([pair_id]))
        rows.append({"rank": rank, "restart": c["restart"],
                     "fit_eye_dB": round(c["fit_eye_db"], 3),
                     "yield_pct": round(100 * c["yield"], 2),
                     "yield_lo": round(100 * c["yield_lo"], 2),
                     "yield_hi": round(100 * c["yield_hi"], 2),
                     "sensitivity": round(c["sens"], 6),
                     "fit_full_loss": round(c["fit_full"], 6),
                     "max_abs_xnorm": round(c["maxabs"], 3),
                     "npz": npz_path.name})
        if rank == 1:  # duplicate the winner under a stable name for stage 07
            top_path = design_dir / f"design_sample_{sample_idx}_{tag}_top.npz"
            import shutil
            shutil.copy(npz_path, top_path)

    csv_path = OUT_DIR / f"portfolio_sample_{sample_idx}_{tag}_{stamp}.csv"
    with open(csv_path, "w", newline="") as fh:
        wri = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wri.writeheader(); wri.writerows(rows)

    # Pareto scatter: fit (x) vs yield (y); non-dominated front highlighted
    fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
    fx = [r["fit_eye_dB"] for r in rows]
    fy = [r["yield_pct"] for r in rows]
    ax.scatter(fx, fy, c="tab:blue", label="candidates")
    front = []
    for r0 in rows:
        if not any((r1["fit_eye_dB"] <= r0["fit_eye_dB"] and
                    r1["yield_pct"] > r0["yield_pct"]) or
                   (r1["fit_eye_dB"] < r0["fit_eye_dB"] and
                    r1["yield_pct"] >= r0["yield_pct"]) for r1 in rows):
            front.append(r0)
    front = sorted(front, key=lambda r: r["fit_eye_dB"])
    ax.plot([r["fit_eye_dB"] for r in front], [r["yield_pct"] for r in front],
            "r--o", label="Pareto front")
    for r in rows[:3]:
        ax.annotate(f"#{r['rank']}", (r["fit_eye_dB"], r["yield_pct"]))
    ax.set_xlabel("nominal eye-band fit error (dB)")
    ax.set_ylabel("MC yield (%)")
    ax.set_title(f"Design portfolio -- sample {sample_idx}, {tag}\n"
                 f"(fit gate {fit_gate} dB, tol {tol_frac:.2f} sigma_pop, "
                 f"N_MC={n_mc})")
    ax.grid(alpha=0.3); ax.legend()
    pareto_path = OUT_DIR / f"pareto_sample_{sample_idx}_{tag}_{stamp}.png"
    fig.savefig(pareto_path, dpi=150); plt.close(fig)

    print("\n" + "=" * 70)
    print(f"  RANKED PORTFOLIO (top 5)  sample {sample_idx}  [{tag}]")
    print("=" * 70)
    print(f"  {'rank':>4s} {'eye-fit dB':>10s} {'yield %':>8s} "
          f"{'sens':>10s} {'file'}")
    for r in rows[:5]:
        print(f"  {r['rank']:>4d} {r['fit_eye_dB']:>10.2f} "
              f"{r['yield_pct']:>8.1f} {r['sensitivity']:>10.5f} {r['npz']}")
    print(f"\n  csv    : {csv_path.name}")
    print(f"  pareto : {pareto_path.name}")
    print(f"  top design (for stage 07): design_sample_{sample_idx}_{tag}_top.npz")
    return rows


# =============================================================================
def main():
    class _Tee:
        def __init__(self, *s): self.s = s
        def write(self, d):
            for x in self.s: x.write(d); x.flush()
        def flush(self):
            for x in self.s: x.flush()

    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--restarts", type=int, default=12)
    ap.add_argument("--lambda-prior", type=float, default=1e-3)
    ap.add_argument("--lambda-bnd", type=float, default=1e-2)
    ap.add_argument("--lambda-j", type=float, default=0.1,
                    help="sensitivity (yield-proxy) weight; 0 = plain latent TTO")
    ap.add_argument("--curriculum", action="store_true")
    ap.add_argument("--tol-frac", type=float, default=0.10,
                    help="1-sigma tolerance as a fraction of population std")
    ap.add_argument("--mc-samples", type=int, default=2000)
    ap.add_argument("--spec-il-margin", type=float, default=1.0,
                    help="eye-band IL may drop at most this many dB below the"
                         " nominal target under perturbation")
    ap.add_argument("--spec-rl-max", type=float, default=-8.0,
                    help="eye-band RL ceiling in dB (absolute)")
    ap.add_argument("--fit-gate", type=float, default=3.0,
                    help="candidates above this eye-band |dB| error vs target"
                         " are disqualified before ranking")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log = open(_THIS_DIR / f"tto_yield_log_{stamp}.log", "w")
    sys.stdout = _Tee(sys.__stdout__, _log)
    sys.stderr = _Tee(sys.__stderr__, _log)

    for idx in args.samples:
        print("\n" + "=" * 68)
        print(f"  SAMPLE {idx}  lambda_j={args.lambda_j}  "
              f"curriculum={args.curriculum}")
        print("=" * 68)
        run_yield_tto(idx, tto_steps=args.steps, lr=args.lr,
                      restarts=args.restarts, lambda_prior=args.lambda_prior,
                      lambda_bnd=args.lambda_bnd, lambda_j=args.lambda_j,
                      curriculum=args.curriculum, tol_frac=args.tol_frac,
                      n_mc=args.mc_samples,
                      spec_il_margin=args.spec_il_margin,
                      spec_rl_max=args.spec_rl_max, fit_gate=args.fit_gate)


if __name__ == "__main__":
    main()