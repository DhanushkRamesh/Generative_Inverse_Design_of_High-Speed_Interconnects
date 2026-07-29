"""
tto_latent_inverse.py
--------------------------------------------------
Unified Test-Time Optimization for generative inverse design.
THREE methods in ONE script, selected by --mode:

  geometry   : baseline -- Adam directly on the raw geometry vector x
               (this is the ORIGINAL TTO; kept so every comparison is
               apples-to-apples with identical loss/steps/lr).
               KNOWN FAILURE (measured on this project): it exploits the
               forward surrogate by driving tan_delta to unphysical values
               (0.10 / 0.33 / 0.63 vs physical ~0.01-0.03 on samples
               0/100/250). This is the out-of-distribution failure named by
               Ren, Padilla & Malof (NeurIPS 2020, arXiv:2009.12919).

  latent     : optimize the cVAE latent z instead of x. The DECODER maps z
               to the design; because the decoder was trained only on real
               dataset geometries, every candidate stays on (near) the
               physical manifold -- tan_delta cannot run away to 0.6.
               Canonical reference: Gomez-Bombarelli et al. 2018
               (arXiv:1610.02415). A prior penalty lambda_prior*||z||^2
               keeps z in the high-density region of N(0,I) where the
               decoder is reliable (Notin, Hernandez-Lobato & Gal,
               NeurIPS 2021, arXiv:2107.00096). An optional NA-style
               boundary loss on the DECODED design adds belt-and-braces
               (Ren et al. 2020). TTO framing follows LaBash et al. 2025
               (arXiv:2505.18188).

  --curriculum (flag, composes with either mode):
               frequency-band unmasking schedule on the element-aware
               weight tensor (Bengio et al., ICML 2009):
                 first third of steps  : only 0-28 GHz weighted (eye band)
                 second third          : 0-56 GHz (adds harmonic band)
                 final third           : full 0-100 GHz
               Same optimizer, same loss -- ONLY the weights change over
               steps. Rationale: the eye band is smooth (wide basins);
               securing it first prevents early trapping in the narrow
               high-frequency resonance valleys.

OUTPUT (per sample, per method)
  design npz  : evaluation_results/generated_designs/
                design_sample_{idx}_{method}.npz
                SAME KEYS as before -> stage 07 (validate_model_generated.py)
                consumes it unchanged (incl. --check-only sanity gate).
  plot        : evaluation_results/tto_{method}_sample_{idx}.png
  log         : tto_{method}_log_YYYYMMDD_HHMMSS.log  (timestamped, never
                overwrites)

USAGE
  python3 tto_latent_inverse.py --mode latent --samples 0 42 100 250
  python3 tto_latent_inverse.py --mode latent --curriculum --samples 0
  python3 tto_latent_inverse.py --mode geometry --samples 0        # baseline
  # then, per design:
  #   validate_model_generated.py --designs .../design_sample_0_latent.npz --check-only
"""

import argparse
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
# PATH CONFIGURATION (identical to the original TTO script)
# =============================================================================
PROJECT_ROOT = Path.home() / "mece_project_inverse_model" / "Generative_Inverse_Design_of_High-Speed_Interconnects"
DATA_PT = PROJECT_ROOT / "data" / "processed" / "Universal-Diff-SI-Array" / "diff_pair_dataset.pt"
WEIGHT_TENSOR_PATH = PROJECT_ROOT / "sandbox_v1" / "data" / "frequency_eda" / "weights_element_aware_per_freq.npy"
FORWARD_MODEL_PATH = PROJECT_ROOT / "sandbox_v1" / "models" / "forward_runs" / "run_2026-06-01_213554_direct_resnet_array" / "checkpoint_best.pt"
INVERSE_RUNS_DIR = PROJECT_ROOT / "sandbox_v1" / "models" / "inverse_runs"


def _find_cvae_checkpoint():
    """Resolve the latest element-aware cVAE run lazily (at run time, not at
    import time) so the module can be imported/tested without checkpoints."""
    runs = sorted(INVERSE_RUNS_DIR.glob("run_*_inverse_element_aware_*"))
    if not runs:
        raise FileNotFoundError(
            f"no run_*_inverse_element_aware_* under {INVERSE_RUNS_DIR}")
    return runs[-1] / "cvae_element_aware_best.pt"
OUT_DIR = PROJECT_ROOT / "sandbox_v1" / "models" / "evaluation_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
_THIS_DIR = Path(__file__).resolve().parent

# =============================================================================
# ARCHITECTURES -- byte-identical module/attribute names to the trained
# checkpoints (state_dict compatibility). Only NEW METHODS are added to
# Tandem_cVAE (methods do not appear in a state_dict, so loading is safe).
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

    # ---- training-time forward (unchanged; keeps checkpoint compatibility) --
    def forward(self, x_local, x_global, x_context, S_target):
        cond = torch.cat([self.s_encoder(S_target), x_global, x_context], dim=1)
        h = self.enc_mlp(torch.cat([x_local, cond], dim=1))
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return self.dec_mlp(torch.cat([z, cond], dim=1)), mu, logvar

    # ---- NEW inference helpers (methods only -- no new parameters) ----------
    def build_condition(self, S_target, x_global, x_context):
        """Condition vector c = [Enc_S(S*), x_global, x_context]."""
        return torch.cat([self.s_encoder(S_target), x_global, x_context], dim=1)

    def decode(self, z, cond):
        """x = Dec(z, c): the manifold projection used by latent TTO."""
        return self.dec_mlp(torch.cat([z, cond], dim=1))


# =============================================================================
# CURRICULUM: frequency-band weight schedule (Bengio et al. 2009)
# =============================================================================
def curriculum_mask(freqs_hz, step, total_steps, device):
    """(1, F, 1, 1) multiplicative mask on the element-aware weights.

    thirds of the run: 0-28 GHz -> 0-56 GHz -> full band. Hard switches;
    a sigmoid ramp is a possible refinement but hard switches are simpler
    to report and worked in preliminary tests of curriculum methods.
    """
    frac = step / max(total_steps - 1, 1)
    if frac < 1.0 / 3.0:
        f_hi = 28e9
    elif frac < 2.0 / 3.0:
        f_hi = 56e9
    else:
        f_hi = float("inf")
    m = (freqs_hz <= f_hi).to(torch.float32)
    return m.view(1, -1, 1, 1).to(device)


# =============================================================================
# LOSSES
# =============================================================================
def weighted_s_loss(S_pred, S_tgt, w_bcast):
    """The element-aware physics loss -- identical to training and to the
    original geometry TTO, so all methods optimize the SAME objective."""
    dr = (S_pred.real.float() - S_tgt.real.float()).pow(2)
    di = (S_pred.imag.float() - S_tgt.imag.float()).pow(2)
    return (dr * w_bcast).mean() + (di * w_bcast).mean()


def boundary_loss(x_norm, margin=2.5):
    """NA-style out-of-distribution penalty (Ren et al. 2020) on the
    NORMALIZED design vector. Features are z-scored, so |x| > ~2.5 sigma is
    leaving the training distribution. Zero inside the envelope."""
    return F.relu(x_norm.abs() - margin).pow(2).sum()


# =============================================================================
# CORE: one TTO run (all modes)
# =============================================================================
def run_tto(sample_idx, mode="latent", curriculum=False, tto_steps=150,
            lr=0.05, restarts=8, lambda_prior=1e-3, lambda_bnd=1e-2,
            seed=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    print(f"Loading data from {DATA_PT.name} ...")
    payload = torch.load(DATA_PT, map_location=device, weights_only=False)
    freqs_hz = payload["frequencies"].to(device)
    freqs_ghz = freqs_hz.cpu().numpy() / 1e9

    print("Loading element-aware weight tensor ...")
    w = torch.from_numpy(np.load(WEIGHT_TENSOR_PATH)).float().to(device)
    w_bcast = w.permute(2, 0, 1).unsqueeze(0)          # (1, F, 4, 4)

    # ---- the target sample --------------------------------------------------
    xg = payload["X_global"][sample_idx:sample_idx + 1].to(device)
    xc = payload["X_context"][sample_idx:sample_idx + 1].to(device)
    xl_true = payload["X_local"][sample_idx:sample_idx + 1].to(device)
    S_tgt = torch.complex(payload["Y_real"][sample_idx:sample_idx + 1].to(torch.float64),
                          payload["Y_imag"][sample_idx:sample_idx + 1].to(torch.float64)).to(device)

    # ---- frozen models -------------------------------------------------------
    print("Loading frozen forward surrogate + cVAE ...")
    fwd = DirectSequenceResNet(d_local=xl_true.shape[1], d_global=xg.shape[1],
                               d_context=xc.shape[1], F_len=len(freqs_hz)).to(device)
    fwd.load_state_dict(torch.load(FORWARD_MODEL_PATH, map_location=device, weights_only=True))
    fwd.eval()
    for p in fwd.parameters():
        p.requires_grad = False

    cvae = Tandem_cVAE(d_local=xl_true.shape[1], d_global=xg.shape[1], d_context=xc.shape[1]).to(device)
    cvae.load_state_dict(torch.load(_find_cvae_checkpoint(), map_location=device, weights_only=True))
    cvae.eval()
    for p in cvae.parameters():
        p.requires_grad = False

    # ---- cVAE initial guess (identical to the original script) --------------
    xl_guess, _, _ = cvae(torch.zeros_like(xl_true), xg, xc, S_tgt)
    cond = cvae.build_condition(S_tgt, xg, xc).detach()

    method_tag = mode + ("_curr" if curriculum else "")
    print(f"\n=== TTO [{method_tag}] sample {sample_idx}: "
          f"{tto_steps} steps, lr={lr}, restarts={restarts if mode=='latent' else 1} ===")

    # =========================================================================
    # OPTIMIZATION
    # =========================================================================
    if mode == "geometry":
        # -------- baseline: descend on raw x (the exploitable method) --------
        xl_opt = xl_guess.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([xl_opt], lr=lr)
        for step in range(tto_steps):
            opt.zero_grad()
            wm = w_bcast * curriculum_mask(freqs_hz, step, tto_steps, device) \
                 if curriculum else w_bcast
            loss = weighted_s_loss(fwd(xl_opt, xg, xc, freqs_hz), S_tgt, wm)
            loss.backward()
            opt.step()
            if step % 25 == 0 or step == tto_steps - 1:
                print(f"  step {step:03d} | loss {loss.item():.5f}")
        xl_best = xl_opt.detach()

    elif mode == "latent":
        # -------- latent TTO with multi-restart -----------------------------
        # Each restart: a different z0 ~ N(0, I). We keep the restart whose
        # FINAL FULL-BAND loss is lowest (curriculum only reshapes the path,
        # the selection criterion is always the true objective).
        best = {"loss": float("inf"), "x": None, "z": None, "r": -1}
        for r in range(restarts):
            z = torch.randn(1, cvae.latent_dim, device=device, requires_grad=True)
            opt = torch.optim.Adam([z], lr=lr)
            for step in range(tto_steps):
                opt.zero_grad()
                x_dec = cvae.decode(z, cond)
                wm = w_bcast * curriculum_mask(freqs_hz, step, tto_steps, device) \
                     if curriculum else w_bcast
                loss = (weighted_s_loss(fwd(x_dec, xg, xc, freqs_hz), S_tgt, wm)
                        + lambda_prior * z.pow(2).sum()        # Notin 2021
                        + lambda_bnd * boundary_loss(x_dec))   # Ren 2020
                loss.backward()
                opt.step()
            with torch.no_grad():
                x_dec = cvae.decode(z, cond)
                full = weighted_s_loss(fwd(x_dec, xg, xc, freqs_hz), S_tgt, w_bcast)
            print(f"  restart {r:02d} | final full-band loss {full.item():.5f} "
                  f"| max|x_norm| {x_dec.abs().max().item():.2f}")
            if full.item() < best["loss"]:
                best = {"loss": full.item(), "x": x_dec.detach().clone(),
                        "z": z.detach().clone(), "r": r}
        xl_best = best["x"]
        print(f"  -> best restart {best['r']} (loss {best['loss']:.5f})")
    else:
        raise ValueError(f"unknown mode {mode}")

    # in-script physicality hint (full check is stage 07 --check-only):
    print(f"  decoded max |x_norm| = {xl_best.abs().max().item():.2f} "
          f"(training data is z-scored: >3 means out-of-distribution)")

    S_guess = fwd(xl_guess, xg, xc, freqs_hz)
    S_tto = fwd(xl_best, xg, xc, freqs_hz)

    # =========================================================================
    # EXPORT (same keys as before -> stage 07 consumes it unchanged)
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

    design_path = design_dir / f"design_sample_{sample_idx}_{method_tag}.npz"
    np.savez(
        design_path,
        x_local_norm=xl_best.cpu().numpy(),
        x_local_cvae=xl_guess.detach().cpu().numpy(),
        x_local_true=xl_true.detach().cpu().numpy(),
        x_global_norm=xg.detach().cpu().numpy(),
        x_context_norm=xc.detach().cpu().numpy(),
        target_real=S_tgt.real.detach().cpu().numpy(),
        target_imag=S_tgt.imag.detach().cpu().numpy(),
        pred_real=S_tto.real.detach().cpu().numpy(),
        pred_imag=S_tto.imag.detach().cpu().numpy(),
        template_sim_id=np.array([template_sim]),
        pair_id=np.array([pair_id]),
    )
    print(f"[stage-07 export] {design_path}")
    print(f"    template {template_sim}  pair {pair_id}")

    # =========================================================================
    # PLOT
    # =========================================================================
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10,
                         "axes.grid": True, "grid.alpha": 0.3})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    def to_db(S, r, c):
        return 20 * torch.log10(S[0, :, r, c].abs() + 1e-12).detach().cpu().numpy()

    for ax, (i, j, ttl) in zip((ax1, ax2),
                               [(0, 0, "Return Loss (|Sdd11|)"),
                                (1, 0, "Insertion Loss (|Sdd21|)")]):
        ax.plot(freqs_ghz, to_db(S_tgt, i, j), 'k-', lw=2.5, label='Target')
        ax.plot(freqs_ghz, to_db(S_guess, i, j), 'tab:red', ls='--', lw=2.0,
                label='cVAE guess')
        ax.plot(freqs_ghz, to_db(S_tto, i, j), 'tab:green', lw=2.0,
                label=f'TTO [{method_tag}]')
        ax.axvspan(0, 28, alpha=0.10, color='orange')
        ax.set_xlabel("Frequency (GHz)"); ax.set_ylabel("dB")
        ax.set_title(ttl); ax.set_ylim(-60, 5); ax.legend()

    plt.suptitle(f"Inverse design TTO -- mode={method_tag}, sample {sample_idx}")
    plt.tight_layout()
    fig_path = OUT_DIR / f"tto_{method_tag}_sample_{sample_idx}.png"
    plt.savefig(fig_path); plt.close(fig)
    print(f"plot: {fig_path.name}")
    return design_path


# =============================================================================
def main():
    class _Tee:
        def __init__(self, *s): self.s = s
        def write(self, d):
            for x in self.s: x.write(d); x.flush()
        def flush(self):
            for x in self.s: x.flush()

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["geometry", "latent"], default="latent")
    ap.add_argument("--curriculum", action="store_true",
                    help="frequency-band unmasking schedule (Bengio 2009)")
    ap.add_argument("--samples", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--restarts", type=int, default=8,
                    help="latent mode: random z0 restarts, best kept")
    ap.add_argument("--lambda-prior", type=float, default=1e-3,
                    help="||z||^2 weight (keeps z near N(0,I); Notin 2021)")
    ap.add_argument("--lambda-bnd", type=float, default=1e-2,
                    help="NA boundary-loss weight on decoded x (Ren 2020)")
    args = ap.parse_args()

    tag = args.mode + ("_curr" if args.curriculum else "")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log = open(_THIS_DIR / f"tto_{tag}_log_{stamp}.log", "w")
    sys.stdout = _Tee(sys.__stdout__, _log)
    sys.stderr = _Tee(sys.__stderr__, _log)

    for idx in args.samples:
        print("\n" + "=" * 68)
        print(f"  SAMPLE {idx}   mode={args.mode}  curriculum={args.curriculum}")
        print("=" * 68)
        run_tto(idx, mode=args.mode, curriculum=args.curriculum,
                tto_steps=args.steps, lr=args.lr, restarts=args.restarts,
                lambda_prior=args.lambda_prior, lambda_bnd=args.lambda_bnd)


if __name__ == "__main__":
    main()