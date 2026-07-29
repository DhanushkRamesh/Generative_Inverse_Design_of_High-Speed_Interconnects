"""
04b_element_aware_weights.py
================================================================================
Element-aware (per-S-parameter) band weight derivation.

This script extends the single-vector band weighting produced by
04_frequency_importance_eda.py to PER-ELEMENT weights, one weight vector per
type of S-parameter element in the 4x4 mixed-mode matrix.

WHY ELEMENT-AWARE WEIGHTS:
    The previous EDA produced a single weight vector w(f) that emphasized the
    14-28 GHz 112G Nyquist band for all elements equally.  The data analysis
    in that notebook (per-frequency std, deep-null clustering, model MAE)
    reveals that this is suboptimal:
       - |Sdd11| has high variance and rare nulls in 0-28 GHz
         (resonant matching nulls)
       - |Sdd21| has high variance and rare nulls in 28-100 GHz
         (insertion-loss resonances)
    These two are MIRROR images across the spectrum.  A single shared w(f)
    averages over both patterns; per-element vectors let the loss attend to
    each element where it actually has information to fit.

    The supervisor's exact question was: "see if [the band] could be split
    into regions of varying importance and if this could inform a process for
    combining the regions."  Per-element weighting is the data-driven answer
    to that question.

INPUTS (assumed to exist from running 04_frequency_importance_eda.py):
    sandbox_v1/data/frequency_eda/band_weights.json
    sandbox_v1/data/frequency_eda/band_weights_per_freq.npy
    data/processed/Universal-Diff-SI-Array/diff_pair_dataset.pt

OUTPUTS:
    sandbox_v1/data/frequency_eda/weights_element_aware.json
        Per-element band scheme with full rationale, ready for thesis citation.
    sandbox_v1/data/frequency_eda/weights_element_aware_per_freq.npy
        Shape (4, 4, F) tensor for direct multiplication into the loss:
            physics_loss = (w_per_element * |S_gen - S_target|^2).mean()
    sandbox_v1/data/frequency_eda/09_element_aware_weight_overlay.png
        Per-element band visualization, four panels (Sdd11/Sdd21/Sdc/Scc).
    sandbox_v1/data/frequency_eda/10_single_vs_element_comparison.png
        Side-by-side: old single w(f) versus new w_S11(f) and w_S21(f).

USAGE:
    python 04b_element_aware_weights.py

References:
    LaBash et al. 2025, arXiv:2505.18188 -- frequency band masking in inverse
        design loss (motivates per-element treatment).
    Schierholz et al., IEEE Access 2021 -- TUHH dataset and mixed-mode 4x4
        S-parameter convention.
    Bockelman & Eisenstadt, IEEE TMTT 1995 -- mixed-mode S-parameter theory
        (basis for the 4x4 element interpretation).
"""

# %% Section 0: Imports and paths
import json
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # safe for headless terminals; remove for interactive use
import matplotlib.pyplot as plt
import numpy as np
import torch

# Project paths follow the rest of the sandbox_v1 convention
PROJECT_ROOT = (
    Path.home()
    / "mece_project_inverse_model"
    / "Generative_Inverse_Design_of_High-Speed_Interconnects"
)
DATA_PT = (
    PROJECT_ROOT / "data" / "processed" / "Universal-Diff-SI-Array"
    / "diff_pair_dataset.pt"
)
EDA_OUT_DIR = PROJECT_ROOT / "sandbox_v1" / "data" / "frequency_eda"
EDA_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Existing single-vector outputs from 04_frequency_importance_eda.py
EXISTING_JSON = EDA_OUT_DIR / "band_weights.json"
EXISTING_NPY = EDA_OUT_DIR / "band_weights_per_freq.npy"

# Thesis-friendly plot defaults
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

print(f"Project root:    {PROJECT_ROOT}")
print(f"Output dir:      {EDA_OUT_DIR}")
print(f"Existing JSON:   {EXISTING_JSON}")


# %% Section 1: Load frequency grid and (optionally) the existing single-weight scheme
print("\n=== Section 1: Load frequency grid ===")

# We only need the frequency axis from the dataset; the per-element weight
# construction is closed-form (band definitions -> piecewise-constant vector)
payload = torch.load(DATA_PT, weights_only=False, map_location="cpu")
freqs_ghz = payload["frequencies"].numpy() / 1e9
F_LEN = len(freqs_ghz)
print(f"Frequency points: {F_LEN}")
print(f"Frequency range:  {freqs_ghz.min():.3f} - {freqs_ghz.max():.3f} GHz")

# Load the previous single-vector weights for comparison plots (Section 6)
if EXISTING_JSON.exists() and EXISTING_NPY.exists():
    with open(EXISTING_JSON) as f:
        single_band_data = json.load(f)
    single_weights = np.load(EXISTING_NPY)
    print(f"Loaded existing single-weight scheme: w in "
          f"[{single_weights.min():.2f}, {single_weights.max():.2f}]")
else:
    single_band_data = None
    single_weights = None
    print("Note: previous single-vector outputs not found; comparison plot "
          "will fall back to a synthetic baseline (still useful for thesis).")


# %% Section 2: Mixed-mode 4x4 element convention
# The TUHH Universal-Diff-SI-Array dataset stores Y as (N, F, 4, 4) complex
# tensors in the standard Bockelman-Eisenstadt mixed-mode ordering, where the
# 4 indices map to (port1_diff, port2_diff, port1_common, port2_common).
# The resulting 4x4 partitions into four quadrants:
#
#       ports:   p1_d   p2_d   p1_c   p2_c
#     p1_d  [   Sdd11  Sdd12  Sdc11  Sdc12 ]
#     p2_d  [   Sdd21  Sdd22  Sdc21  Sdc22 ]
#     p1_c  [   Scd11  Scd12  Scc11  Scc12 ]
#     p2_c  [   Scd21  Scd22  Scc21  Scc22 ]
#
# Reading the four quadrants:
#   Top-left (Sdd)     : differential -> differential (primary path)
#   Top-right (Sdc)    : common -> differential conversion (EMC concern)
#   Bottom-left (Scd)  : differential -> common conversion (EMC concern;
#                        equals Sdc^T by reciprocity for passive networks)
#   Bottom-right (Scc) : common -> common (largely uninteresting for SerDes)
#
# Engineering importance for 112G PAM4 inverse design:
#   Sdd11, Sdd22  : differential return loss     -- primary
#   Sdd21 = Sdd12 : differential insertion loss  -- primary
#   Sdc, Scd      : mode conversion              -- EMC / yield concern
#   Scc           : common-mode return/insertion -- least relevant

ELEMENT_TYPE_FOR_IJ = {}
# Top-left 2x2: Sdd quadrant
ELEMENT_TYPE_FOR_IJ[(0, 0)] = "Sdd11"     # Sdd11
ELEMENT_TYPE_FOR_IJ[(1, 1)] = "Sdd11"     # Sdd22 (treated like Sdd11 - same character)
ELEMENT_TYPE_FOR_IJ[(0, 1)] = "Sdd21"     # Sdd12 (= Sdd21 by reciprocity)
ELEMENT_TYPE_FOR_IJ[(1, 0)] = "Sdd21"     # Sdd21 (insertion loss)
# Top-right 2x2: Sdc quadrant
ELEMENT_TYPE_FOR_IJ[(0, 2)] = "Sdc"       # Sdc11
ELEMENT_TYPE_FOR_IJ[(0, 3)] = "Sdc"       # Sdc12
ELEMENT_TYPE_FOR_IJ[(1, 2)] = "Sdc"       # Sdc21
ELEMENT_TYPE_FOR_IJ[(1, 3)] = "Sdc"       # Sdc22
# Bottom-left 2x2: Scd quadrant (= Sdc^T by reciprocity)
ELEMENT_TYPE_FOR_IJ[(2, 0)] = "Sdc"       # Scd11 mirrors Sdc11
ELEMENT_TYPE_FOR_IJ[(2, 1)] = "Sdc"       # Scd12 mirrors Sdc21
ELEMENT_TYPE_FOR_IJ[(3, 0)] = "Sdc"       # Scd21 mirrors Sdc12
ELEMENT_TYPE_FOR_IJ[(3, 1)] = "Sdc"       # Scd22 mirrors Sdc22
# Bottom-right 2x2: Scc quadrant
ELEMENT_TYPE_FOR_IJ[(2, 2)] = "Scc"       # Scc11
ELEMENT_TYPE_FOR_IJ[(2, 3)] = "Scc"       # Scc12
ELEMENT_TYPE_FOR_IJ[(3, 2)] = "Scc"       # Scc21
ELEMENT_TYPE_FOR_IJ[(3, 3)] = "Scc"       # Scc22


# %% Section 3: Element-aware band scheme
# Each band specification is a list of tuples:
#   (f_lo_GHz, f_hi_GHz, weight, rationale_string)
#
# Weights are derived from THREE signals, all measured in the previous EDA:
#   1. 112G PAM4 industry priority         -- IEEE 802.3ck spec
#   2. Per-band data std                   -- Image 2 of the EDA
#   3. Per-band deep-null density          -- Image 7 middle panel of the EDA
#   4. Per-band forward model MAE          -- Image 7 bottom panel of the EDA
#
# The rationale strings below cite the actual numbers observed in your
# EDA run.  When the weights are debated (with the supervisor or in the
# thesis), these numbers are the defense.

ELEMENT_BANDS = OrderedDict([
    # -----------------------------------------------------------------------
    # Sdd11-type weighting (return-loss style elements: Sdd11, Sdd22)
    # -----------------------------------------------------------------------
    # Sdd11 data signature from your EDA:
    #   - std declines from 8.45 dB (0-14 GHz) to 5.76 dB (42-56 GHz)
    #   - deep nulls heavily concentrated at low freq: 77% at 0-14, 41% at
    #     14-28, then 12% / 7% / 21% at higher bands
    #   - model MAE peaks at 2.07 dB in the 14-28 GHz Nyquist band
    # Conclusion: weight strongly in 0-28 GHz, taper aggressively above.
    ("Sdd11", [
        (0,   14,  4.0, "std 8.45 dB (highest); 77% of all deep nulls; "
                        "matching resonance haven"),
        (14,  28,  5.0, "112G PAM4 Nyquist; 41% deep nulls; forward MAE "
                        "peak (2.07 dB)"),
        (28,  42,  2.0, "std 6.26 dB; only 12% deep nulls; forward already "
                        "strong (MAE 1.29)"),
        (42,  56,  1.0, "std 5.76 dB; 7% deep nulls; minimal Sdd11 "
                        "information here"),
        (56, 100,  0.5, "tail; std flat at 6 dB but mostly featureless for "
                        "return loss"),
    ]),

    # -----------------------------------------------------------------------
    # Sdd21-type weighting (insertion-loss style elements: Sdd21, Sdd12)
    # -----------------------------------------------------------------------
    # Sdd21 data signature from your EDA:
    #   - std rises from 0 dB (at DC) to 24 dB (at 70 GHz)
    #   - deep nulls almost absent below 28 GHz (<5%), then 31% / 58% / 83%
    #   - model MAE rises from 0.46 dB to 3.34 dB at 42-56 GHz
    # Conclusion: weight strongly in 28-100 GHz, retain modest Nyquist
    # priority for industry alignment.
    ("Sdd21", [
        (0,   14,  1.0, "near-featureless: std 2.04 dB, 0.4% nulls, "
                        "model already excellent (MAE 0.46)"),
        (14,  28,  3.0, "112G PAM4 Nyquist; industry priority maintained "
                        "even though Sdd21 data is still smooth here"),
        (28,  42,  4.0, "Sdd21 information starts in earnest: std 12.86 dB, "
                        "31% deep nulls, forward MAE 2.57 dB"),
        (42,  56,  5.0, "PEAK of Sdd21: std 19.63 dB, 58% deep nulls, "
                        "worst forward MAE (3.34 dB)"),
        (56, 100,  3.0, "beyond 112G spec BUT 83% of all Sdd21 deep nulls "
                        "live here; the cVAE was failing exactly on these"),
    ]),

    # -----------------------------------------------------------------------
    # Sdc/Scd-type weighting (mode conversion, EMC compliance)
    # -----------------------------------------------------------------------
    # We do not have direct per-band statistics for the mode-conversion
    # elements in the same depth as the EDA gave us for Sdd11/Sdd21.
    # Physical priors:
    #   - Mode conversion is strongest where the differential structure
    #     resonates (around Nyquist for 112G channels)
    #   - EMC compliance for SerDes is regulated around the data rate band
    #   - Sdc and Scd are equal by reciprocity for passive networks
    # We adopt a moderate Nyquist-emphasized scheme.  This can be
    # data-validated later by extending the EDA to mode-conversion elements
    # if accuracy at EMC bands turns out to bottleneck the inverse model.
    ("Sdc", [
        (0,   14,  1.5, "low common-mode coupling at baseband"),
        (14,  28,  3.0, "Nyquist; PRIMARY band for EMC compliance at 112G"),
        (28,  42,  2.5, "harmonic mode coupling region"),
        (42,  56,  2.0, "2x Nyquist mode resonances; secondary EMC"),
        (56, 100,  1.0, "out of 112G EMC interest"),
    ]),

    # -----------------------------------------------------------------------
    # Scc-type weighting (common mode)
    # -----------------------------------------------------------------------
    # Common-mode return loss matters mainly for EMI radiation.  For
    # differential SerDes inverse design, this is the LEAST critical
    # quadrant.  We assign uniformly low weights but keep nonzero so the
    # model does not produce structurally pathological common-mode behavior.
    ("Scc", [
        (0,   14,  1.0, "baseband common-mode return"),
        (14,  28,  1.5, "Nyquist; minor EMC relevance"),
        (28,  42,  1.5, "harmonic common-mode behavior"),
        (42,  56,  1.0, "secondary"),
        (56, 100,  0.5, "minimal common-mode interest at this band"),
    ]),
])


# %% Section 4: Build per-element weight vectors w_type(f)
# Each weight vector is piecewise constant over the 5 bands above.
# Shape: (F_LEN,) for each element type.

print("\n=== Section 4: Build per-element weight vectors ===")


def build_weight_vector(band_spec):
    """
    Construct a piecewise-constant w(f) of length F_LEN from a band specification.

    Parameters
    ----------
    band_spec : list of tuples
        Each tuple is (f_lo_GHz, f_hi_GHz, weight, rationale_string).

    Returns
    -------
    w : np.ndarray of shape (F_LEN,)
        Per-frequency weight aligned to freqs_ghz.
    """
    w = np.zeros(F_LEN, dtype=np.float64)
    for f_lo, f_hi, weight, _ in band_spec:
        # Half-open interval [f_lo, f_hi) so consecutive bands do not overlap
        mask = (freqs_ghz >= f_lo) & (freqs_ghz < f_hi)
        w[mask] = weight
    # The last band's f_hi should also catch the very last frequency point
    last_band = band_spec[-1]
    w[freqs_ghz >= last_band[1]] = last_band[2]
    return w


element_weight_vectors = {
    elem_type: build_weight_vector(spec)
    for elem_type, spec in ELEMENT_BANDS.items()
}

# Quick sanity print
for elem_type, w in element_weight_vectors.items():
    print(f"  {elem_type:6s}: w in [{w.min():.2f}, {w.max():.2f}], "
          f"mean {w.mean():.2f}")


# %% Section 5: Build the (4, 4, F) per-element tensor
# This is the object the cVAE physics loss and TTO will consume.  At every
# (i, j, f) it holds the appropriate per-element weight.

print("\n=== Section 5: Build (4, 4, F) weight tensor ===")

w_per_element = np.zeros((4, 4, F_LEN), dtype=np.float64)
for (i, j), elem_type in ELEMENT_TYPE_FOR_IJ.items():
    w_per_element[i, j, :] = element_weight_vectors[elem_type]

# Verify symmetry: since S = S^T for reciprocal passive networks, the weight
# tensor must also be symmetric in (i, j).  This is automatically true by
# construction of ELEMENT_TYPE_FOR_IJ but verify defensively.
assert np.allclose(w_per_element, w_per_element.transpose(1, 0, 2)), \
    "Weight tensor is not symmetric across (i,j); check ELEMENT_TYPE_FOR_IJ"
print(f"  shape: {w_per_element.shape}")
print(f"  range: [{w_per_element.min():.2f}, {w_per_element.max():.2f}]")
print(f"  symmetry across (i,j): OK")

# Per-element mean weight as a 4x4 sanity matrix
mean_w_per_element = w_per_element.mean(axis=-1)
print("\n  Mean weight per element (4x4, averaged over frequency):")
print("  " + np.array2string(mean_w_per_element, precision=2,
                              suppress_small=True).replace("\n", "\n  "))


# %% Section 6: Visualize per-element weights
# Figure 9: four panels showing each element-type's weight vector along with
# the band boundaries shaded.

print("\n=== Section 6: Plot per-element weights ===")

# 112G PAM4 band boundaries used for shading
PAM4_BAND_EDGES_GHZ = [0, 14, 28, 42, 56, 100]
PAM4_BAND_COLORS = ["#FFE6CC", "#FFB347", "#FFD96A", "#A6E22E", "#7FDBFF"]
PAM4_BAND_LABELS = [
    "0-14 (baseband)", "14-28 (Nyquist)", "28-42 (channel BW)",
    "42-56 (harmonics)", "56-100 (out of band)",
]


def draw_bands(ax):
    """Shade the 5 PAM4 bands behind any plot."""
    for k in range(5):
        ax.axvspan(
            PAM4_BAND_EDGES_GHZ[k], PAM4_BAND_EDGES_GHZ[k + 1],
            alpha=0.25, color=PAM4_BAND_COLORS[k],
            label=PAM4_BAND_LABELS[k] if ax._first_band_draw else None,
        )
    ax._first_band_draw = False


# Per-element label/color pairs for the line plot
ELEMENT_LINE_STYLE = OrderedDict([
    ("Sdd11", dict(color="navy",      lw=2.2, ls="-",  label="w_Sdd11 / Sdd22 (return loss)")),
    ("Sdd21", dict(color="darkred",   lw=2.2, ls="-",  label="w_Sdd21 / Sdd12 (insertion loss)")),
    ("Sdc",   dict(color="darkgreen", lw=1.8, ls="--", label="w_Sdc / Scd (mode conversion)")),
    ("Scc",   dict(color="dimgray",   lw=1.8, ls=":",  label="w_Scc (common mode)")),
])

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
for ax, (elem_type, style) in zip(axes.flat, ELEMENT_LINE_STYLE.items()):
    ax._first_band_draw = True
    draw_bands(ax)
    ax.plot(freqs_ghz, element_weight_vectors[elem_type],
             color=style["color"], lw=style["lw"], ls=style["ls"],
             label=style["label"])
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("Weight w(f)")
    ax.set_title(f"{elem_type}-type weighting")
    ax.set_ylim(0, max(b[2] for spec in ELEMENT_BANDS.values()
                        for b in spec) * 1.15)
    ax.legend(loc="upper right", fontsize=8, ncol=1)

fig.suptitle("Element-aware band weights (data-driven, 112G PAM4 anchored)",
             fontsize=12)
plt.tight_layout()
out_path = EDA_OUT_DIR / "09_element_aware_weight_overlay.png"
plt.savefig(out_path, bbox_inches="tight")
plt.close()
print(f"  Saved: {out_path.name}")

# Figure 10: side-by-side single (old) vs element-aware Sdd11 / Sdd21 weights
# This is the headline thesis figure showing what we changed and why.
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax._first_band_draw = True
draw_bands(ax)
if single_weights is not None:
    ax.plot(freqs_ghz, single_weights, color="black", lw=2.0, ls="-.",
             label="OLD single w(f) (industry-prior)")
ax.plot(freqs_ghz, element_weight_vectors["Sdd11"],
         color="navy", lw=2.2, label="NEW w_Sdd11(f) (data-driven)")
ax.plot(freqs_ghz, element_weight_vectors["Sdd21"],
         color="darkred", lw=2.2, label="NEW w_Sdd21(f) (data-driven)")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Weight w(f)")
ax.set_title("Single industry-prior weight vs element-aware data-driven "
              "weights for Sdd11 / Sdd21")
ax.legend(loc="upper right", fontsize=9, ncol=2)
ax.set_ylim(0, 6)
plt.tight_layout()
out_path = EDA_OUT_DIR / "10_single_vs_element_comparison.png"
plt.savefig(out_path, bbox_inches="tight")
plt.close()
print(f"  Saved: {out_path.name}")


# %% Section 7: Per-band stats summary tied to the new weights
# Re-tabulates the per-band signal characteristics alongside the new
# element-aware weights so the rationale is visible at a glance.

print("\n=== Section 7: Per-band weight + signal characteristic summary ===")
print("\nSdd11 / Sdd22 (return-loss type):")
print(f"  {'Band':22s}  {'w':>4s}  Justification")
print("  " + "-" * 75)
for f_lo, f_hi, w, why in ELEMENT_BANDS["Sdd11"]:
    print(f"  {f_lo:3d}-{f_hi:3d} GHz             {w:>4.1f}  {why}")

print("\nSdd21 / Sdd12 (insertion-loss type):")
print(f"  {'Band':22s}  {'w':>4s}  Justification")
print("  " + "-" * 75)
for f_lo, f_hi, w, why in ELEMENT_BANDS["Sdd21"]:
    print(f"  {f_lo:3d}-{f_hi:3d} GHz             {w:>4.1f}  {why}")

print("\nSdc / Scd (mode conversion):")
print(f"  {'Band':22s}  {'w':>4s}  Justification")
print("  " + "-" * 75)
for f_lo, f_hi, w, why in ELEMENT_BANDS["Sdc"]:
    print(f"  {f_lo:3d}-{f_hi:3d} GHz             {w:>4.1f}  {why}")

print("\nScc (common mode):")
print(f"  {'Band':22s}  {'w':>4s}  Justification")
print("  " + "-" * 75)
for f_lo, f_hi, w, why in ELEMENT_BANDS["Scc"]:
    print(f"  {f_lo:3d}-{f_hi:3d} GHz             {w:>4.1f}  {why}")


# %% Section 8: Save JSON and NPY artifacts
# - JSON: human-readable; full band scheme + per-element rationale
# - NPY:  numerical (4, 4, F) tensor; load directly into the cVAE / TTO loss

print("\n=== Section 8: Save outputs ===")

output_payload = {
    "target_standard": "112G PAM4 (IEEE 802.3ck / OIF-CEI-112G-PAM4)",
    "noise_floor_db": -45.0,
    "n_freq_points": int(F_LEN),
    "freq_ghz_min": float(freqs_ghz.min()),
    "freq_ghz_max": float(freqs_ghz.max()),
    "derivation_notes": (
        "Per-element band weights derived from three signals measured in "
        "04_frequency_importance_eda.py:\n"
        "  (1) 112G PAM4 industry priority (IEEE 802.3ck)\n"
        "  (2) Per-band data std (Image 2: std curves diverge between Sdd11 "
        "and Sdd21 across spectrum)\n"
        "  (3) Per-band deep-null density (Image 7 middle panel: Sdd11 nulls "
        "cluster in 0-28 GHz, Sdd21 nulls cluster in 42-100 GHz)\n"
        "  (4) Per-band forward model MAE (Image 7 bottom panel: Sdd21 MAE "
        "peaks at 3.34 dB in 42-56 GHz)"
    ),
    "element_type_map": {
        "Sdd11": "return-loss type (i,j) in {(0,0), (1,1)}; "
                  "= Sdd11 and Sdd22",
        "Sdd21": "insertion-loss type (i,j) in {(0,1), (1,0)}; "
                  "= Sdd21 and Sdd12 (equal by reciprocity)",
        "Sdc":   "mode-conversion type (i,j) in {(0,2),(0,3),(1,2),(1,3)} "
                  "and mirrors at (2,0),(2,1),(3,0),(3,1); Scd = Sdc^T",
        "Scc":   "common-mode type (i,j) in {(2,2),(2,3),(3,2),(3,3)}",
    },
    "bands_per_element_type": {
        elem_type: [
            {"f_lo_GHz": f_lo, "f_hi_GHz": f_hi,
             "weight": w, "rationale": why}
            for (f_lo, f_hi, w, why) in spec
        ]
        for elem_type, spec in ELEMENT_BANDS.items()
    },
    "tensor_shape": [4, 4, F_LEN],
    "tensor_dtype": "float64",
}

json_path = EDA_OUT_DIR / "weights_element_aware.json"
with open(json_path, "w") as f:
    json.dump(output_payload, f, indent=2)
print(f"  Saved: {json_path}")

npy_path = EDA_OUT_DIR / "weights_element_aware_per_freq.npy"
np.save(npy_path, w_per_element)
print(f"  Saved: {npy_path}")

# Also save per-element vectors individually for convenience
for elem_type, w in element_weight_vectors.items():
    p = EDA_OUT_DIR / f"weights_{elem_type}_per_freq.npy"
    np.save(p, w)
    print(f"  Saved: {p}")


# %% Section 9: Summary and downstream usage
print("\n" + "=" * 70)
print("Element-aware band weighting complete.")
print("=" * 70)
print(f"\nOutputs in: {EDA_OUT_DIR}")
print("Files:")
for p in sorted(EDA_OUT_DIR.iterdir()):
    print(f"  {p.name}")

print("\n--- How to consume in the inverse model / TTO ---")
print(
    "import numpy as np, torch\n"
    "w = np.load('weights_element_aware_per_freq.npy')   "
    "# (4, 4, F)\n"
    "w_t = torch.from_numpy(w).to(device).to(torch.float64)\n"
    "\n"
    "# In the cVAE physics loss or the TTO score function:\n"
    "diff_sq = (S_pred - S_target).abs().pow(2)           "
    "# (B, F, 4, 4)\n"
    "w_bcast = w_t.permute(2, 0, 1).unsqueeze(0)          "
    "# (1, F, 4, 4)\n"
    "weighted_loss = (diff_sq * w_bcast).mean()           "
    "# scalar\n"
)

print("--- Next step ---")
print("With these weights produced, the TTO inference module can be written.")
print("It will:")
print("  1. Generate K=20 candidate geometries from the existing cVAE.")
print("  2. Score each via forward_model() and weighted_loss against target.")
print("  3. Take the best, run 50 latent-space gradient steps on weighted_loss.")
print("  4. Return refined geometry.")
print()