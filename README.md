# Signal-Integrity-Surrogate-Model
ML-based surrogate model for Signal Integrity using Neural Networks. Features a Forward Model (Physical -> S parameters) and Inverse Model (S-Params -> Physical) for high-speed circuit design optimization.
```text
├── data/                       # Raw and processed TUHH SI/PI Database files
│   ├── processed/
│   └── raw/
├── models/                     # Neural network architectures, training, and TTO scripts
│   ├── train_direct_sequence_resnet.py          # Forward FDTD surrogate training
│   ├── train_tandem_cvae_inverse_masked.py      # Generative cVAE inverse model
│   ├── tto_latent_inverse.py                    # Nominal-fit latent space TTO
│   └── tto_yield_aware_inverse.py               # Worst-case yield-aware TTO
├── notebooks/                  # Jupyter notebooks for EDA and thesis visualization
│   ├── raw_data_eda.ipynb
│   ├── yield_aware_design_evaluation.ipynb      # Computes Table II / Table III statistics
│   └── fig3_reconstruction_histogram.ipynb      # Generates publication-ready KDE plots
├── openEMS_sim/                # Full-wave FDTD verification and time-domain tools
│   ├── baseline_known_geometry_best.py          # FDTD execution for target geometry baselines
│   ├── build_array_model_mur_final.py           # 3D via builder with Mur absorbing boundaries
│   ├── eye_diagram_validation.py                # 112G PAM4 PRBS13Q eye diagram generator
│   └── parse_geometry.py                        # Bridges latent output to OpenEMS XML
├── results/                    # Aggregated simulation logs, model weights, and CSV evaluations
│   ├── data/
│   └── models/evaluation_results/
└── utils/                      # Shared helper functions
    └── physics_utils.py        # Passivity, reciprocity, and Mixed-Mode (BE) conversions
