Yireld optmization run output

(.venv) dhanushkramesh@Dhanush:~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/sandbox_v1/models$ python3 tto_yield_aware_inverse.py --
samples 0 --mode variance --lambda-j 0 --fit-gate 1.5

====================================================================
  SAMPLE 0  lambda_j=0.0  curriculum=False
====================================================================

  TOLERANCE MODEL [ipc] x scale 1 (1-sigma perturbations; limit=3*sigma Cpk=1 convention for 'ipc'):
          VIA_RADIUS : abs 0.5 mil (limit/3)      sigma_phys=0.5 mil        sigma_norm=0.1073
      ANTIPAD_RADIUS : abs 0.67 mil (limit/3)     sigma_phys=0.67 mil        sigma_norm=0.1207
               PITCH : abs 0.33 mil (limit/3)     sigma_phys=0.33 mil        sigma_norm=0.0386
               TDIEL : rel +/-3.3% (limit/3)      sigma_phys=0.6077 feat-units sigma_norm=0.0277
                TMET : rel +/-3.3% (limit/3)      sigma_phys=0.07233 feat-units sigma_norm=0.0739
        PERMITTIVITY : rel +/-0.67% (limit/3)     sigma_phys=0.01959 feat-units sigma_norm=0.0305
        CONDUCTIVITY : rel +/-3.3% (limit/3)      sigma_phys=0.01423 decades    sigma_norm=0.3035
         LOSSTANGENT : rel +/-3.3% (limit/3)      sigma_phys=0.01423 decades    sigma_norm=0.0351
  (IPC gives ACCEPTANCE LIMITS, not process sigmas; limit/3 is a stated Cpk=1 assumption. CONDUCTIVITY entry is an engineering assumption with no standards anchor. See PartC_Yield_Plan_v3.md.)

=== YIELD TTO [var_L0_ipc] sample 0: 12 restarts x 150 steps
    lambda_j=0.0  [ABLATION: variance proxy]
    spec: eye-band IL >= target-1.0 dB, RL <= -8.0 dB | tol_frac=0.1
  restart 00 | eye-fit  0.70 dB | sens  0.00755 | yield  54.8% [52.6,57.0] | max|x| 1.60
  restart 01 | eye-fit  0.69 dB | sens  0.00743 | yield  53.9% [51.8,56.1] | max|x| 1.59
  restart 02 | eye-fit  0.70 dB | sens  0.00757 | yield  52.1% [49.9,54.3] | max|x| 1.60
  restart 03 | eye-fit  0.70 dB | sens  0.00750 | yield  53.8% [51.6,56.0] | max|x| 1.60
  restart 04 | eye-fit  0.70 dB | sens  0.00757 | yield  53.0% [50.8,55.2] | max|x| 1.60
  restart 05 | eye-fit  2.76 dB | sens  0.00310 | yield   0.0% [0.0,0.2] | max|x| 1.41
  restart 06 | eye-fit  2.76 dB | sens  0.00308 | yield   0.0% [0.0,0.2] | max|x| 1.41
  restart 07 | eye-fit  0.70 dB | sens  0.00758 | yield  54.7% [52.5,56.9] | max|x| 1.60
  restart 08 | eye-fit  0.70 dB | sens  0.00756 | yield  54.9% [52.7,57.0] | max|x| 1.60
  restart 09 | eye-fit  2.76 dB | sens  0.00308 | yield   0.0% [0.0,0.2] | max|x| 1.41
  restart 10 | eye-fit  2.76 dB | sens  0.00309 | yield   0.0% [0.0,0.2] | max|x| 1.41
  restart 11 | eye-fit  2.76 dB | sens  0.00309 | yield   0.0% [0.0,0.2] | max|x| 1.41

  fit gate (1.5 dB eye-band): 7/12 candidates pass
  portfolio diversity: pairwise |dx| mean 0.025, max 0.082 (normalized units; ~0 would mean all restarts found the same design)

======================================================================
  RANKED PORTFOLIO (top 5)  sample 0  [var_L0_ipc]
======================================================================
  rank eye-fit dB  yield %       sens file
     1       0.70     54.9    0.00756 design_sample_0_var_L0_ipc_r08.npz
     2       0.70     54.8    0.00755 design_sample_0_var_L0_ipc_r00.npz
     3       0.70     54.7    0.00758 design_sample_0_var_L0_ipc_r07.npz
     4       0.69     54.0    0.00743 design_sample_0_var_L0_ipc_r01.npz
     5       0.70     53.8    0.00750 design_sample_0_var_L0_ipc_r03.npz

  csv    : portfolio_sample_0_var_L0_ipc_20260718_212517.csv
  pareto : pareto_sample_0_var_L0_ipc_20260718_212517.png
  top design (for stage 07): design_sample_0_var_L0_ipc_top.npz
*****************************

(.venv) dhanushkramesh@Dhanush:~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/sandbox_v1/models$ python3 tto_yield_aware_inverse.py --
samples 0 --mode variance --lambda-j 1.0 --fit-gate 1.5

====================================================================
  SAMPLE 0  lambda_j=1.0  curriculum=False
====================================================================

  TOLERANCE MODEL [ipc] x scale 1 (1-sigma perturbations; limit=3*sigma Cpk=1 convention for 'ipc'):
          VIA_RADIUS : abs 0.5 mil (limit/3)      sigma_phys=0.5 mil        sigma_norm=0.1073
      ANTIPAD_RADIUS : abs 0.67 mil (limit/3)     sigma_phys=0.67 mil        sigma_norm=0.1207
               PITCH : abs 0.33 mil (limit/3)     sigma_phys=0.33 mil        sigma_norm=0.0386
               TDIEL : rel +/-3.3% (limit/3)      sigma_phys=0.6077 feat-units sigma_norm=0.0277
                TMET : rel +/-3.3% (limit/3)      sigma_phys=0.07233 feat-units sigma_norm=0.0739
        PERMITTIVITY : rel +/-0.67% (limit/3)     sigma_phys=0.01959 feat-units sigma_norm=0.0305
        CONDUCTIVITY : rel +/-3.3% (limit/3)      sigma_phys=0.01423 decades    sigma_norm=0.3035
         LOSSTANGENT : rel +/-3.3% (limit/3)      sigma_phys=0.01423 decades    sigma_norm=0.0351
  (IPC gives ACCEPTANCE LIMITS, not process sigmas; limit/3 is a stated Cpk=1 assumption. CONDUCTIVITY entry is an engineering assumption with no standards anchor. See PartC_Yield_Plan_v3.md.)

=== YIELD TTO [var_L1_ipc] sample 0: 12 restarts x 150 steps
    lambda_j=1.0  [ABLATION: variance proxy]
    spec: eye-band IL >= target-1.0 dB, RL <= -8.0 dB | tol_frac=0.1
  restart 00 | eye-fit  0.67 dB | sens  0.00646 | yield  47.9% [45.7,50.1] | max|x| 1.49
  restart 01 | eye-fit  0.67 dB | sens  0.00642 | yield  47.3% [45.1,49.5] | max|x| 1.48
  restart 02 | eye-fit  0.67 dB | sens  0.00644 | yield  45.2% [43.1,47.4] | max|x| 1.49
  restart 03 | eye-fit  0.67 dB | sens  0.00643 | yield  47.5% [45.4,49.7] | max|x| 1.49
  restart 04 | eye-fit  0.67 dB | sens  0.00644 | yield  47.3% [45.2,49.5] | max|x| 1.49
  restart 05 | eye-fit  2.79 dB | sens  0.00296 | yield   0.0% [0.0,0.2] | max|x| 1.38
  restart 06 | eye-fit  2.79 dB | sens  0.00297 | yield   0.0% [0.0,0.2] | max|x| 1.38
  restart 07 | eye-fit  0.67 dB | sens  0.00643 | yield  48.5% [46.3,50.7] | max|x| 1.49
  restart 08 | eye-fit  0.67 dB | sens  0.00647 | yield  48.6% [46.5,50.8] | max|x| 1.49
  restart 09 | eye-fit  2.79 dB | sens  0.00297 | yield   0.0% [0.0,0.2] | max|x| 1.38
  restart 10 | eye-fit  2.79 dB | sens  0.00297 | yield   0.0% [0.0,0.2] | max|x| 1.38
  restart 11 | eye-fit  2.79 dB | sens  0.00298 | yield   0.0% [0.0,0.2] | max|x| 1.38

  fit gate (1.5 dB eye-band): 7/12 candidates pass
  portfolio diversity: pairwise |dx| mean 0.013, max 0.041 (normalized units; ~0 would mean all restarts found the same design)

======================================================================
  RANKED PORTFOLIO (top 5)  sample 0  [var_L1_ipc]
======================================================================
  rank eye-fit dB  yield %       sens file
     1       0.67     48.6    0.00647 design_sample_0_var_L1_ipc_r08.npz
     2       0.67     48.5    0.00643 design_sample_0_var_L1_ipc_r07.npz
     3       0.67     47.9    0.00646 design_sample_0_var_L1_ipc_r00.npz
     4       0.67     47.5    0.00643 design_sample_0_var_L1_ipc_r03.npz
     5       0.67     47.4    0.00644 design_sample_0_var_L1_ipc_r04.npz

  csv    : portfolio_sample_0_var_L1_ipc_20260718_212725.csv
  pareto : pareto_sample_0_var_L1_ipc_20260718_212725.png
  top design (for stage 07): design_sample_0_var_L1_ipc_top.npz
******************************************

(.venv) dhanushkramesh@Dhanush:~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/sandbox_v1/models$ for E in 0.30 0.20 0.10 0.05; do
  python3 tto_yield_aware_inverse.py --samples 0 --mode chance --eps $E --fit-gate 1.5
done

====================================================================
  SAMPLE 0  lambda_j=0.1  curriculum=False
====================================================================

  TOLERANCE MODEL [ipc] x scale 1 (1-sigma perturbations; limit=3*sigma Cpk=1 convention for 'ipc'):
          VIA_RADIUS : abs 0.5 mil (limit/3)      sigma_phys=0.5 mil        sigma_norm=0.1073
      ANTIPAD_RADIUS : abs 0.67 mil (limit/3)     sigma_phys=0.67 mil        sigma_norm=0.1207
               PITCH : abs 0.33 mil (limit/3)     sigma_phys=0.33 mil        sigma_norm=0.0386
               TDIEL : rel +/-3.3% (limit/3)      sigma_phys=0.6077 feat-units sigma_norm=0.0277
                TMET : rel +/-3.3% (limit/3)      sigma_phys=0.07233 feat-units sigma_norm=0.0739
        PERMITTIVITY : rel +/-0.67% (limit/3)     sigma_phys=0.01959 feat-units sigma_norm=0.0305
        CONDUCTIVITY : rel +/-3.3% (limit/3)      sigma_phys=0.01423 decades    sigma_norm=0.3035
         LOSSTANGENT : rel +/-3.3% (limit/3)      sigma_phys=0.01423 decades    sigma_norm=0.0351
  (IPC gives ACCEPTANCE LIMITS, not process sigmas; limit/3 is a stated Cpk=1 assumption. CONDUCTIVITY entry is an engineering assumption with no standards anchor. See PartC_Yield_Plan_v3.md.)

=== YIELD TTO [cc_e0.3_L1_ipc] sample 0: 12 restarts x 150 steps
    eps=0.3 -> kappa=1.53  lambda_cc=1.0  [Cantelli chance constraint]
    spec: eye-band IL >= target-1.0 dB, RL <= -8.0 dB | tol_frac=0.1
  restart 00 | eye-fit  2.04 dB | sens  0.01112 | yield  97.4% [96.6,98.0] | max|x| 2.75
  restart 01 | eye-fit  1.28 dB | sens  0.00412 | yield  84.3% [82.6,85.8] | max|x| 2.62
  restart 02 | eye-fit  3.28 dB | sens  0.01443 | yield  97.2% [96.4,97.9] | max|x| 3.20
  restart 03 | eye-fit  2.14 dB | sens  0.01368 | yield  96.3% [95.4,97.0] | max|x| 2.92
  restart 04 | eye-fit  1.84 dB | sens  0.00990 | yield  96.8% [95.9,97.5] | max|x| 3.71
  restart 05 | eye-fit  2.49 dB | sens  0.00967 | yield  98.7% [98.0,99.1] | max|x| 2.81
  restart 06 | eye-fit  2.84 dB | sens  0.01046 | yield  99.4% [98.9,99.6] | max|x| 2.35
  restart 07 | eye-fit  2.03 dB | sens  0.01120 | yield  96.5% [95.7,97.3] | max|x| 3.01
  restart 08 | eye-fit  2.79 dB | sens  0.00523 | yield  90.1% [88.8,91.4] | max|x| 3.65
  restart 09 | eye-fit  2.43 dB | sens  0.01557 | yield  97.0% [96.2,97.7] | max|x| 2.38
  restart 10 | eye-fit  1.28 dB | sens  0.00435 | yield  88.1% [86.6,89.4] | max|x| 2.09
  restart 11 | eye-fit  2.04 dB | sens  0.01224 | yield  98.2% [97.5,98.7] | max|x| 2.62

  fit gate (1.5 dB eye-band): 2/12 candidates pass
  portfolio diversity: pairwise |dx| mean 1.728, max 1.728 (normalized units; ~0 would mean all restarts found the same design)

======================================================================
  RANKED PORTFOLIO (top 5)  sample 0  [cc_e0.3_L1_ipc]
======================================================================
  rank eye-fit dB  yield %       sens file
     1       1.27     88.1    0.00436 design_sample_0_cc_e0.3_L1_ipc_r10.npz
     2       1.28     84.3    0.00412 design_sample_0_cc_e0.3_L1_ipc_r01.npz

  csv    : portfolio_sample_0_cc_e0.3_L1_ipc_20260718_213348.csv
  pareto : pareto_sample_0_cc_e0.3_L1_ipc_20260718_213348.png
  top design (for stage 07): design_sample_0_cc_e0.3_L1_ipc_top.npz

#################
====================================================================
  SAMPLE 0  lambda_j=0.1  curriculum=False
====================================================================

  TOLERANCE MODEL [ipc] x scale 1 (1-sigma perturbations; limit=3*sigma Cpk=1 convention for 'ipc'):
          VIA_RADIUS : abs 0.5 mil (limit/3)      sigma_phys=0.5 mil        sigma_norm=0.1073
      ANTIPAD_RADIUS : abs 0.67 mil (limit/3)     sigma_phys=0.67 mil        sigma_norm=0.1207
               PITCH : abs 0.33 mil (limit/3)     sigma_phys=0.33 mil        sigma_norm=0.0386
               TDIEL : rel +/-3.3% (limit/3)      sigma_phys=0.6077 feat-units sigma_norm=0.0277
                TMET : rel +/-3.3% (limit/3)      sigma_phys=0.07233 feat-units sigma_norm=0.0739
        PERMITTIVITY : rel +/-0.67% (limit/3)     sigma_phys=0.01959 feat-units sigma_norm=0.0305
        CONDUCTIVITY : rel +/-3.3% (limit/3)      sigma_phys=0.01423 decades    sigma_norm=0.3035
         LOSSTANGENT : rel +/-3.3% (limit/3)      sigma_phys=0.01423 decades    sigma_norm=0.0351
  (IPC gives ACCEPTANCE LIMITS, not process sigmas; limit/3 is a stated Cpk=1 assumption. CONDUCTIVITY entry is an engineering assumption with no standards anchor. See PartC_Yield_Plan_v3.md.)

=== YIELD TTO [cc_e0.2_L1_ipc] sample 0: 12 restarts x 150 steps
    eps=0.2 -> kappa=2.00  lambda_cc=1.0  [Cantelli chance constraint]
    spec: eye-band IL >= target-1.0 dB, RL <= -8.0 dB | tol_frac=0.1
  restart 00 | eye-fit  2.69 dB | sens  0.01198 | yield 100.0% [99.7,100.0] | max|x| 3.45
  restart 01 | eye-fit  1.51 dB | sens  0.00399 | yield  91.5% [90.1,92.6] | max|x| 2.81
  restart 02 | eye-fit  3.35 dB | sens  0.01208 | yield  99.3% [98.8,99.6] | max|x| 3.06
  restart 03 | eye-fit  2.64 dB | sens  0.01308 | yield  99.0% [98.5,99.4] | max|x| 3.72
  restart 04 | eye-fit  2.31 dB | sens  0.00994 | yield  99.4% [99.0,99.7] | max|x| 4.68
  restart 05 | eye-fit  3.74 dB | sens  0.00861 | yield  99.0% [98.5,99.4] | max|x| 3.69
  restart 06 | eye-fit  4.12 dB | sens  0.00964 | yield 100.0% [99.7,100.0] | max|x| 2.66
  restart 07 | eye-fit  2.57 dB | sens  0.01202 | yield  99.4% [98.9,99.6] | max|x| 3.54
  restart 08 | eye-fit  2.96 dB | sens  0.00544 | yield  96.5% [95.7,97.3] | max|x| 4.67
  restart 09 | eye-fit  2.79 dB | sens  0.01425 | yield  99.2% [98.8,99.5] | max|x| 3.22
  restart 10 | eye-fit  1.56 dB | sens  0.00396 | yield  94.0% [92.8,94.9] | max|x| 2.40
  restart 11 | eye-fit  2.74 dB | sens  0.01294 | yield  99.5% [99.1,99.7] | max|x| 3.33

  fit gate (1.5 dB eye-band): 0/12 candidates pass
  [warn] nothing passed the gate -- ranking ALL candidates by yield anyway; consider raising --fit-gate or lowering lambda_j
  portfolio diversity: pairwise |dx| mean 5.593, max 8.785 (normalized units; ~0 would mean all restarts found the same design)

======================================================================
  RANKED PORTFOLIO (top 5)  sample 0  [cc_e0.2_L1_ipc]
======================================================================
  rank eye-fit dB  yield %       sens file
     1       2.69    100.0    0.01198 design_sample_0_cc_e0.2_L1_ipc_r00.npz
     2       4.12    100.0    0.00964 design_sample_0_cc_e0.2_L1_ipc_r06.npz
     3       2.74     99.5    0.01294 design_sample_0_cc_e0.2_L1_ipc_r11.npz
     4       2.31     99.4    0.00994 design_sample_0_cc_e0.2_L1_ipc_r04.npz
     5       2.57     99.3    0.01202 design_sample_0_cc_e0.2_L1_ipc_r07.npz

  csv    : portfolio_sample_0_cc_e0.2_L1_ipc_20260718_213931.csv
  pareto : pareto_sample_0_cc_e0.2_L1_ipc_20260718_213931.png
  top design (for stage 07): design_sample_0_cc_e0.2_L1_ipc_top.npz
#####################

====================================================================
  SAMPLE 0  lambda_j=0.1  curriculum=False
====================================================================

  TOLERANCE MODEL [ipc] x scale 1 (1-sigma perturbations; limit=3*sigma Cpk=1 convention for 'ipc'):
          VIA_RADIUS : abs 0.5 mil (limit/3)      sigma_phys=0.5 mil        sigma_norm=0.1073
      ANTIPAD_RADIUS : abs 0.67 mil (limit/3)     sigma_phys=0.67 mil        sigma_norm=0.1207
               PITCH : abs 0.33 mil (limit/3)     sigma_phys=0.33 mil        sigma_norm=0.0386
               TDIEL : rel +/-3.3% (limit/3)      sigma_phys=0.6077 feat-units sigma_norm=0.0277
                TMET : rel +/-3.3% (limit/3)      sigma_phys=0.07233 feat-units sigma_norm=0.0739
        PERMITTIVITY : rel +/-0.67% (limit/3)     sigma_phys=0.01959 feat-units sigma_norm=0.0305
        CONDUCTIVITY : rel +/-3.3% (limit/3)      sigma_phys=0.01423 decades    sigma_norm=0.3035
         LOSSTANGENT : rel +/-3.3% (limit/3)      sigma_phys=0.01423 decades    sigma_norm=0.0351
  (IPC gives ACCEPTANCE LIMITS, not process sigmas; limit/3 is a stated Cpk=1 assumption. CONDUCTIVITY entry is an engineering assumption with no standards anchor. See PartC_Yield_Plan_v3.md.)

=== YIELD TTO [cc_e0.1_L1_ipc] sample 0: 12 restarts x 150 steps
    eps=0.1 -> kappa=3.00  lambda_cc=1.0  [Cantelli chance constraint]
    spec: eye-band IL >= target-1.0 dB, RL <= -8.0 dB | tol_frac=0.1
  restart 00 | eye-fit  3.74 dB | sens  0.01071 | yield 100.0% [99.8,100.0] | max|x| 4.47
  restart 01 | eye-fit  2.18 dB | sens  0.00356 | yield 100.0% [99.7,100.0] | max|x| 3.78
  restart 02 | eye-fit  4.83 dB | sens  0.01580 | yield 100.0% [99.7,100.0] | max|x| 4.71
  restart 03 | eye-fit  3.73 dB | sens  0.01135 | yield 100.0% [99.7,100.0] | max|x| 5.16
  restart 04 | eye-fit  2.06 dB | sens  0.00424 | yield  99.7% [99.3,99.8] | max|x| 5.83
  restart 05 | eye-fit  3.30 dB | sens  0.00515 | yield  98.4% [97.7,98.8] | max|x| 3.74
  restart 06 | eye-fit  2.02 dB | sens  0.00316 | yield  99.9% [99.6,100.0] | max|x| 2.97
  restart 07 | eye-fit  3.46 dB | sens  0.00973 | yield 100.0% [99.8,100.0] | max|x| 5.61
  restart 08 | eye-fit  3.31 dB | sens  0.00565 | yield  99.4% [99.0,99.7] | max|x| 6.57
  restart 09 | eye-fit  3.89 dB | sens  0.01144 | yield 100.0% [99.8,100.0] | max|x| 4.41
  restart 10 | eye-fit  2.09 dB | sens  0.00357 | yield  99.2% [98.7,99.5] | max|x| 2.84
  restart 11 | eye-fit  3.57 dB | sens  0.00973 | yield 100.0% [99.8,100.0] | max|x| 5.39

  fit gate (1.5 dB eye-band): 0/12 candidates pass
  [warn] nothing passed the gate -- ranking ALL candidates by yield anyway; consider raising --fit-gate or lowering lambda_j
  portfolio diversity: pairwise |dx| mean 6.326, max 9.994 (normalized units; ~0 would mean all restarts found the same design)

======================================================================
  RANKED PORTFOLIO (top 5)  sample 0  [cc_e0.1_L1_ipc]
======================================================================
  rank eye-fit dB  yield %       sens file
     1       3.74    100.0    0.01071 design_sample_0_cc_e0.1_L1_ipc_r00.npz
     2       3.46    100.0    0.00974 design_sample_0_cc_e0.1_L1_ipc_r07.npz
     3       3.90    100.0    0.01144 design_sample_0_cc_e0.1_L1_ipc_r09.npz
     4       3.57    100.0    0.00973 design_sample_0_cc_e0.1_L1_ipc_r11.npz
     5       2.19    100.0    0.00356 design_sample_0_cc_e0.1_L1_ipc_r01.npz

  csv    : portfolio_sample_0_cc_e0.1_L1_ipc_20260718_214511.csv
  pareto : pareto_sample_0_cc_e0.1_L1_ipc_20260718_214511.png
  top design (for stage 07): design_sample_0_cc_e0.1_L1_ipc_top.npz

#########
====================================================================
  SAMPLE 0  lambda_j=0.1  curriculum=False
====================================================================

  TOLERANCE MODEL [ipc] x scale 1 (1-sigma perturbations; limit=3*sigma Cpk=1 convention for 'ipc'):
          VIA_RADIUS : abs 0.5 mil (limit/3)      sigma_phys=0.5 mil        sigma_norm=0.1073
      ANTIPAD_RADIUS : abs 0.67 mil (limit/3)     sigma_phys=0.67 mil        sigma_norm=0.1207
               PITCH : abs 0.33 mil (limit/3)     sigma_phys=0.33 mil        sigma_norm=0.0386
               TDIEL : rel +/-3.3% (limit/3)      sigma_phys=0.6077 feat-units sigma_norm=0.0277
                TMET : rel +/-3.3% (limit/3)      sigma_phys=0.07233 feat-units sigma_norm=0.0739
        PERMITTIVITY : rel +/-0.67% (limit/3)     sigma_phys=0.01959 feat-units sigma_norm=0.0305
        CONDUCTIVITY : rel +/-3.3% (limit/3)      sigma_phys=0.01423 decades    sigma_norm=0.3035
         LOSSTANGENT : rel +/-3.3% (limit/3)      sigma_phys=0.01423 decades    sigma_norm=0.0351
  (IPC gives ACCEPTANCE LIMITS, not process sigmas; limit/3 is a stated Cpk=1 assumption. CONDUCTIVITY entry is an engineering assumption with no standards anchor. See PartC_Yield_Plan_v3.md.)

=== YIELD TTO [cc_e0.05_L1_ipc] sample 0: 12 restarts x 150 steps
    eps=0.05 -> kappa=4.36  lambda_cc=1.0  [Cantelli chance constraint]
    spec: eye-band IL >= target-1.0 dB, RL <= -8.0 dB | tol_frac=0.1
  restart 00 | eye-fit  2.85 dB | sens  0.00347 | yield  99.2% [98.6,99.5] | max|x| 5.13
  restart 01 | eye-fit  2.54 dB | sens  0.00265 | yield  99.4% [98.9,99.6] | max|x| 4.04
  restart 02 | eye-fit  6.56 dB | sens  0.01292 | yield 100.0% [99.8,100.0] | max|x| 5.51
  restart 03 | eye-fit  3.39 dB | sens  0.00572 | yield  99.0% [98.4,99.3] | max|x| 6.44
  restart 04 | eye-fit  3.02 dB | sens  0.00350 | yield 100.0% [99.7,100.0] | max|x| 6.80
  restart 05 | eye-fit  3.27 dB | sens  0.00295 | yield   0.1% [0.0,0.4] | max|x| 2.54
  restart 06 | eye-fit  2.62 dB | sens  0.00257 | yield  81.9% [80.2,83.5] | max|x| 3.62
  restart 07 | eye-fit  3.41 dB | sens  0.00593 | yield  99.3% [98.8,99.6] | max|x| 7.58
  restart 08 | eye-fit  3.74 dB | sens  0.00508 | yield 100.0% [99.7,100.0] | max|x| 8.85
  restart 09 | eye-fit  3.13 dB | sens  0.00474 | yield   7.3% [6.3,8.6] | max|x| 5.27
  restart 10 | eye-fit  2.87 dB | sens  0.00306 | yield 100.0% [99.8,100.0] | max|x| 3.40
  restart 11 | eye-fit  3.38 dB | sens  0.00587 | yield 100.0% [99.8,100.0] | max|x| 7.47

  fit gate (1.5 dB eye-band): 0/12 candidates pass
  [warn] nothing passed the gate -- ranking ALL candidates by yield anyway; consider raising --fit-gate or lowering lambda_j
  portfolio diversity: pairwise |dx| mean 6.264, max 11.190 (normalized units; ~0 would mean all restarts found the same design)

======================================================================
  RANKED PORTFOLIO (top 5)  sample 0  [cc_e0.05_L1_ipc]
======================================================================
  rank eye-fit dB  yield %       sens file
     1       6.56    100.0    0.01292 design_sample_0_cc_e0.05_L1_ipc_r02.npz
     2       2.87    100.0    0.00306 design_sample_0_cc_e0.05_L1_ipc_r10.npz
     3       3.38    100.0    0.00587 design_sample_0_cc_e0.05_L1_ipc_r11.npz
     4       3.02    100.0    0.00350 design_sample_0_cc_e0.05_L1_ipc_r04.npz
     5       3.74    100.0    0.00507 design_sample_0_cc_e0.05_L1_ipc_r08.npz

  csv    : portfolio_sample_0_cc_e0.05_L1_ipc_20260718_215053.csv
  pareto : pareto_sample_0_cc_e0.05_L1_ipc_20260718_215053.png
  top design (for stage 07): design_sample_0_cc_e0.05_L1_ipc_top.npz
*******************************

(.venv) dhanushkramesh@Dhanush:~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/sandbox_v1/models$ python3 tto_yield_aware_inverse.py --
samples 0 --mode worstcase --fit-gate 1.5

====================================================================
  SAMPLE 0  lambda_j=0.1  curriculum=False
====================================================================

  TOLERANCE MODEL [ipc] x scale 1 (1-sigma perturbations; limit=3*sigma Cpk=1 convention for 'ipc'):
          VIA_RADIUS : abs 0.5 mil (limit/3)      sigma_phys=0.5 mil        sigma_norm=0.1073
      ANTIPAD_RADIUS : abs 0.67 mil (limit/3)     sigma_phys=0.67 mil        sigma_norm=0.1207
               PITCH : abs 0.33 mil (limit/3)     sigma_phys=0.33 mil        sigma_norm=0.0386
               TDIEL : rel +/-3.3% (limit/3)      sigma_phys=0.6077 feat-units sigma_norm=0.0277
                TMET : rel +/-3.3% (limit/3)      sigma_phys=0.07233 feat-units sigma_norm=0.0739
        PERMITTIVITY : rel +/-0.67% (limit/3)     sigma_phys=0.01959 feat-units sigma_norm=0.0305
        CONDUCTIVITY : rel +/-3.3% (limit/3)      sigma_phys=0.01423 decades    sigma_norm=0.3035
         LOSSTANGENT : rel +/-3.3% (limit/3)      sigma_phys=0.01423 decades    sigma_norm=0.0351
  (IPC gives ACCEPTANCE LIMITS, not process sigmas; limit/3 is a stated Cpk=1 assumption. CONDUCTIVITY entry is an engineering assumption with no standards anchor. See PartC_Yield_Plan_v3.md.)

=== YIELD TTO [wc_L1_ipc] sample 0: 12 restarts x 150 steps
    lambda_cc=1.0  [Wang/Sigmund worst-case corners]
    spec: eye-band IL >= target-1.0 dB, RL <= -8.0 dB | tol_frac=0.1
  restart 00 | eye-fit  1.29 dB | sens  0.01061 | yield  84.0% [82.3,85.5] | max|x| 1.77
  restart 01 | eye-fit  1.06 dB | sens  0.00542 | yield  85.3% [83.7,86.8] | max|x| 1.81
  restart 02 | eye-fit  1.42 dB | sens  0.01050 | yield  88.4% [86.9,89.7] | max|x| 1.78
  restart 03 | eye-fit  1.18 dB | sens  0.00842 | yield  82.8% [81.1,84.4] | max|x| 2.29
  restart 04 | eye-fit  1.33 dB | sens  0.00915 | yield  84.5% [82.8,86.0] | max|x| 1.65
  restart 05 | eye-fit  2.20 dB | sens  0.00706 | yield  81.8% [80.0,83.4] | max|x| 2.72
  restart 06 | eye-fit  2.67 dB | sens  0.00570 | yield  68.8% [66.7,70.8] | max|x| 2.02
  restart 07 | eye-fit  1.00 dB | sens  0.00461 | yield  65.0% [62.8,67.0] | max|x| 1.74
  restart 08 | eye-fit  1.30 dB | sens  0.01022 | yield  84.7% [83.0,86.2] | max|x| 1.79
  restart 09 | eye-fit  1.41 dB | sens  0.00712 | yield  84.8% [83.2,86.3] | max|x| 2.43
  restart 10 | eye-fit  1.50 dB | sens  0.01272 | yield  86.9% [85.3,88.3] | max|x| 1.88
  restart 11 | eye-fit  1.24 dB | sens  0.01136 | yield  78.8% [77.0,80.5] | max|x| 2.11

  fit gate (1.5 dB eye-band): 10/12 candidates pass
  portfolio diversity: pairwise |dx| mean 3.275, max 5.224 (normalized units; ~0 would mean all restarts found the same design)

======================================================================
  RANKED PORTFOLIO (top 5)  sample 0  [wc_L1_ipc]
======================================================================
  rank eye-fit dB  yield %       sens file
     1       1.42     88.4    0.01050 design_sample_0_wc_L1_ipc_r02.npz
     2       1.50     86.8    0.01272 design_sample_0_wc_L1_ipc_r10.npz
     3       1.06     85.3    0.00542 design_sample_0_wc_L1_ipc_r01.npz
     4       1.41     84.8    0.00712 design_sample_0_wc_L1_ipc_r09.npz
     5       1.30     84.7    0.01022 design_sample_0_wc_L1_ipc_r08.npz

  csv    : portfolio_sample_0_wc_L1_ipc_20260718_215435.csv
  pareto : pareto_sample_0_wc_L1_ipc_20260718_215435.png
  top design (for stage 07): design_sample_0_wc_L1_ipc_top.npz