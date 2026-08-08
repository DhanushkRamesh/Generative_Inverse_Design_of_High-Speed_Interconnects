from pathlib import Path
import pandas as pd

# 1. Setup absolute paths
THIS_DIR = Path(__file__).resolve().parent
PROJ_ROOT = THIS_DIR.parent

PARAM_CSV = PROJ_ROOT / "data" / "raw" / "Universal-Diff-SI-Array" / "parameter.csv"
EVAL_CSV = PROJ_ROOT / "sandbox_v1" / "models" / "validation_eval.csv"

# Load the data
print("Loading CSV files...")
param_df = pd.read_csv(PARAM_CSV)
eval_df = pd.read_csv(EVAL_CSV)

# Ensure SIMULATION ID is treated as a string for merging
param_df['SIMULATION'] = param_df['SIMULATION'].astype(str)
eval_df['sim_id'] = eval_df['sim_id'].astype(str) 

# 1. Filter for FAST simulations (SIGNAL_AMOUNT <= 8 means 16 ports max)
# We relaxed this from 4 to 8 to catch more of your 50 random samples!
fast_sims = param_df[param_df['SIGNAL_AMOUNT'] <= 12]['SIMULATION'].tolist()

# 2. Filter the evaluation data to ONLY include fast simulations
fast_eval_df = eval_df[eval_df['sim_id'].isin(fast_sims)].copy()

# 3. Sort by Yield % to make selection easy
fast_eval_df = fast_eval_df.sort_values(by='wc_yield_pct', ascending=False)

print("\n=== FAST TO SIMULATE CANDIDATES (16 PORTS OR LESS) ===")

print("\n--- BEST YIELD CANDIDATES (> 90%) ---")
best = fast_eval_df[(fast_eval_df['wc_yield_pct'] >= 90) & (fast_eval_df['wc_fit_dB'] < 2.0)]
print(best[['sample', 'sim_id', 'wc_yield_pct', 'wc_fit_dB']])

print("\n--- MODERATE YIELD CANDIDATES (50% - 80%) ---")
moderate = fast_eval_df[(fast_eval_df['wc_yield_pct'] >= 50) & (fast_eval_df['wc_yield_pct'] <= 80)]
print(moderate[['sample', 'sim_id', 'wc_yield_pct', 'wc_fit_dB']])

print("\n--- WORST YIELD CANDIDATES (< 20%) ---")
worst = fast_eval_df[(fast_eval_df['wc_yield_pct'] < 20) & (fast_eval_df['wc_fit_dB'] < 3.0)]
print(worst[['sample', 'sim_id', 'wc_yield_pct', 'wc_fit_dB']])