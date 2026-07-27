"""
get_val_samples.py
------------------
Identifies exact dataset indices that belong to the VALIDATION set.
Reads the raw parameter.csv to calculate the number of ports 
(SIGNAL_AMOUNT * 2) for each simulation.
Recommends low-port validation samples for faster OpenEMS verification.
"""
from pathlib import Path
import numpy as np
import torch
import csv
import random

# 1. Setup paths
THIS_DIR = Path(__file__).resolve().parent
PROJ_ROOT = THIS_DIR.parent.parent
DATA_PT = PROJ_ROOT / "data" / "processed" / "Universal-Diff-SI-Array" / "diff_pair_dataset.pt"
CSV_PATH = PROJ_ROOT / "data" / "raw" / "Universal-Diff-SI-Array" / "parameter.csv"
SEED = 42  # Must match your training seed

def main():
    print(f"Loading dataset...")
    payload = torch.load(DATA_PT, map_location="cpu", weights_only=False)
    
    sim_ids_raw = payload["sim_ids"]
    if torch.is_tensor(sim_ids_raw):
        sim_ids_raw = sim_ids_raw.cpu().numpy()
    sim_ids = np.array(sim_ids_raw)

    # 2. Replicate the train/val split logic verbatim
    unique_sims = np.unique(sim_ids)
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique_sims)
    
    n_train = int(0.85 * len(unique_sims))
    val_sims = set(unique_sims[n_train:])

    print("Reading parameter.csv to calculate port counts (SIGNAL_AMOUNT * 2)...")
    port_map = {}
    with open(CSV_PATH, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sim_id = row['SIMULATION'].strip()
                # 1 signal via = 2 ports (Top and Bottom)
                n_ports = int(float(row['SIGNAL_AMOUNT']) * 2)
                port_map[sim_id] = n_ports
            except KeyError:
                pass

    # 3. Find validation indices and map their port counts
    val_data = []
    for idx, sid in enumerate(sim_ids):
        if sid in val_sims:
            n_ports = port_map.get(sid, 999) # Default to high if not found
            val_data.append({"index": idx, "sim_id": sid, "n_ports": n_ports})

    # 4. Filter for FAST validation samples (e.g., <= 20 ports)
    fast_val_data = [d for d in val_data if d["n_ports"] <= 20]
    
    print("\n" + "="*55)
    print(f"Total Validation Samples found : {len(val_data)}")
    print(f"FAST Validation Samples (<=20 ports) : {len(fast_val_data)}")
    print("="*55)
    
    # 5. Print a table of 15 random FAST validation samples
    random.seed(42)
    selected_samples = random.sample(fast_val_data, min(15, len(fast_val_data)))
    selected_samples.sort(key=lambda x: x["index"])
    
    print(f"{'INDEX':<10} | {'SIM_ID':<18} | {'N_PORTS':<10}")
    print("-" * 45)
    for sample in selected_samples:
        print(f"{sample['index']:<10} | {sample['sim_id']:<18} | {sample['n_ports']:<10}")

    print("\n" + "="*55)
    print("Recommended command (Testing 4 fast validation targets):")
    idxs = [str(s['index']) for s in selected_samples[:4]]
    print(f"python3 yield_aware_inverse.py --samples {' '.join(idxs)} --restarts 12 --lambda-j 0.0")

if __name__ == "__main__":
    main()