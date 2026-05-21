"""
make_splits.py — train/val/test split for the Universal-Diff-SI-Array dataset.

Rules:
  1. Split is by sim_id, never by pair. All pairs from one sim land in the
     same split. This is the leakage rule from the handover.
  2. Stratified by num_ports so train/val/test each see the same distribution
     of array sizes. Without stratification, an unlucky split can put all
     32-port arrays in test and the model fails for reasons unrelated to its
     architecture.
  3. 80/10/10. Seed 0. Saved once, used by every downstream experiment.
  4. Output includes both sim_id-level and row-level index arrays. Row-level
     indices index directly into the .pt tensors.

Run once. Re-running with the same seed produces the same split (verified by
asserting equality if the file already exists).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path.home() / "mece_project_inverse_model" / "Generative_Inverse_Design_of_High-Speed_Interconnects"
PT_PATH = PROJECT_ROOT / "data" / "processed" / "Universal-Diff-SI-Array" / "diff_pair_dataset.pt"
SPLIT_PATH = PROJECT_ROOT / "sandbox_v1" / "data" / "splits.pt"

SEED = 0
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
# Test fraction is the remainder; 0.10 by arithmetic.


def main():
    print(f"Loading {PT_PATH}")
    payload = torch.load(PT_PATH, weights_only=False)
    assert payload["dataset_type"] == "Array"

    sim_ids = np.array(payload["sim_ids"])
    num_ports = payload["num_ports"].numpy()
    N_pairs = len(sim_ids)

    # Build sim-level table: one row per unique sim_id, with its num_ports
    sim_to_ports = {}
    for sid, np_ in zip(sim_ids, num_ports):
        sim_to_ports.setdefault(sid, np_)  # all pairs of one sim share num_ports (audited)
    unique_sims = np.array(sorted(sim_to_ports.keys()))
    sim_ports = np.array([sim_to_ports[s] for s in unique_sims])
    N_sims = len(unique_sims)
    print(f"  {N_pairs} pairs across {N_sims} unique sims")
    print(f"  num_ports distribution across sims:")
    for p, c in sorted(Counter(sim_ports).items()):
        print(f"    {p} ports: {c} sims")

    # Stratified split by num_ports
    rng = np.random.default_rng(SEED)
    train_sims, val_sims, test_sims = [], [], []
    for p in sorted(set(sim_ports.tolist())):
        sims_at_p = unique_sims[sim_ports == p]
        sims_at_p_shuffled = rng.permutation(sims_at_p)
        n = len(sims_at_p_shuffled)
        n_train = int(round(n * TRAIN_FRAC))
        n_val = int(round(n * VAL_FRAC))
        # Remainder to test (avoids rounding drift)
        train_sims.extend(sims_at_p_shuffled[:n_train].tolist())
        val_sims.extend(sims_at_p_shuffled[n_train:n_train + n_val].tolist())
        test_sims.extend(sims_at_p_shuffled[n_train + n_val:].tolist())

    train_sims = set(train_sims)
    val_sims = set(val_sims)
    test_sims = set(test_sims)

    # Sanity: every sim in exactly one split
    assert train_sims.isdisjoint(val_sims)
    assert train_sims.isdisjoint(test_sims)
    assert val_sims.isdisjoint(test_sims)
    assert len(train_sims) + len(val_sims) + len(test_sims) == N_sims

    # Build row-level (per-pair) index arrays
    train_idx = np.array([i for i, s in enumerate(sim_ids) if s in train_sims], dtype=np.int64)
    val_idx = np.array([i for i, s in enumerate(sim_ids) if s in val_sims], dtype=np.int64)
    test_idx = np.array([i for i, s in enumerate(sim_ids) if s in test_sims], dtype=np.int64)

    assert len(train_idx) + len(val_idx) + len(test_idx) == N_pairs

    # Verify no pair leakage by sim_id
    train_sim_set = set(sim_ids[train_idx])
    val_sim_set = set(sim_ids[val_idx])
    test_sim_set = set(sim_ids[test_idx])
    assert train_sim_set.isdisjoint(val_sim_set)
    assert train_sim_set.isdisjoint(test_sim_set)
    assert val_sim_set.isdisjoint(test_sim_set)

    # Distribution check
    print(f"\n  split    sims  pairs    pairs/sim")
    for name, idx, sset in [("train", train_idx, train_sims),
                            ("val  ", val_idx, val_sims),
                            ("test ", test_idx, test_sims)]:
        print(f"  {name}  {len(sset):5d}  {len(idx):5d}    {len(idx)/max(len(sset),1):.2f}")

    print(f"\n  num_ports distribution per split (pairs):")
    train_ports = num_ports[train_idx]
    val_ports = num_ports[val_idx]
    test_ports = num_ports[test_idx]
    all_p = sorted(set(num_ports.tolist()))
    print(f"    {'ports':>6} {'train':>8} {'val':>6} {'test':>6}")
    for p in all_p:
        nt = int((train_ports == p).sum())
        nv = int((val_ports == p).sum())
        ne = int((test_ports == p).sum())
        print(f"    {p:>6} {nt:>8} {nv:>6} {ne:>6}")

    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "seed": SEED,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "stratified_by": "num_ports",
        "train_idx": torch.tensor(train_idx, dtype=torch.int64),
        "val_idx": torch.tensor(val_idx, dtype=torch.int64),
        "test_idx": torch.tensor(test_idx, dtype=torch.int64),
        "train_sim_ids": sorted(train_sim_set),
        "val_sim_ids": sorted(val_sim_set),
        "test_sim_ids": sorted(test_sim_set),
        "n_pairs_total": int(N_pairs),
        "n_sims_total": int(N_sims),
    }

    # If the file already exists with the same seed, verify identical split
    if SPLIT_PATH.exists():
        prev = torch.load(SPLIT_PATH, weights_only=False)
        if prev.get("seed") == SEED:
            same = (torch.equal(prev["train_idx"], artifact["train_idx"])
                    and torch.equal(prev["val_idx"], artifact["val_idx"])
                    and torch.equal(prev["test_idx"], artifact["test_idx"]))
            if same:
                print(f"\nSplit file already exists and matches. Not overwriting: {SPLIT_PATH}")
                return
            else:
                raise RuntimeError(
                    f"Existing split at {SPLIT_PATH} has same seed but DIFFERENT indices. "
                    f"Refusing to overwrite. Investigate before deleting the old file."
                )

    torch.save(artifact, SPLIT_PATH)
    print(f"\nSaved: {SPLIT_PATH}")


if __name__ == "__main__":
    main()