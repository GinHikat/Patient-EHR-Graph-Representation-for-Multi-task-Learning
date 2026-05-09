"""
build_drug_vocab.py
===================
One-time preprocessing step — run before training.
Builds top-50 drug vocab and computes per-drug pos_weight from train admissions.

Output:
    top50_drug_vocab.json       — drug_name -> index mapping (0-49)
    drug_rec_pos_weights.npy    — (50,) pos_weight per drug from train only
"""

import json
import numpy as np
import pandas as pd
from collections import Counter

ADMISSION_NODES_PATH = 'admission_nodes.json'
TRAIN_DF_PATH        = 'train_df.csv'
DRUG_VOCAB_OUT       = 'top50_drug_vocab.json'
DRUG_WEIGHTS_OUT     = 'drug_rec_pos_weights.npy'
N_DRUGS              = 50


def build_drug_vocab_and_weights(admission_nodes, train_df, n_drugs=N_DRUGS):
    """
    Args:
        admission_nodes : dict {adm_id: {'diagnoses': [...], 'drugs': [...]}}
        train_df        : admission DataFrame for train split only
        n_drugs         : top-K drugs to include (default 50)

    Returns:
        drug_to_idx         : dict {drug_name_lower: int 0 to n_drugs-1}
        pos_weight_drug_rec : np.ndarray (n_drugs,) float32
    """

    # Step 1: Count drug frequency across ALL admissions
    drug_counter = Counter(
        d.lower()
        for v in admission_nodes.values()
        for d in v.get('drugs', [])
    )

    top_drugs   = [d for d, _ in drug_counter.most_common(n_drugs)]
    drug_to_idx = {name: i for i, name in enumerate(top_drugs)}

    print(f"Top-{n_drugs} drug vocab built")
    print(f"  Most common : {top_drugs[:5]}")
    print(f"  Least common: {top_drugs[-5:]}")

    # Step 2: Coverage check
    total_adm = len(admission_nodes)
    top_set   = set(top_drugs)
    covered   = sum(
        1 for v in admission_nodes.values()
        if any(d.lower() in top_set for d in v.get('drugs', []))
    )
    print(f"  Coverage    : {100 * covered / total_adm:.1f}% of admissions")

    # Step 3: Compute pos_weight from TRAIN admissions only
    train_adm_ids = set(train_df['id'].astype(str))
    n_train       = len(train_adm_ids)
    pos_counts    = np.zeros(n_drugs, dtype=np.float32)

    for adm_id, v in admission_nodes.items():
        if adm_id not in train_adm_ids:
            continue
        for d in v.get('drugs', []):
            i = drug_to_idx.get(d.lower())
            if i is not None:
                pos_counts[i] += 1

    neg_counts          = n_train - pos_counts
    pos_weight_drug_rec = neg_counts / np.clip(pos_counts, 1, None)
    # max is 10.3 — no capping needed

    print(f"\nDrug pos_weight (from {n_train:,} train admissions):")
    print(f"  min : {pos_weight_drug_rec.min():.2f}")
    print(f"  max : {pos_weight_drug_rec.max():.2f}")
    print(f"  mean: {pos_weight_drug_rec.mean():.2f}")

    return drug_to_idx, pos_weight_drug_rec.astype(np.float32)


if __name__ == '__main__':
    print("Loading admission_nodes...")
    with open(ADMISSION_NODES_PATH) as f:
        admission_nodes = json.load(f)
    print(f"Loaded {len(admission_nodes):,} admissions")

    print("\nLoading train_df...")
    train_df = pd.read_csv(TRAIN_DF_PATH)
    print(f"Loaded {len(train_df):,} train admissions")

    drug_to_idx, pos_weight_drug_rec = build_drug_vocab_and_weights(
        admission_nodes, train_df, n_drugs=N_DRUGS
    )

    with open(DRUG_VOCAB_OUT, 'w') as f:
        json.dump(drug_to_idx, f, indent=2)
    print(f"\nSaved {DRUG_VOCAB_OUT}")

    np.save(DRUG_WEIGHTS_OUT, pos_weight_drug_rec)
    print(f"Saved {DRUG_WEIGHTS_OUT}")

    print("\nDone. Files ready for training:")
    print(f"  {DRUG_VOCAB_OUT}   — drug_name -> index (0-{N_DRUGS-1})")
    print(f"  {DRUG_WEIGHTS_OUT} — ({N_DRUGS},) pos_weight per drug")