import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import re
import json
import pickle
from pathlib import Path

import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

base_data_dir = os.path.join(project_root, 'data')
downstream_data_path = os.path.join(base_data_dir, 'downstream')

from shared_functions.global_functions import *

data_dir = os.getenv('DATA_DIR')

# Load all patient IDs for processing
with open(os.path.join(downstream_data_path, 'patients.txt')) as f:
    all_patient_ids = [int(line.strip()) for line in f.readlines()]

OMR_FIELDS = ['blood_pressure_systolic', 'blood_pressure_diastolic', 'Weight', 'BMI']
OMR_SIZE   = len(OMR_FIELDS)  # 4
OMR_NON_VALUE_FIELDS = {'id', 'name', 'patient_id', 'date'}

OMR_OUTPUT_DIR  = os.path.join(data_dir, 'OMR_Embedding')
OMR_SCALER_PATH = 'omr_scaler.pkl'
OMR_CHECKPOINT  = 'omr_checkpoint.pkl'

def omr_to_vectors(panel: dict):
    value_vec = np.zeros(OMR_SIZE, dtype=np.float32)
    mask_vec  = np.zeros(OMR_SIZE, dtype=np.float32)

    for i, field in enumerate(OMR_FIELDS):
        val = panel.get(field)
        if val is None:
            continue
        try:
            fval = float(val)
            if np.isnan(fval) or np.isinf(fval):  
                continue
            value_vec[i] = fval
            mask_vec[i]  = 1.0
        except (ValueError, TypeError):
            pass

    return value_vec, mask_vec, panel.get('date')

def get_omr_for_batch(patient_ids: list[int]):
    result = query_neo4j(
        '''
        MATCH (n:OMR)
        WHERE n.patient_id IN $patient_ids
        RETURN n { .* } AS panel
        ORDER BY n.patient_id, n.date
        ''',
        patient_ids=patient_ids
    )
    return [row['panel'] for row in result]

def fit_and_build_omr(patient_ids, output_dir=OMR_OUTPUT_DIR, batch_size=500,
                      save_path=OMR_SCALER_PATH, checkpoint_path=OMR_CHECKPOINT):
    """
    Fit scaler using Welford's algorithm across multiple batches.
    
    Args:
        patient_ids (list[int]): List of patient IDs to process.
        output_dir (str): Directory to save computed OMR embeddings.
        batch_size (int): Number of patients per batch.
        save_path (str): Path to save the final scaler.
        checkpoint_path (str): Path to save intermediate checkpoints.
        
    Returns:
        dict: The computed scaler containing 'mean', 'std', and 'count'.
    """
    
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    if Path(checkpoint_path).exists():
        with open(checkpoint_path, 'rb') as f:
            ckpt = pickle.load(f)
        count     = ckpt['count']
        mean      = ckpt['mean']
        M2        = ckpt['M2']
        start_idx = ckpt['next_batch_idx']
        skipped   = ckpt['skipped']
        print(f'Resuming from batch {start_idx // batch_size} '
              f'({start_idx}/{len(patient_ids)} patients done)')
    else:
        count     = np.zeros(OMR_SIZE, dtype=np.float64)
        mean      = np.zeros(OMR_SIZE, dtype=np.float64)
        M2        = np.zeros(OMR_SIZE, dtype=np.float64)
        start_idx = 0
        skipped   = 0
        print('Starting fresh')

    if start_idx < len(patient_ids):
        for i in tqdm(range(start_idx, len(patient_ids), batch_size),
                      desc='Pass 1 — OMR fit scaler',
                      initial=start_idx // batch_size,
                      total=len(patient_ids) // batch_size):
            batch  = patient_ids[i:i+batch_size]
            panels = get_omr_for_batch(batch)

            by_patient = {}
            for panel in panels:
                pid = panel['patient_id']
                by_patient.setdefault(pid, []).append(panel)

            for pid, patient_panels in by_patient.items():
                if (out / f'{pid}.npz').exists():
                    continue

                patient_panels.sort(key=lambda p: p.get('date', ''))
                values_list, masks_list, times_list = [], [], []

                for panel in patient_panels:
                    v, m, date = omr_to_vectors(panel)

                    if date is None or int(m.sum()) == 0:
                        skipped += 1
                        continue

                    # ── Welford update with NaN guard ─────────────────────────
                    for idx in np.where(m == 1.0)[0]:
                        if np.isnan(v[idx]) or np.isinf(v[idx]):  # ← guard
                            continue
                        count[idx] += 1
                        delta       = v[idx] - mean[idx]
                        mean[idx]  += delta / count[idx]
                        M2[idx]    += delta * (v[idx] - mean[idx])

                    values_list.append(v)
                    masks_list.append(m)
                    times_list.append(str(date))

                if values_list:
                    np.savez_compressed(
                        out / f'{pid}.npz',
                        values=np.stack(values_list).astype(np.float32),
                        masks =np.stack(masks_list).astype(np.float32)
                    )
                    with open(out / f'{pid}_times.json', 'w') as f:
                        json.dump(times_list, f)

            tmp = checkpoint_path + '.tmp'
            with open(tmp, 'wb') as f:
                pickle.dump({
                    'count':          count,
                    'mean':           mean,
                    'M2':             M2,
                    'next_batch_idx': i + batch_size,
                    'skipped':        skipped
                }, f)
            os.replace(tmp, checkpoint_path)

        print(f'Pass 1 done. Skipped {skipped} panels.')
    else:
        print('Pass 1 already complete, skipping.')

    if Path(save_path).exists():
        print(f'Scaler already exists at {save_path}, loading.')
        with open(save_path, 'rb') as f:
            scaler = pickle.load(f)
    else:
        variance = np.where(count > 1, M2 / (count - 1), 1.0)
        stds     = np.where(np.sqrt(variance) > 0, np.sqrt(variance), 1.0)

        # ── Final NaN/Inf guard on scaler stats ───────────────────────────────
        means_clean = np.nan_to_num(mean, nan=0.0).astype(np.float32)
        stds_clean  = np.nan_to_num(stds, nan=1.0, posinf=1.0).astype(np.float32)
        stds_clean  = np.clip(stds_clean, 1e-6, 1e6)

        scaler = {
            'mean':  means_clean,
            'std':   stds_clean,
            'count': count.astype(np.int64)
        }
        with open(save_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f'Scaler saved → {save_path}')
        print(f'Field counts: { {f: int(count[i]) for i, f in enumerate(OMR_FIELDS)} }')

    npz_files  = list(out.glob('*.npz'))
    needs_norm = [f for f in npz_files
                  if not (out / f.name.replace('.npz', '_done')).exists()]

    print(f'Pass 2: {len(needs_norm)} files to normalize, '
          f'{len(npz_files) - len(needs_norm)} already done.')

    for fpath in tqdm(needs_norm, desc='Pass 2 — normalizing'):
        data   = np.load(fpath)
        values = data['values']
        masks  = data['masks']

        norm_values = (values - scaler['mean']) / scaler['std']
        norm_values = np.nan_to_num(norm_values, nan=0.0)  # ← final safety net
        norm_values = norm_values * masks

        np.savez_compressed(fpath, values=norm_values.astype(np.float32), masks=masks)
        (out / fpath.name.replace('.npz', '_done')).touch()

    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
    for f in out.glob('*_done'):
        f.unlink()

    print('All done.')
    return scaler

if __name__ == '__main__':
    fit_and_build_omr(all_patient_ids)