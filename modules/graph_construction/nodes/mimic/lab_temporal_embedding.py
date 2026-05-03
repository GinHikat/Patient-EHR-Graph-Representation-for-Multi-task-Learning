import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import re
import json
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from shared_functions.global_functions import *

# Load Lab vocab for vector generation
with open('lab_vocab.json', 'r') as f:
    LAB_VOCAB = json.load(f)

# Load all patient IDs for processing
with open('patients.txt') as f:
    all_patient_ids = [int(line.strip()) for line in f.readlines()]

VOCAB_SIZE = len(LAB_VOCAB)  # 170
NON_VALUE_FIELDS = {'id', 'name', 'patient_id', 'admission_id', 'charttime'}
BATCH_SIZE = 500

data_dir = os.getenv('DATA_DIR')
mimic_path = os.path.join(data_dir, 'mimic_iv')
hosp = os.path.join(mimic_path, 'hosp')

OUTPUT_DIR = os.path.join(hosp, 'Lab_Embedding')
os.makedirs(OUTPUT_DIR, exist_ok = True)
SCALER_PATH = 'lab_scaler.pkl'
CHECKPOINT_PATH = 'fit_checkpoint.pkl'

def get_all_patient_ids(batch_size=10000) -> list[int]:
    all_ids = []
    skip = 0

    with tqdm(desc='Fetching patient IDs') as pbar:
        while True:
            result = query_neo4j(
                '''
                MATCH (r:Result)
                RETURN DISTINCT r.patient_id AS patient_id
                SKIP $skip LIMIT $limit
                ''',
                skip=skip,
                limit=batch_size
            )
            if not result:
                break
            batch_ids = [row['patient_id'] for row in result]
            all_ids.extend(batch_ids)
            skip += batch_size
            pbar.update(len(batch_ids))
            pbar.set_postfix({'total': len(all_ids)})

    return all_ids

def panel_to_vectors(panel):
    value_vec = np.zeros(VOCAB_SIZE, dtype=np.float32)
    mask_vec  = np.zeros(VOCAB_SIZE, dtype=np.float32)
    for field, value in panel.items():
        if field in NON_VALUE_FIELDS or field not in LAB_VOCAB or value is None:
            continue
        try:
            value_vec[LAB_VOCAB[field]] = float(value)
            mask_vec[LAB_VOCAB[field]]  = 1.0
        except (ValueError, TypeError):
            pass
    return value_vec, mask_vec, panel.get('charttime')

def get_panels_for_batch(patient_ids):
    result = query_neo4j(
        '''
        MATCH (r:Result)
        WHERE r.patient_id IN $patient_ids
        RETURN r { .* } AS panel
        ORDER BY r.patient_id, r.charttime
        ''',
        patient_ids=patient_ids
    )
    return [row['panel'] for row in result]

def fit_scaler_streaming(patient_ids, save_path='lab_scaler.pkl'):
    # Welford's online mean/variance — O(1) memory regardless of dataset size
    count = np.zeros(VOCAB_SIZE, dtype=np.float64)
    mean  = np.zeros(VOCAB_SIZE, dtype=np.float64)
    M2    = np.zeros(VOCAB_SIZE, dtype=np.float64)  # sum of squared deviations

    for i in tqdm(range(0, len(patient_ids), BATCH_SIZE), desc='Fitting scaler'):
        batch = patient_ids[i:i+BATCH_SIZE]
        panels = get_panels_for_batch(batch)

        for panel in panels:
            v, m, _ = panel_to_vectors(panel)
            for idx in np.where(m == 1.0)[0]:  # only measured positions
                count[idx] += 1
                delta       = v[idx] - mean[idx]
                mean[idx]  += delta / count[idx]
                M2[idx]    += delta * (v[idx] - mean[idx])

    # Finalize std
    variance = np.where(count > 1, M2 / (count - 1), 1.0)
    stds     = np.where(np.sqrt(variance) > 0, np.sqrt(variance), 1.0)

    scaler = {
        'mean':  mean.astype(np.float32),
        'std':   stds.astype(np.float32),
        'count': count.astype(np.int64)   # useful to keep — tells you how often each test appears
    }
    with open(save_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f'Scaler saved → {save_path}')
    print(f'Tests with < 100 observations: {int((count < 100).sum())} — consider reviewing vocab')
    return scaler

def fit_and_build(patient_ids, output_dir=OUTPUT_DIR, batch_size=500,
                  save_path=SCALER_PATH, checkpoint_path=CHECKPOINT_PATH):
                  
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    # Resume checkpoint 
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
        count     = np.zeros(VOCAB_SIZE, dtype=np.float64)
        mean      = np.zeros(VOCAB_SIZE, dtype=np.float64)
        M2        = np.zeros(VOCAB_SIZE, dtype=np.float64)
        start_idx = 0
        skipped   = 0
        print('Starting fresh')

    # Pass 1: fit scaler + write raw panels
    if start_idx < len(patient_ids):
        batches = range(start_idx, len(patient_ids), batch_size)
        for i in tqdm(batches, desc='Pass 1 — fit scaler',
                      initial=start_idx // batch_size,
                      total=len(patient_ids) // batch_size):
            batch = patient_ids[i:i+batch_size]
            panels = get_panels_for_batch(batch)

            by_patient = {}
            for panel in panels:
                pid = panel['patient_id']
                by_patient.setdefault(pid, []).append(panel)

            for pid, patient_panels in by_patient.items():
                # Skip already written patients
                if (out / f'{pid}_values.npy').exists():
                    continue

                patient_panels.sort(key=lambda p: p.get('charttime', ''))

                values_list = []
                masks_list  = []
                times_list  = []

                for panel in patient_panels:
                    v, m, charttime = panel_to_vectors(panel)

                    if charttime is None or int(m.sum()) == 0:
                        skipped += 1
                        continue

                    # Update Welford stats
                    for idx in np.where(m == 1.0)[0]:
                        count[idx] += 1
                        delta       = v[idx] - mean[idx]
                        mean[idx]  += delta / count[idx]
                        M2[idx]    += delta * (v[idx] - mean[idx])

                    values_list.append(v)
                    masks_list.append(m)
                    times_list.append(str(charttime))

                if values_list:
                    np.save(out / f'{pid}_values.npy', np.stack(values_list).astype(np.float32))
                    np.save(out / f'{pid}_masks.npy',  np.stack(masks_list).astype(np.float32))
                    with open(out / f'{pid}_times.json', 'w') as f:
                        json.dump(times_list, f)

            # Save checkpoint after every batch
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'count':          count,
                    'mean':           mean,
                    'M2':             M2,
                    'next_batch_idx': i + batch_size,
                    'skipped':        skipped
                }, f)

        print(f'Pass 1 done. Skipped {skipped} panels.')
    else:
        print('Pass 1 already complete, skipping.')

    # Finalize scaler 
    if Path(save_path).exists():
        print(f'Scaler already exists at {save_path}, loading.')
        with open(save_path, 'rb') as f:
            scaler = pickle.load(f)
    else:
        variance = np.where(count > 1, M2 / (count - 1), 1.0)
        stds     = np.where(np.sqrt(variance) > 0, np.sqrt(variance), 1.0)
        scaler   = {
            'mean':  mean.astype(np.float32),
            'std':   stds.astype(np.float32),
            'count': count.astype(np.int64)
        }
        with open(save_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f'Scaler saved → {save_path}')
        print(f'Tests with < 100 observations: {int((count < 100).sum())}')

    # Pass 2: normalize in place
    value_files  = list(out.glob('*_values.npy'))
    needs_norm   = [f for f in value_files
                    if not (out / f.name.replace('_values.npy', '_done')).exists()]

    print(f'Pass 2: {len(needs_norm)} files to normalize, '
          f'{len(value_files) - len(needs_norm)} already done.')

    for fpath in tqdm(needs_norm, desc='Pass 2 — normalizing'):
        pid_stem = fpath.name.replace('_values.npy', '')

        values = np.load(fpath)                              # (T, 170)
        masks  = np.load(out / f'{pid_stem}_masks.npy')     # (T, 170)

        norm_values = (values - scaler['mean']) / scaler['std']
        norm_values = norm_values * masks                    # zero out missing

        np.save(fpath, norm_values.astype(np.float32))      # overwrite in place

        # Sentinel file to mark as done
        (out / f'{pid_stem}_done').touch()

    # Clean up checkpoints
    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
    # Clean up sentinel files
    for f in out.glob('*_done'):
        f.unlink()

    print('All done.')
    return scaler
