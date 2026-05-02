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
    """
    Pass 1: fit scaler + write raw panels (patient-separated)
    Pass 2: normalize in place and drop admissions
    """
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
        for i in tqdm(batches, desc='Pass 1 — fit scaler', initial=start_idx // batch_size,
                      total=len(patient_ids) // batch_size):
            batch = patient_ids[i:i+batch_size]
            panels = get_panels_for_batch(batch)

            by_patient = {}
            for panel in panels:
                pid = panel['patient_id']
                by_patient.setdefault(pid, []).append(panel)

            for pid, patient_panels in by_patient.items():
                # Skip patients already written (in case of mid-batch crash)
                if (out / f'{pid}.json').exists():
                    continue

                patient_panels.sort(key=lambda p: p.get('charttime', ''))
                patient_data = {}

                for panel in patient_panels:
                    v, m, charttime = panel_to_vectors(panel)

                    if charttime is None or int(m.sum()) == 0:
                        skipped += 1
                        continue

                    for idx in np.where(m == 1.0)[0]:
                        count[idx] += 1
                        delta       = v[idx] - mean[idx]
                        mean[idx]  += delta / count[idx]
                        M2[idx]    += delta * (v[idx] - mean[idx])

                    patient_data[str(charttime)] = {
                        'values': v.tolist(),
                        'mask':   m.tolist()
                    }

                if patient_data:
                    with open(out / f'{pid}.json', 'w') as f:
                        json.dump(patient_data, f)

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

    # Pass 2: normalize in place (skip already normalized files)
    patient_files = list(out.glob('*.json'))
    needs_norm    = []

    for fpath in patient_files:
        with open(fpath) as f:
            data = json.load(f)
        # Check first entry — if values are already normalized they'll be near [-3, 3]
        # We use a sentinel key to track this cleanly instead
        first = next(iter(data.values()))
        if first.get('normalized', False) is False:
            needs_norm.append(fpath)

    print(f'Pass 2: {len(needs_norm)} files to normalize, '
          f'{len(patient_files) - len(needs_norm)} already done.')

    for fpath in tqdm(needs_norm, desc='Pass 2 — normalizing'):
        with open(fpath) as f:
            patient_data = json.load(f)

        for charttime, entry in patient_data.items():
            v = np.array(entry['values'], dtype=np.float32)
            m = np.array(entry['mask'],   dtype=np.float32)
            norm_v = (v - scaler['mean']) / scaler['std']
            norm_v = norm_v * m
            entry['values']     = norm_v.tolist()
            entry['normalized'] = True  # sentinel so we can skip on resume

        with open(fpath, 'w') as f:
            json.dump(patient_data, f)

    # Clean up checkpoint once fully done
    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        print('Checkpoint cleared.')

    print('All done.')
    return scaler
