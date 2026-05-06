import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import re
import json
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

import sys, os
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))

if project_root not in sys.path:
    sys.path.append(project_root)

load_dotenv()

from shared_functions.global_functions import *

data_dir = os.getenv('DATA_DIR')
base_data_dir = os.path.join(project_root, 'data')
downstream_data_path = os.path.join(base_data_dir, 'downstream')

# Paths for vocab and patients
VOCAB_PATH    = os.path.join(downstream_data_path, 'embedding', 'lab_vocab.json')
PATIENTS_PATH = os.path.join(downstream_data_path, 'patients.txt')

# Load Lab vocab for vector generation
with open(VOCAB_PATH, 'r') as f:
    LAB_VOCAB = json.load(f)

# Load all patient IDs for processing
with open(PATIENTS_PATH) as f:
    all_patient_ids = [int(line.strip()) for line in f.readlines()]

VOCAB_SIZE = len(LAB_VOCAB)  # 170
NON_VALUE_FIELDS = {'id', 'name', 'patient_id', 'admission_id', 'charttime'}
BATCH_SIZE = 500
mimic_path = os.path.join(data_dir, 'mimic_iv')
hosp = os.path.join(mimic_path, 'hosp')

base_data_dir = os.path.join(project_root, 'data')
downstream_data_path = os.path.join(base_data_dir, 'downstream')

OUTPUT_DIR = os.path.join(data_dir, 'Lab_Embedding')
os.makedirs(OUTPUT_DIR, exist_ok=True)
SCALER_PATH       = 'train_lab_scaler.pkl'       # now train-only scaler
CHECKPOINT_PATH   = 'fit_checkpoint.pkl'
RAW_CHECKPOINT    = 'raw_panels_checkpoint.pkl'


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

# Fit scaler on TRAIN patients only + write their raw files
def fit_scaler(patient_ids, output_dir=OUTPUT_DIR, batch_size=BATCH_SIZE,
               save_path=SCALER_PATH, checkpoint_path=CHECKPOINT_PATH):
    """
    Fit Welford scaler on train patients only and write their raw
    (unnormalized) value/mask/times files.
    Does NOT normalize anything — normalization happens in normalize_all().
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
        print('Starting fresh — fitting on train patients only')

    if start_idx < len(patient_ids):
        batches = range(start_idx, len(patient_ids), batch_size)
        for i in tqdm(batches, desc='Step 1 — fit scaler (train only)',
                      initial=start_idx // batch_size,
                      total=len(patient_ids) // batch_size):
            batch  = patient_ids[i:i + batch_size]
            panels = get_panels_for_batch(batch)

            by_patient = {}
            for panel in panels:
                pid = panel['patient_id']
                by_patient.setdefault(pid, []).append(panel)

            for pid, patient_panels in by_patient.items():
                if (out / f'{pid}_values.npy').exists():
                    continue

                patient_panels.sort(key=lambda p: p.get('charttime', ''))
                values_list, masks_list, times_list = [], [], []

                for panel in patient_panels:
                    v, m, charttime = panel_to_vectors(panel)

                    if charttime is None or int(m.sum()) == 0:
                        skipped += 1
                        continue

                    # Welford update — TRAIN patients only
                    for idx in np.where(m == 1.0)[0]:
                        count[idx] += 1
                        delta       = v[idx] - mean[idx]
                        mean[idx]  += delta / count[idx]
                        M2[idx]    += delta * (v[idx] - mean[idx])

                    values_list.append(v)
                    masks_list.append(m)
                    times_list.append(str(charttime))

                if values_list:
                    # Save RAW — normalize in step 3
                    np.save(out / f'{pid}_values.npy',
                            np.stack(values_list).astype(np.float32))
                    np.save(out / f'{pid}_masks.npy',
                            np.stack(masks_list).astype(np.float32))
                    with open(out / f'{pid}_times.json', 'w') as f:
                        json.dump(times_list, f)

            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'count':          count,
                    'mean':           mean,
                    'M2':             M2,
                    'next_batch_idx': i + batch_size,
                    'skipped':        skipped,
                }, f)

        print(f'Step 1 done. Skipped {skipped} panels.')
    else:
        print('Step 1 already complete, skipping.')

    # Finalize and save scaler
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
            'count': count.astype(np.int64),
        }
        with open(save_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f'Train scaler saved → {save_path}')
        print(f'Tests with < 100 observations: {int((count < 100).sum())}')

    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()

    return scaler

# Write raw files for val/test patients (no scaler update)
def write_raw_panels(patient_ids, output_dir=OUTPUT_DIR, batch_size=BATCH_SIZE,
                     checkpoint_path=RAW_CHECKPOINT):
    """
    Write raw (unnormalized) value/mask/times files for val/test patients.
    Does NOT update scaler stats — scaler is train-only.
    """
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    # Resume checkpoint
    if Path(checkpoint_path).exists():
        with open(checkpoint_path, 'rb') as f:
            ckpt     = pickle.load(f)
        start_idx = ckpt['next_batch_idx']
        skipped   = ckpt['skipped']
        print(f'Resuming raw panels from batch {start_idx // batch_size}')
    else:
        start_idx = 0
        skipped   = 0

    if start_idx < len(patient_ids):
        batches = range(start_idx, len(patient_ids), batch_size)
        for i in tqdm(batches, desc='Step 2 — writing raw panels (val/test)',
                      initial=start_idx // batch_size,
                      total=len(patient_ids) // batch_size):
            batch  = patient_ids[i:i + batch_size]
            panels = get_panels_for_batch(batch)

            by_patient = {}
            for panel in panels:
                pid = panel['patient_id']
                by_patient.setdefault(pid, []).append(panel)

            for pid, patient_panels in by_patient.items():
                if (out / f'{pid}_values.npy').exists():
                    continue

                patient_panels.sort(key=lambda p: p.get('charttime', ''))
                values_list, masks_list, times_list = [], [], []

                for panel in patient_panels:
                    v, m, charttime = panel_to_vectors(panel)
                    if charttime is None or int(m.sum()) == 0:
                        skipped += 1
                        continue
                    # No Welford update here
                    values_list.append(v)
                    masks_list.append(m)
                    times_list.append(str(charttime))

                if values_list:
                    np.save(out / f'{pid}_values.npy',
                            np.stack(values_list).astype(np.float32))
                    np.save(out / f'{pid}_masks.npy',
                            np.stack(masks_list).astype(np.float32))
                    with open(out / f'{pid}_times.json', 'w') as f:
                        json.dump(times_list, f)

            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'next_batch_idx': i + batch_size, 'skipped': skipped}, f)

        print(f'Step 2 done. Skipped {skipped} panels.')
    else:
        print('Step 2 already complete, skipping.')

    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()

# Normalize ALL patients using train scaler
def normalize_all(scaler, patient_ids, output_dir=OUTPUT_DIR):
    """
    Normalize ALL patients (train + val + test) using the train-only scaler.
    Overwrites *_values.npy in place.
    """
    out = Path(output_dir)

    value_files = [out / f'{pid}_values.npy' for pid in patient_ids
                   if (out / f'{pid}_values.npy').exists()]
    needs_norm  = [f for f in value_files
                   if not (out / f.name.replace('_values.npy', '_done')).exists()]

    print(f'Step 3: {len(needs_norm)} files to normalize, '
          f'{len(value_files) - len(needs_norm)} already done.')

    for fpath in tqdm(needs_norm, desc='Step 3 — normalizing with train scaler'):
        pid_stem = fpath.name.replace('_values.npy', '')

        values = np.load(fpath)                             # (T, 170) raw
        masks  = np.load(out / f'{pid_stem}_masks.npy')    # (T, 170)

        norm_values = (values - scaler['mean']) / scaler['std']
        norm_values = norm_values * masks                   # zero out missing

        np.save(fpath, norm_values.astype(np.float32))

        (out / f'{pid_stem}_done').touch()

    for f in out.glob('*_done'):
        f.unlink()

    print('Step 3 done — all patients normalized with train scaler.')

if __name__ == '__main__':

    train_pids = np.loadtxt(os.path.join(downstream_data_path, 'models', 'split_train_pids.txt'), dtype=int).tolist()
    val_pids   = np.loadtxt(os.path.join(downstream_data_path, 'models','split_val_pids.txt'),   dtype=int).tolist()
    test_pids  = np.loadtxt(os.path.join(downstream_data_path, 'models','split_test_pids.txt'),  dtype=int).tolist()
    all_pids   = train_pids + val_pids + test_pids

    # fit scaler on train only + write train raw files
    scaler = fit_scaler(train_pids, save_path=SCALER_PATH)

    # write raw files for val + test (no scaler update)
    write_raw_panels(val_pids)
    write_raw_panels(test_pids)

    # normalize everyone with train scaler
    normalize_all(scaler, all_pids)