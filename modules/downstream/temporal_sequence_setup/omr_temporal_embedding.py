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

OMR_OUTPUT_DIR      = os.path.join(data_dir, 'OMR_Embedding')
OMR_SCALER_PATH     = 'train_omr_scaler.pkl'       # now train-only scaler
OMR_CHECKPOINT      = 'omr_fit_checkpoint.pkl'
OMR_RAW_CHECKPOINT  = 'omr_raw_checkpoint.pkl'

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

# Fit scaler on TRAIN patients only + write their raw files
def fit_omr_scaler(patient_ids, output_dir=OMR_OUTPUT_DIR, batch_size=500,
                   save_path=OMR_SCALER_PATH, checkpoint_path=OMR_CHECKPOINT):
    """
    Fit Welford scaler on train patients only and write their raw
    (unnormalized) .npz files.
    Does NOT normalize — normalization happens in normalize_omr_all().
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
        print('Starting fresh — fitting on train patients only')

    if start_idx < len(patient_ids):
        for i in tqdm(range(start_idx, len(patient_ids), batch_size),
                      desc='Step 1 — OMR fit scaler (train only)',
                      initial=start_idx // batch_size,
                      total=len(patient_ids) // batch_size):
            batch  = patient_ids[i:i + batch_size]
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

                    # Welford update — TRAIN patients only
                    for idx in np.where(m == 1.0)[0]:
                        if np.isnan(v[idx]) or np.isinf(v[idx]):
                            continue
                        count[idx] += 1
                        delta       = v[idx] - mean[idx]
                        mean[idx]  += delta / count[idx]
                        M2[idx]    += delta * (v[idx] - mean[idx])

                    values_list.append(v)
                    masks_list.append(m)
                    times_list.append(str(date))

                if values_list:
                    # Save RAW — normalize in step 3
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
                    'skipped':        skipped,
                }, f)
            os.replace(tmp, checkpoint_path)

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

        means_clean = np.nan_to_num(mean, nan=0.0).astype(np.float32)
        stds_clean  = np.nan_to_num(stds, nan=1.0, posinf=1.0).astype(np.float32)
        stds_clean  = np.clip(stds_clean, 1e-6, 1e6)

        scaler = {
            'mean':  means_clean,
            'std':   stds_clean,
            'count': count.astype(np.int64),
        }
        with open(save_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f'Train OMR scaler saved → {save_path}')
        print(f'Field counts: { {f: int(count[i]) for i, f in enumerate(OMR_FIELDS)} }')

    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()

    return scaler

# Write raw files for val/test patients (no scaler update)
def write_raw_omr_panels(patient_ids, output_dir=OMR_OUTPUT_DIR, batch_size=500,
                         checkpoint_path=OMR_RAW_CHECKPOINT):
    """
    Write raw (unnormalized) .npz files for val/test patients.
    Does NOT update scaler stats — scaler is train-only.
    """
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    if Path(checkpoint_path).exists():
        with open(checkpoint_path, 'rb') as f:
            ckpt = pickle.load(f)
        start_idx = ckpt['next_batch_idx']
        skipped   = ckpt['skipped']
        print(f'Resuming raw OMR panels from batch {start_idx // batch_size}')
    else:
        start_idx = 0
        skipped   = 0

    if start_idx < len(patient_ids):
        for i in tqdm(range(start_idx, len(patient_ids), batch_size),
                      desc='Step 2 — writing raw OMR panels (val/test)',
                      initial=start_idx // batch_size,
                      total=len(patient_ids) // batch_size):
            batch  = patient_ids[i:i + batch_size]
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
                    # No Welford update here
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
                pickle.dump({'next_batch_idx': i + batch_size, 'skipped': skipped}, f)
            os.replace(tmp, checkpoint_path)

        print(f'Step 2 done. Skipped {skipped} panels.')
    else:
        print('Step 2 already complete, skipping.')

    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()

# Normalize ALL patients using train scaler
def normalize_omr_all(scaler, patient_ids, output_dir=OMR_OUTPUT_DIR):
    """
    Normalize ALL patients (train + val + test) using the train-only scaler.
    Overwrites .npz files in place.
    """
    out = Path(output_dir)

    npz_files  = [out / f'{pid}.npz' for pid in patient_ids
                  if (out / f'{pid}.npz').exists()]
    needs_norm = [f for f in npz_files
                  if not (out / f.name.replace('.npz', '_done')).exists()]

    print(f'Step 3: {len(needs_norm)} files to normalize, '
          f'{len(npz_files) - len(needs_norm)} already done.')

    for fpath in tqdm(needs_norm, desc='Step 3 — normalizing OMR with train scaler'):
        data   = np.load(fpath)
        values = data['values']
        masks  = data['masks']

        norm_values = (values - scaler['mean']) / scaler['std']
        norm_values = np.nan_to_num(norm_values, nan=0.0)  # safety net
        norm_values = norm_values * masks

        np.savez_compressed(fpath,
                            values=norm_values.astype(np.float32),
                            masks=masks)
        (out / fpath.name.replace('.npz', '_done')).touch()

    for f in out.glob('*_done'):
        f.unlink()

    print('Step 3 done — all OMR patients normalized with train scaler.')

if __name__ == '__main__':
    train_pids = np.loadtxt(
        os.path.join(downstream_data_path, 'models', 'split_train_pids.txt'), dtype=int).tolist()
    val_pids   = np.loadtxt(
        os.path.join(downstream_data_path, 'models', 'split_val_pids.txt'),   dtype=int).tolist()
    test_pids  = np.loadtxt(
        os.path.join(downstream_data_path, 'models', 'split_test_pids.txt'),  dtype=int).tolist()
    all_pids   = train_pids + val_pids + test_pids

    # fit scaler on train only + write train raw files
    scaler = fit_omr_scaler(train_pids, save_path=OMR_SCALER_PATH)

    # write raw files for val + test (no scaler update)
    write_raw_omr_panels(val_pids)
    write_raw_omr_panels(test_pids)

    # normalize everyone with train scaler
    normalize_omr_all(scaler, all_pids)