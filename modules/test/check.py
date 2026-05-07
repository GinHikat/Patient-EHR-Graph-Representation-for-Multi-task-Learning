import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
if project_root not in sys.path:
    sys.path.append(project_root)

if current_dir not in sys.path:
    sys.path.append(current_dir)

base_data_dir = os.path.join(project_root, 'data')
base_data_path = os.path.join(base_data_dir, 'Timeline')

TIMELINE_DIR = os.path.join(base_data_dir, 'Timelines')  # update to your path

train_df = pd.read_csv(os.path.join(base_data_path, 'models', 'train_df.csv'))
val_df   = pd.read_csv(os.path.join(base_data_path, 'models', 'val_df.csv'))
test_df  = pd.read_csv(os.path.join(base_data_path, 'models', 'test_df.csv'))

# ── Collect all adm_ids that appear in timelines ──────────────────────────
print('Scanning timeline meta files...')
timeline_adm_ids = set()
timeline_dir = Path(TIMELINE_DIR)

meta_files = list(timeline_dir.glob('*_meta.json'))
print(f'Found {len(meta_files):,} timeline meta files')

for meta_path in tqdm(meta_files, desc='Reading meta'):
    with open(meta_path) as f:
        meta = json.load(f)
    for entry in meta:
        if entry['type'] == 'DISCHARGE' and entry.get('adm_id'):
            timeline_adm_ids.add(str(entry['adm_id']))

print(f'Total unique adm_ids in timelines: {len(timeline_adm_ids):,}')

# ── Check each split ───────────────────────────────────────────────────────
for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    df_adm_ids = set(df['id'].astype(str))
    matched    = df_adm_ids & timeline_adm_ids
    missing    = df_adm_ids - timeline_adm_ids

    print(f'\n{name}:')
    print(f'  Total admissions in df : {len(df_adm_ids):,}')
    print(f'  Found in timelines     : {len(matched):,}  ({100*len(matched)/len(df_adm_ids):.1f}%)')
    print(f'  Missing from timelines : {len(missing):,}  ({100*len(missing)/len(df_adm_ids):.1f}%)')
    if missing and len(missing) <= 10:
        print(f'  Missing adm_ids: {list(missing)}')