import pandas as pd
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

import sys, os
from dotenv import load_dotenv

load_dotenv()

data_dir = os.getenv('DATA_DIR')
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))

if project_root not in sys.path:
    sys.path.append(project_root)

load_dotenv()

from shared_functions.global_functions import *

data_dir = os.getenv('DATA_DIR')
base_data_dir = os.path.join(project_root, 'data')
downstream_data_path = os.path.join(base_data_dir, 'downstream')

N_DIAGNOSES = 200   # top-200 diagnosis 

pos_weight_mortality   = torch.tensor([63.93])   
pos_weight_los         = torch.tensor([4.39])    
pos_weight_readmission = torch.tensor([2.88])    
pos_weight_progression = torch.load('progression_pos_weights.npy')  # 200-dim

class EHRDataset(Dataset):
    """
    Args:
        admissions_df   : DataFrame with columns:
                          id, patient_id, inhospital_dead, los_log,
                          readmission_30d
        timeline_dir    : path to Timelines/ folder
        admission_nodes : dict {adm_id (str): {'diagnoses': [...], 'drugs': [...]}}
        diag_to_idx     : dict {diagnosis_name (str, lower): int 0-199}
        patient_cache   : dict {patient_id: Tensor (64,)}  — precomputed
        admission_cache : dict {adm_id (str): Tensor (64,)} — precomputed
    """

    def __init__(
        self,
        admissions_df,
        timeline_dir,
        admission_nodes,
        diag_to_idx,
        patient_cache,
        admission_cache,
    ):
        self.timeline_dir    = Path(timeline_dir)
        self.admission_nodes = admission_nodes
        self.diag_to_idx     = diag_to_idx
        self.patient_cache   = patient_cache
        self.admission_cache = admission_cache

        # One row per admission — drop rows with missing critical labels
        df = admissions_df.copy()
        df = df[df['inhospital_dead'].notna()]
        df = df[df['los_log'].notna()]
        df['id']         = df['id'].astype(str)
        df['patient_id'] = df['patient_id'].astype(str)
        self.df = df.reset_index(drop=True)

        # Cache meta per patient to avoid reloading for each admission
        self._meta_cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        adm_id = str(row['id'])
        pid    = str(row['patient_id'])

        # Load timeline (cache meta per patient)
        if pid not in self._meta_cache:
            meta_path = self.timeline_dir / f'{pid}_meta.json'
            with open(meta_path) as f:
                self._meta_cache[pid] = json.load(f)

        meta = self._meta_cache[pid]

        # Find DISCHARGE position for this admission
        discharge_pos = None
        for i, entry in enumerate(meta):
            if entry['type'] == 'DISCHARGE' and str(entry.get('adm_id')) == adm_id:
                discharge_pos = i
                break

        if discharge_pos is None:
            # Admission has no DISCHARGE token — skip by returning None
            # collate_fn filters these out
            return None

        # Load and slice timeline causally up to DISCHARGE
        emb_path = self.timeline_dir / f'{pid}_emb.npy'
        dt_path  = self.timeline_dir / f'{pid}_dt.npy'

        emb = np.load(emb_path)   # (T_full, 128)
        dt  = np.load(dt_path)    # (T_full,)

        # Slice up to and including DISCHARGE token — causal
        emb = emb[:discharge_pos + 1]   # (T_adm, 128)
        dt  = dt[:discharge_pos + 1]    # (T_adm,)

        # Static vectors from precomputed cache
        patient_vec   = self.patient_cache.get(pid)
        admission_vec = self.admission_cache.get(adm_id)

        if patient_vec is None or admission_vec is None:
            return None

        # Build progression multilabel vector
        progression = np.zeros(N_DIAGNOSES, dtype=np.float32)
        adm_data    = self.admission_nodes.get(adm_id, {})
        for diag in adm_data.get('diagnoses', []):
            i = self.diag_to_idx.get(diag.lower())
            if i is not None:
                progression[i] = 1.0

        # Labels
        mortality   = float(row['inhospital_dead'])
        los_log     = float(row['los_log'])
        readmission = float(row['readmission_30d']) if not np.isnan(row['readmission_30d']) else -1.0
        # -1.0 = missing readmission label (last admission or patient died)
        # masked out in loss computation

        return {
            'emb':          torch.tensor(emb,         dtype=torch.float32),  # (T, 128)
            'dt':           torch.tensor(dt,           dtype=torch.float32),  # (T,)
            'patient_vec':  patient_vec,                                       # (64,)
            'admission_vec': admission_vec,                                    # (64,)
            'mortality':    torch.tensor(mortality,    dtype=torch.float32),
            'los_log':      torch.tensor(los_log,      dtype=torch.float32),
            'readmission':  torch.tensor(readmission,  dtype=torch.float32),
            'progression':  torch.tensor(progression,  dtype=torch.float32),  # (200,)
            'adm_id':       adm_id,
            'pid':          pid,
        }

## Helper functions

def ehr_collate_fn(batch):
    """
    Pads (T, 128) timelines to the max T in the batch.
    Filters out None samples (missing DISCHARGE token or missing cache entry).
    """
    # Filter out None samples
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    # Find max timeline length in this batch
    max_T = max(b['emb'].shape[0] for b in batch)

    emb_padded  = []
    dt_padded   = []
    lengths     = []

    for b in batch:
        T = b['emb'].shape[0]
        pad = max_T - T

        # Pad with zeros at the end
        emb_padded.append(
            torch.nn.functional.pad(b['emb'], (0, 0, 0, pad))  # (max_T, 128)
        )
        dt_padded.append(
            torch.nn.functional.pad(b['dt'], (0, pad))          # (max_T,)
        )
        lengths.append(T)

    return {
        'emb':           torch.stack(emb_padded),                              # (B, max_T, 128)
        'dt':            torch.stack(dt_padded),                               # (B, max_T)
        'lengths':       torch.tensor(lengths, dtype=torch.long),              # (B,)
        'patient_vec':   torch.stack([b['patient_vec']   for b in batch]),     # (B, 64)
        'admission_vec': torch.stack([b['admission_vec'] for b in batch]),     # (B, 64)
        'mortality':     torch.stack([b['mortality']     for b in batch]),     # (B,)
        'los_log':       torch.stack([b['los_log']       for b in batch]),     # (B,)
        'readmission':   torch.stack([b['readmission']   for b in batch]),     # (B,)
        'progression':   torch.stack([b['progression']   for b in batch]),     # (B, 200)
        'adm_ids':       [b['adm_id'] for b in batch],
        'pids':          [b['pid']    for b in batch],
    }

if __name__ == '__main__':
    import pandas as pd
    from torch.utils.data import DataLoader

    # Minimal fake data to test shapes
    fake_df = pd.DataFrame({
        'id':             ['22595853', '22841357'],
        'patient_id':     ['10000032', '10000032'],
        'inhospital_dead': [0.0, 0.0],
        'los_log':        [np.log1p(2.5), np.log1p(1.8)],
        'readmission_30d': [1.0, 0.0],
    })

    fake_admission_nodes = {
        '22595853': {'diagnoses': ['hypertension', 'diabetes mellitus'], 'drugs': []},
        '22841357': {'diagnoses': ['atrial fibrillation'],               'drugs': []},
    }

    fake_diag_to_idx = {'hypertension': 0, 'diabetes mellitus': 1, 'atrial fibrillation': 2}

    fake_patient_cache   = {'10000032': torch.zeros(64)}
    fake_admission_cache = {'22595853': torch.zeros(64), '22841357': torch.zeros(64)}

    dataset = EHRDataset(
        admissions_df   = fake_df,
        timeline_dir    = os.path.join(data_dir, 'Timelines'),
        admission_nodes = fake_admission_nodes,
        diag_to_idx     = fake_diag_to_idx,
        patient_cache   = fake_patient_cache,
        admission_cache = fake_admission_cache,
    )
    sample = dataset[0]
    print('emb shape:',        sample['emb'].shape)
    print('dt shape:',         sample['dt'].shape)
    print('progression shape:',sample['progression'].shape)
    
    loader = DataLoader(dataset, batch_size=2, collate_fn=ehr_collate_fn)
    batch  = next(iter(loader))
    print('Batch emb shape:', batch['emb'].shape)   # (2, max_T, 128)

    print('EHRDataset defined OK ✓')
    print('Import and use:')
    print('  from ehr_dataset import EHRDataset, ehr_collate_fn')