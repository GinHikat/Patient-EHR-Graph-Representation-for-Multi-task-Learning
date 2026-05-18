import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import pickle
import json
from datetime import datetime

import sys, os
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
THESIS_PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../../../"))
if THESIS_PROJECT_ROOT not in sys.path:
    sys.path.append(THESIS_PROJECT_ROOT)

# Add modules/downstream to sys.path to allow importing 'presetup'
downstream_root = os.path.abspath(os.path.join(current_dir, "../"))

if downstream_root not in sys.path:
    sys.path.append(downstream_root)

if current_dir not in sys.path:
    sys.path.append(current_dir)

from presetup.GAT import KG_GAT
from presetup.unified_encoder import *
from shared_functions.global_functions import query_neo4j

def parse_time(t) -> datetime:
    """Normalize all timestamp formats to datetime."""
    if t is None:
        return None
    if isinstance(t, str):
        t = t.replace('T', ' ').split('.')[0].strip()
        return datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
    if hasattr(t, 'year') and hasattr(t, 'month') and hasattr(t, 'day'):
        return datetime(t.year, t.month, t.day, 
                        getattr(t, 'hour', 0), 
                        getattr(t, 'minute', 0), 
                        getattr(t, 'second', 0))
    return t

load_dotenv() 

data_dir = os.getenv('DATA_DIR')
base_data_dir = os.path.join(THESIS_PROJECT_ROOT, 'data')
downstream_data_path = os.path.join(base_data_dir, 'downstream')


with open(os.path.join(downstream_data_path, 'admission_nodes.json'), 'r') as f:
    admission_nodes = json.load(f)

with open(os.path.join(downstream_data_path, 'patients.txt')) as f:
    all_patient_ids = [int(line.strip()) for line in f.readlines()]

kg_embeddings = np.load(os.path.join(downstream_data_path,'kg_nodes_embed_gat.npy'))
all_nodes    = pd.read_csv(os.path.join(downstream_data_path,'kg_nodes.csv'), dtype={'id': str}, low_memory=False)
name_to_idx  = dict(zip(all_nodes['name'].str.lower(), all_nodes['node_idx']))

# Outpatient note lookup cache (initially empty, populated dynamically or via batch pre-fetching)
outpatient_lookup = defaultdict(list)

# Load CCSR diagnosis map for category-to-ICD resolution
map_path = os.path.join(base_data_dir, 'diagnosis_map.csv')
cat_to_icds = {}
if os.path.exists(map_path):
    print("Loading CCSR Diagnosis Mapping table...")
    map_df = pd.read_csv(map_path, low_memory=False)
    map_df['ccsr_category_str'] = map_df['ccsr_category'].astype(str).str.strip().str.lower()
    map_df['icd_code_str'] = map_df['icd_code'].astype(str).str.strip().str.lower()
    cat_to_icds = map_df.groupby('ccsr_category_str')['icd_code_str'].apply(list).to_dict()

# Map specific ICD code IDs to GAT node indices
all_nodes['id_str'] = all_nodes['id'].astype(str).str.strip().str.lower()
id_to_idx = dict(zip(all_nodes['id_str'], all_nodes['node_idx']))

# Direct Neo4j [:HAS_DIAGNOSIS] edges are used instead of radiology.csv loading.

OUTPUT_DIR = os.path.join(data_dir, 'Lab_Embedding')
OMR_OUTPUT_DIR  = os.path.join(data_dir, 'OMR_Embedding')
TIMELINE_DIR = os.path.join(data_dir, 'Timelines')
MAX_DIAGNOSES = 30
MAX_DRUGS     = 40
MAX_NODES     = MAX_DIAGNOSES + MAX_DRUGS  
SAMPLE_PID = 10000032

### Load saved artifacts
def load_lab(pid, output_dir=OUTPUT_DIR):
    '''
    Load lab values and masks for a patient.

    Args:
        pid (str): Patient ID.
        output_dir (str): Directory containing the lab data.

    Returns:
        tuple: (values, masks, times)
            values (np.ndarray): Array of shape (T, 170) containing normalized lab values.
            masks (np.ndarray): Array of shape (T, 170) containing masks for the lab values.
            times (list): List of T chart times.
    '''
    out = Path(output_dir)
    values = np.load(out / f'{pid}_values.npy')   # (T, 170) normalized
    masks  = np.load(out / f'{pid}_masks.npy')    # (T, 170)
    with open(out / f'{pid}_times.json') as f:
        times = json.load(f)                       # list of T charttimes
    return values, masks, times

def load_omr(pid, output_dir=OMR_OUTPUT_DIR):
    '''
    Load OMR values and masks for a patient.

    Args:
        pid (str): Patient ID.
        output_dir (str): Directory containing the OMR data.

    Returns:
        tuple: (values, masks, times)
            values (np.ndarray): Array of shape (T, 170) containing normalized OMR values.
            masks (np.ndarray): Array of shape (T, 170) containing masks for the OMR values.
            times (list): List of T chart times.
    '''
    out  = Path(output_dir)
    data = np.load(out / f'{pid}.npz')
    with open(out / f'{pid}_times.json') as f:
        times = json.load(f)
    return data['values'], data['masks'], times

def load_patient_timeline(pid, output_dir=TIMELINE_DIR):
    out = Path(output_dir)
    emb  = np.load(out / f'{pid}_emb.npy')    # (T, 128)
    dt   = np.load(out / f'{pid}_dt.npy')     # (T,)
    with open(out / f'{pid}_meta.json') as f:
        meta = json.load(f)                    # list of T dicts
    return emb, dt, meta

### Save artifacts
def collate_admission_nodes(admission_ids: list[str], admission_nodes: dict,
                             kg_embeddings: np.ndarray, name_to_idx: dict) -> tuple:
    """
    Build padded node embedding tensor + mask for a batch of admissions.

    Returns:
        padded : (B, MAX_NODES, 128)
        mask   : (B, MAX_NODES) bool
    """
    B      = len(admission_ids)
    padded = torch.zeros(B, MAX_NODES, 128)
    mask   = torch.zeros(B, MAX_NODES, dtype=torch.bool)

    for i, adm_id in enumerate(admission_ids):
        nodes  = admission_nodes.get(str(adm_id), {'diagnoses': [], 'drugs': []})

        # Cap at max, combine diagnoses first then drugs
        dx    = nodes['diagnoses'][:MAX_DIAGNOSES]
        drugs = nodes['drugs'][:MAX_DRUGS]
        all_nodes_list = dx + drugs

        for j, name in enumerate(all_nodes_list):
            idx = name_to_idx.get(name.lower())
            if idx is not None:
                padded[i, j] = torch.tensor(kg_embeddings[idx])
                mask[i, j]   = True

    return padded, mask  # (B, 70, 128), (B, 70)

MAX_OUTPATIENT_DIAGNOSES = 50

def collate_outpatient_nodes(categories_list: list[str], cat_to_icds: dict,
                             id_to_idx: dict, kg_embeddings: np.ndarray) -> tuple:
    """
    Build padded node embedding tensor + mask for a list of CCSR categories.

    Returns:
        padded : (1, MAX_OUTPATIENT_DIAGNOSES, 128)
        mask   : (1, MAX_OUTPATIENT_DIAGNOSES) bool
    """
    padded = torch.zeros(1, MAX_OUTPATIENT_DIAGNOSES, 128)
    mask   = torch.zeros(1, MAX_OUTPATIENT_DIAGNOSES, dtype=torch.bool)

    # Flatten all categories to their corresponding specific ICD codes
    flat_icds = []
    for cat in categories_list:
        icds = cat_to_icds.get(cat.lower(), [])
        flat_icds.extend(icds)
    
    # Deduplicate to prevent redundant nodes
    flat_icds = list(set(flat_icds))
    
    # Cap at MAX_OUTPATIENT_DIAGNOSES
    flat_icds = flat_icds[:MAX_OUTPATIENT_DIAGNOSES]

    for j, code in enumerate(flat_icds):
        idx = id_to_idx.get(code.lower())
        if idx is not None:
            padded[0, j] = torch.tensor(kg_embeddings[idx])
            mask[0, j]   = True

    return padded, mask


ICU_UNIT_VOCAB = {
    'medical intensive care unit (micu)': 1,
    'surgical intensive care unit (sicu)': 2,
    'medical/surgical intensive care unit (micu/sicu)': 3,
    'cardiac vascular intensive care unit (cvicu)': 4,
    'coronary care unit (ccu)': 5,
    'neuro surgical intensive care unit (neuro sicu)': 6,
    'trauma sicu (tsicu)': 7,
    'intensive care unit (icu)': 8,
    'special care nursery (scn)': 9,
}

CARE_UNIT_VOCAB = {
    'medicine': 1,
    'surgery': 2,
    'emergency department': 3,
    'obstetrics/gynecology': 4,
    'psychiatry': 5,
    'pediatrics': 6,
    'neurology': 7,
    'orthopedics': 8,
    'observation': 9,
    'post anesthesia care unit (pacu)': 10,
    'post-anesthesia care unit (pacu)': 10,
    'cardiac surgery': 11,
    'labor & delivery': 12,
    'unknown': 13,
    'medical intensive care unit (micu)': 14,
    'surgical intensive care unit (sicu)': 15,
    'medical/surgical intensive care unit (micu/sicu)': 16,
    'coronary care unit (ccu)': 17,
    'cardiac vascular intensive care unit (cvicu)': 18,
    'trauma intensive care unit (tsicu)': 19,
    'trauma sicu (tsicu)': 19,
    'neuroscience intensive care unit (nicu)': 20,
    'neuro surgical intensive care unit (neuro sicu)': 20,
    'neonatal intensive care unit (nicu)': 21,
    'pediatric intensive care unit (picu)': 22,
    'neuro/surgical intensive care unit (nsicu)': 23,
    'medical/surgical': 24,
    'medical/surgical/gynecology': 25,
    'medical/surgical/trauma': 26,
    'special care nursery (scn)': 27,
    'intensive care unit (icu)': 28,
}

TRANSFER_TYPE_VOCAB = {
    'admission': 1,
    'transfer': 2,
    'discharge': 3,
    'emergency department': 4,
    'admit': 1,
    'ed': 4
}

def build_patient_timeline(
    pid,
    admission_nodes,
    kg_embeddings,
    name_to_idx,
    lab_encoder,
    omr_encoder,
    special_encoder,
    admission_encoder,
    icu_encoder,
    transfer_encoder,
    outpatient_encoder,
    icu_lookup=None,
    transfer_lookup=None,
    admission_lookup=None,
    outpatient_lookup=None,
    cat_to_icds=None,
    id_to_idx=None,
    device=torch.device('cpu')):
    """
    Build unified timeline for one patient, enriched with ICU stays, transfers, and outpatient notes.

    Returns:
        embeddings  : (T, 128) tensor — one vector per event
        times       : list of T datetime objects
        event_meta  : list of T dicts — {type, adm_id/note_id (if applicable), ...}
    """
    events = []   # list of {time, type, data}

    # Labs
    try:
        lab_vals, lab_masks, lab_times = load_lab(pid)
        for t, v, m in zip(lab_times, lab_vals, lab_masks):
            pt = parse_time(t)
            if pt:
                events.append({'time': pt, 'type': 'lab',
                                'values': v, 'masks': m})
    except FileNotFoundError:
        pass

    # OMR
    try:
        omr_vals, omr_masks, omr_times = load_omr(pid)
        for t, v, m in zip(omr_times, omr_vals, omr_masks):
            pt = parse_time(t)
            if pt:
                events.append({'time': pt, 'type': 'omr',
                                'values': v, 'masks': m})
    except FileNotFoundError:
        pass

    pid_str = str(pid).strip().lower()

    # ICU Stays (Start events only)
    try:
        if icu_lookup is not None:
            icu_res = icu_lookup.get(pid_str, [])
        else:
            icu_res = query_neo4j('''
                MATCH (a:Admission)-[:HAS_ICUSTAY]->(i:ICU)
                WHERE a.patient_id = $pid
                RETURN i.unit AS unit, i.start_time AS intime
                ORDER BY i.start_time
            ''', pid=pid)
        for row in icu_res:
            t = parse_time(row['intime'])
            unit = row['unit']
            if t and unit:
                events.append({
                    'time': t,
                    'type': 'ICU',
                    'unit': unit
                })
    except Exception:
        pass

    # Transfers (Start events only)
    try:
        if transfer_lookup is not None:
            trans_res = transfer_lookup.get(pid_str, [])
        else:
            trans_res = query_neo4j('''
                MATCH (a:Admission)-[:HAS_TRANSFER]->(t:Transfer)
                WHERE a.patient_id = $pid
                RETURN t.care_unit AS care_unit, t.type AS type, t.start_time AS intime
                ORDER BY t.start_time
            ''', pid=pid)
        for row in trans_res:
            t = parse_time(row['intime'])
            care_unit = row['care_unit']
            transfer_type = row.get('type') or row.get('transfer_type')
            if t and care_unit:
                if str(care_unit).strip().lower() == 'unknown':
                    continue
                events.append({
                    'time': t,
                    'type': 'Transfer',
                    'care_unit': care_unit,
                    'transfer_type': transfer_type or 'transfer'
                })
    except Exception:
        pass

    # Admissions → ADMIT + DISCHARGE + AdmissionEmb
    try:
        if admission_lookup is not None:
            adm_result = admission_lookup.get(pid_str, [])
        else:
            adm_result = query_neo4j('''
                MATCH (a:Admission)
                WHERE a.patient_id = $pid
                RETURN a.id AS adm_id, a.admit_time AS admittime, a.discharge_time AS dischtime
                ORDER BY a.admit_time
            ''', pid=pid)

        for row in adm_result:
            adm_id   = str(row['adm_id'])
            admit_t  = parse_time(row['admittime'])
            disch_t  = parse_time(row['dischtime'])

            if admit_t:
                events.append({'time': admit_t,  'type': 'ADMIT',     'adm_id': adm_id})
            if disch_t:
                events.append({'time': disch_t,  'type': 'admission_emb', 'adm_id': adm_id})
                events.append({'time': disch_t,  'type': 'DISCHARGE', 'adm_id': adm_id})
    except Exception:
        pass

    # Outpatient Notes (OutNote events)
    try:
        if outpatient_lookup is not None and len(outpatient_lookup) > 0:
            out_res = outpatient_lookup.get(pid_str, [])
        else:
            out_res = []
            raw_res = query_neo4j('''
                MATCH (p:Patient)-[:HAS_OUTPATIENT_NOTE]->(d:Outpatient)
                WHERE p.id = $pid
                OPTIONAL MATCH (d)-[:HAS_DIAGNOSIS]->(diag:Diagnosis)
                RETURN d.id AS note_id, d.time AS charttime, collect(diag.id) AS categories
                ORDER BY d.time
            ''', pid=pid)
            for row in raw_res:
                note_id_str = str(row['note_id']).strip().lower()
                categories = [str(cat).strip().lower() for cat in row.get('categories', []) if cat]
                out_res.append({
                    'note_id': note_id_str,
                    'time': parse_time(row['charttime']),
                    'categories': categories
                })
        for event in out_res:
            t = event['time']
            # Only include OutNote events that contain at least one diagnosis category
            if t and event['categories']:
                events.append({
                    'time': t,
                    'type': 'OutNote',
                    'categories': event['categories'],
                    'note_id': event['note_id']
                })
    except Exception:
        pass

    # Sort by time. For events at the exact same timestamp, ensure admission_emb is placed before DISCHARGE
    def sort_key(e):
        t_type = e['type']
        if t_type == 'admission_emb':
            priority = 0
        elif t_type == 'DISCHARGE':
            priority = 1
        else:
            priority = -1
        return (e['time'], priority)

    events.sort(key=sort_key)

    if not events:
        return None, None, None

    # Encode each event → 128-dim
    embeddings = []
    times      = []
    meta       = []

    for event in events:
        t = event['type']

        if t == 'lab':
            v   = torch.tensor(event['values'], dtype=torch.float32).unsqueeze(0).to(device)
            m   = torch.tensor(event['masks'],  dtype=torch.float32).unsqueeze(0).to(device)
            emb = lab_encoder(v, m)                          # (1, 128)

        elif t == 'omr':
            v   = torch.tensor(event['values'], dtype=torch.float32).unsqueeze(0).to(device)
            m   = torch.tensor(event['masks'],  dtype=torch.float32).unsqueeze(0).to(device)
            emb = omr_encoder(v, m)                          # (1, 128)

        elif t == 'ADMIT':
            tok = torch.tensor([SPECIAL_TOKENS['ADMIT']]).to(device)
            emb = special_encoder(tok)                       # (1, 128)

        elif t == 'DISCHARGE':
            tok = torch.tensor([SPECIAL_TOKENS['DISCHARGE']]).to(device)
            emb = special_encoder(tok)                       # (1, 128)

        elif t == 'admission_emb':
            padded, mask = collate_admission_nodes(
                [event['adm_id']], admission_nodes, kg_embeddings, name_to_idx
            )
            padded = padded.to(device)
            mask   = mask.to(device)
            emb    = admission_encoder(padded, mask)          # (1, 128)

        elif t == 'OutNote':
            padded, mask = collate_outpatient_nodes(
                event['categories'], cat_to_icds, id_to_idx, kg_embeddings
            )
            padded = padded.to(device)
            mask   = mask.to(device)
            emb    = outpatient_encoder(padded, mask)         # (1, 128)

        elif t == 'ICU':
            unit_name = str(event['unit']).lower()
            unit_id = ICU_UNIT_VOCAB.get(unit_name, 0)
            tok = torch.tensor([unit_id], dtype=torch.long).to(device)
            emb = icu_encoder(tok)                           # (1, 128)

        elif t == 'Transfer':
            cu_name = str(event['care_unit']).lower()
            cu_id = CARE_UNIT_VOCAB.get(cu_name, 0)
            type_name = str(event['transfer_type']).lower()
            type_id = TRANSFER_TYPE_VOCAB.get(type_name, 0)
            
            cu_tok = torch.tensor([cu_id], dtype=torch.long).to(device)
            type_tok = torch.tensor([type_id], dtype=torch.long).to(device)
            emb = transfer_encoder(cu_tok, type_tok)          # (1, 128)

        embeddings.append(emb.squeeze(0))                    # (128,)
        times.append(event['time'])
        meta.append({
            'type': t, 
            'adm_id': event.get('adm_id'),
            'unit': event.get('unit'),
            'care_unit': event.get('care_unit'),
            'transfer_type': event.get('transfer_type'),
            'note_id': event.get('note_id')
        })

    embeddings = torch.stack(embeddings)                     # (T, 128)
    return embeddings, times, meta

def build_and_save_all_timelines(
    patient_ids,
    admission_nodes,
    kg_embeddings,
    name_to_idx,
    lab_encoder,
    omr_encoder,
    special_encoder,
    admission_encoder,
    icu_encoder,
    transfer_encoder,
    outpatient_encoder,
    output_dir=TIMELINE_DIR,
    batch_size=100,
    device=torch.device('cpu')):

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    skipped  = 0
    CKPT     = 'timeline_checkpoint.pkl'

    # Resume
    if Path(CKPT).exists():
        with open(CKPT, 'rb') as f:
            ckpt = pickle.load(f)
        start_idx = ckpt['next_idx']
        skipped   = ckpt['skipped']
        print(f'Resuming from patient {start_idx}/{len(patient_ids)}')
    else:
        start_idx = 0
        print('Starting fresh')

    lab_encoder.eval()
    omr_encoder.eval()
    special_encoder.eval()
    admission_encoder.eval()
    icu_encoder.eval()
    transfer_encoder.eval()
    outpatient_encoder.eval()

    # Process in chunks of 1000 patients for extreme bulk query speeds
    db_batch_size = 1000
    for chunk_start in range(start_idx, len(patient_ids), db_batch_size):
        chunk_end = min(chunk_start + db_batch_size, len(patient_ids))
        batch_pids = patient_ids[chunk_start:chunk_end]

        # Filter down to only those that do NOT exist on disk to save database/CPU cycles
        pids_to_process = [pid for pid in batch_pids if not (out / f'{pid}_emb.npy').exists()]

        # Generate local batch-lookup tables
        icu_lookup = defaultdict(list)
        transfer_lookup = defaultdict(list)
        admission_lookup = defaultdict(list)
        outpatient_lookup = defaultdict(list)

        if pids_to_process:
            # 1. Bulk pre-fetch ICU Stays
            try:
                icu_batch = query_neo4j('''
                    MATCH (a:Admission)-[:HAS_ICUSTAY]->(i:ICU)
                    WHERE a.patient_id IN $pids
                    RETURN a.patient_id AS pid, i.unit AS unit, i.start_time AS intime
                    ORDER BY i.start_time
                ''', pids=pids_to_process)
                for row in icu_batch:
                    key = str(row['pid']).strip().lower()
                    icu_lookup[key].append({
                        'unit': row['unit'],
                        'intime': row['intime']
                    })
            except Exception:
                pass

            # 2. Bulk pre-fetch Transfers
            try:
                trans_batch = query_neo4j('''
                    MATCH (a:Admission)-[:HAS_TRANSFER]->(t:Transfer)
                    WHERE a.patient_id IN $pids
                    RETURN a.patient_id AS pid, t.care_unit AS care_unit, t.type AS type, t.start_time AS intime
                    ORDER BY t.start_time
                ''', pids=pids_to_process)
                for row in trans_batch:
                    key = str(row['pid']).strip().lower()
                    transfer_lookup[key].append({
                        'care_unit': row['care_unit'],
                        'type': row['type'],
                        'intime': row['intime']
                    })
            except Exception:
                pass

            # 3. Bulk pre-fetch Admissions
            try:
                adm_batch = query_neo4j('''
                    MATCH (a:Admission)
                    WHERE a.patient_id IN $pids
                    RETURN a.patient_id AS pid, a.id AS adm_id, a.admit_time AS admittime, a.discharge_time AS dischtime
                    ORDER BY a.admit_time
                ''', pids=pids_to_process)
                for row in adm_batch:
                    key = str(row['pid']).strip().lower()
                    admission_lookup[key].append({
                        'adm_id': row['adm_id'],
                        'admittime': row['admittime'],
                        'dischtime': row['dischtime']
                    })
            except Exception:
                pass

            # 4. Bulk pre-fetch Outpatient Notes (OutNotes) from Neo4j
            try:
                out_batch = query_neo4j('''
                    MATCH (p:Patient)-[:HAS_OUTPATIENT_NOTE]->(d:Outpatient)
                    WHERE p.id IN $pids
                    OPTIONAL MATCH (d)-[:HAS_DIAGNOSIS]->(diag:Diagnosis)
                    RETURN p.id AS pid, d.id AS note_id, d.time AS charttime, collect(diag.id) AS categories
                    ORDER BY d.time
                ''', pids=pids_to_process)
                for row in out_batch:
                    key = str(row['pid']).strip().lower()
                    note_id_str = str(row['note_id']).strip().lower()
                    categories = [str(cat).strip().lower() for cat in row.get('categories', []) if cat]
                    outpatient_lookup[key].append({
                        'note_id': note_id_str,
                        'time': parse_time(row['charttime']),
                        'categories': categories
                    })
            except Exception:
                pass

        # Process the batch patients individually using the pre-loaded caches
        for pid in tqdm(batch_pids, desc=f'Building timelines [{chunk_start}-{chunk_end}]'):
            # Skip already done
            if (out / f'{pid}_emb.npy').exists():
                continue

            with torch.no_grad():
                embeddings, times, meta = build_patient_timeline(
                    pid, admission_nodes, kg_embeddings, name_to_idx,
                    lab_encoder, omr_encoder, special_encoder, admission_encoder,
                    icu_encoder, transfer_encoder, outpatient_encoder,
                    icu_lookup=icu_lookup,
                    transfer_lookup=transfer_lookup,
                    admission_lookup=admission_lookup,
                    outpatient_lookup=outpatient_lookup,
                    cat_to_icds=cat_to_icds,
                    id_to_idx=id_to_idx,
                    device=device
                )

            if embeddings is None or len(times) == 0:
                skipped += 1
                continue

            delta_t = compute_delta_t(times)

            # Save
            np.save(out / f'{pid}_emb.npy',   embeddings.numpy().astype(np.float32))  # (T, 128)
            np.save(out / f'{pid}_dt.npy',    delta_t.numpy().astype(np.float32))      # (T,)
            with open(out / f'{pid}_meta.json', 'w') as f:
                json.dump([{
                    'time': t.isoformat(),
                    **m
                } for t, m in zip(times, meta)], f)

        # Checkpoint after each chunk completes
        tmp = CKPT + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump({'next_idx': chunk_end, 'skipped': skipped}, f)
        os.replace(tmp, CKPT)

    # Cleanup
    if Path(CKPT).exists():
        Path(CKPT).unlink()

    print(f'Done. Skipped {skipped} patients.')

# Compute Δt (days from each event to the last event)
def compute_delta_t(times: list) -> torch.Tensor:
    """
    Δt[i] = days between event i and the final event in the sequence.
    Used for temporal decay weighting.
    """
    t_last = times[-1]
    deltas = [(t_last - t).total_seconds() / 86400 for t in times]
    return torch.tensor(deltas, dtype=torch.float32)         # (T,)

if __name__ == '__main__':
    
    lab_encoder        = LabPanelEncoder()
    omr_encoder        = OMREncoder()
    special_encoder    = SpecialTokenEncoder()
    admission_encoder  = AdmissionEncoder()
    icu_encoder        = ICUEncoder()
    transfer_encoder   = TransferEncoder()
    outpatient_encoder = OutpatientEncoder()

    # Build and save timeline for all Patient
    build_and_save_all_timelines(
        all_patient_ids,
        admission_nodes,
        kg_embeddings,
        name_to_idx,
        lab_encoder,
        omr_encoder,
        special_encoder,
        admission_encoder,
        icu_encoder,
        transfer_encoder,
        outpatient_encoder
    )
