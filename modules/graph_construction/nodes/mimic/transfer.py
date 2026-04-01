from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict
import duckdb
from dateutil.relativedelta import relativedelta

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from shared_functions.gg_sheet_drive import *
from modules.dataset_preprocessing.utils import *

# Path definitions
data_dir = os.getenv('DATA_DIR')
mimic_path = os.path.join(data_dir, 'mimic_iv')
hosp = os.path.join(mimic_path, 'hosp')
icu = os.path.join(mimic_path, 'icu')

start_idx = 0
BATCH_SIZE = 500

# Definitions

transfer = read_full('hosp', 'transfers')

to_date(transfer, 'intime')
to_date(transfer, 'outtime')
transfer = transfer.sort_values(['subject_id', 'hadm_id', 'intime'])

# Drop rows with missing hadm_id, which means no related admission
transfer = transfer.dropna(subset = 'hadm_id')

title(transfer, 'eventtype')
title(transfer, 'careunit')

## For time offset realignment

    transfer = pd.merge(transfer, patient, on = 'subject_id', how = 'left')

    date_cols = ['intime', 'outtime']
    for col in date_cols:
        transfer[col] = pd.to_datetime(transfer[col], errors='coerce')

    for col in date_cols:
        transfer[col] = shift_year(transfer[col], transfer['offset'])

### Rename labels

    # For Eventtype
        mapping = {
            'Ed': 'Emergency Department',
            'Admit': 'Admission'
        }

        transfer['eventtype'] = transfer['eventtype'].replace(mapping)

    # For Care Unit
        mapping_careunit = {

            'Medical Intensive Care Unit (Micu)'                  : 'Medical Intensive Care Unit (MICU)',
            'Surgical Intensive Care Unit (Sicu)'                 : 'Surgical Intensive Care Unit (SICU)',
            'Medical/Surgical Intensive Care Unit (Micu/Sicu)'    : 'Medical/Surgical Intensive Care Unit (MICU/SICU)',
            'Cardiac Vascular Intensive Care Unit (Cvicu)'        : 'Cardiac Vascular Intensive Care Unit (CVICU)',
            'Coronary Care Unit (Ccu)'                            : 'Coronary Care Unit (CCU)',
            'Neuro Surgical Intensive Care Unit (Neuro Sicu)'     : 'Neuro Surgical Intensive Care Unit (Neuro SICU)',
            'Trauma Sicu (Tsicu)'                                 : 'Trauma SICU (TSICU)',
            'Intensive Care Unit (Icu)'                           : 'Intensive Care Unit (ICU)',
            'Special Care Nursery (Scn)'                          : 'Special Care Nursery (SCN)',

            'Med/Surg'                      : 'Medical/Surgical',
            'Med/Surg/Gyn'                  : 'Medical/Surgical/Gynecology',
            'Medical/Surgical (Gynecology)' : 'Medical/Surgical/Gynecology',
            'Med/Surg/Trauma'               : 'Medical/Surgical/Trauma',

            'Pacu': 'Post-Anesthesia Care Unit (PACU)'
        }

        transfer['careunit'] = transfer['careunit'].replace(mapping_careunit)

#### Connect subsequent Transfer between 1 Admission, no Admission-Transfer yet

    transfer['next_transfer'] = transfer.groupby(['subject_id', 'hadm_id'])['transfer_id'].shift(-1).astype('Int64')
    transfer['duration'] = (
        transfer['outtime'] - transfer['intime']
    ).dt.total_seconds()

    # Save the Transfer-Transfer edge in a distinct dataframe
    transfer_edge = transfer[['transfer_id', 'next_transfer', 'duration']]
    transfer_edge = transfer_edge.dropna(subset = 'next_transfer')

##### Set up Transfer Node and connect with Admission (temporarily by Relation, not temporal)

    query = """
        UNWIND $rows AS row

        MERGE (t:Transfer:Test:MIMIC {id: row.transfer_id})
        MATCH (adm:Admission:Test:MIMIC {id: row.hadm_id})

        SET t.patient_id        = row.subject_id,
            t.admission_id      = row.hadm_id,
            t.name              = toString(row.subject_id) + '_' + toString(row.hadm_id) + '_' + toString(row.transfer_id),
            t.type              = row.eventtype,
            t.care_unit         = row.careunit,
            t.start_time        = row.intime,
            t.end_time           = CASE 
                WHEN row.outtime IS NULL THEN row.intime
                ELSE row.outtime 
            END,
            t.duration          = row.duration

        MERGE (adm)-[:HAS_TRANSFER]->(t)
        """

    for i in tqdm(range(start_idx, len(transfer), BATCH_SIZE), desc="Batch processing"):

        batch = transfer.iloc[i:i+BATCH_SIZE]

        rows = []
        for _, row in batch.iterrows():
            rows.append({
                "subject_id"    : int(row["subject_id"]),
                "hadm_id"       : int(row["hadm_id"]),
                "transfer_id"   : int(row["transfer_id"]),
                "eventtype"     : row["eventtype"] if pd.notna(row["eventtype"]) else "Unknown",
                "careunit"      : row["careunit"] if pd.notna(row["careunit"]) else None,
                "intime"        : row["intime"].isoformat() if pd.notna(row["intime"]) else None,
                "outtime"       : row["outtime"].isoformat() if pd.notna(row["outtime"]) else None,
                "duration"      : float(row["duration"]) if pd.notna(row["duration"]) else 0
            })

        dml_ddl_neo4j(
            query,
            progress=False,
            rows=rows
        )

###### Setup Transfer-Transfer by Transfer-to 
    query = """
        UNWIND $rows AS row

        MATCH (prev:Transfer:Test:MIMIC {id: row.transfer_id})
        MATCH (next:Transfer:Test:MIMIC {id: row.next_transfer})

        MERGE (prev)-[r:TRANSFER_TO]->(next)
        SET r.time_delay = row.time_delay
        """

    for i in tqdm(range(start_idx, len(transfer_edge), BATCH_SIZE), desc="Batch processing"):

        batch = transfer_edge.iloc[i:i+BATCH_SIZE]

        rows = []
        for _, row in batch.iterrows():
            rows.append({
                "transfer_id": row["transfer_id"],
                "next_transfer": row["next_transfer"],
                "time_delay": float(row["duration"]) if pd.notna(row["duration"]) else None
            })

        dml_ddl_neo4j(
            query,
            progress=False,
            rows=rows
        )

        