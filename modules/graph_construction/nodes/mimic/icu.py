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

# Start with each ICU stay
    icustay = read_full('icu', 'icustays')

    # Deal with year offset 

        to_date(icustay, 'intime')
        to_date(icustay, 'outtime')

        icustay = pd.merge(icustay, patient, on = 'subject_id', how = 'left')
        date_cols = ['intime', 'outtime']
        for col in date_cols:
            icustay[col] = pd.to_datetime(icustay[col], errors='coerce')
        for col in date_cols:
            icustay[col] = shift_year(icustay[col], icustay['offset'])

    icustay['los'] = (icustay['outtime'] - icustay['intime']).dt.total_seconds()
    icustay['unit'] = icustay['first_careunit']
    icustay = icustay.drop(['first_careunit', 'last_careunit', 'change_unit'], axis = 1)

    # Write Stay nodes

        query = """
            UNWIND $rows AS row

            MERGE (d:ICU:Stay:Test:MIMIC {id: row.stay_id})
            MATCH (p:Admission:Test:MIMIC {id: row.hadm_id})

            SET d.start_time = row.intime,
                d.name = toString(row.subject_id) + '_' + toString(row.hadm_id) + '_' + toString(row.stay_id),
                d.end_time = row.outtime,
                d.length_of_stay = row.los,
                d.unit = row.unit

            MERGE (p)-[:HAS_ICUSTAY]->(d)
            """

        # Process in batches
        for i in tqdm(range(start_idx, len(icustay), BATCH_SIZE), desc="Batch processing"):

            batch = icustay.iloc[i:i+BATCH_SIZE]

            rows = []
            for _, row in batch.iterrows():
                rows.append({
                    'subject_id': row['subject_id'],
                    'hadm_id': row['hadm_id'],
                    "stay_id": row["stay_id"],
                    "intime": row["intime"],
                    'outtime': row['outtime'],
                    'los': row['los'],
                    'unit': row['unit']
                })

            dml_ddl_neo4j(
                query,
                progress=False,
                rows=rows
            )

