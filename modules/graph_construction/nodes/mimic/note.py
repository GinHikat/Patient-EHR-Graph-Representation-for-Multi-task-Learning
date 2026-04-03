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

# Deal with Discharge first
    discharge = read_full('note', 'discharge')

    query = """
        UNWIND $rows AS row

        MERGE (d:Note:Test:MIMIC {id: row.note_id})
        MATCH (p:Admission:Test:MIMIC {id: row.hadm_id})

        SET d.name = row.note_id,
            d.time = row.charttime,
            d.text = row.text,
            d.admission_id = row.hadm_id,
            d.patient_id = row.subject_id,
            d.type = 'Discharge'

        MERGE (p)-[:HAS_NOTE]->(d)
        """

    # Process in batches
    for i in tqdm(range(start_idx, len(discharge), BATCH_SIZE), desc="Batch processing"):

        batch = discharge.iloc[i:i+BATCH_SIZE]

        rows = []
        for _, row in batch.iterrows():
            rows.append({
                'subject_id': row['subject_id'],
                'hadm_id': row['hadm_id'],
                "note_id": row["note_id"],
                "charttime": row["charttime"],
                'text': row['text']
            })

        dml_ddl_neo4j(
            query,
            progress=False,
            rows=rows
        )

    