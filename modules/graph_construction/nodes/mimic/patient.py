from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict
import duckdb

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from shared_functions.gg_sheet_drive import *
from modules.dataset_preprocessing.utils import *
from modules.dataset_preprocessing.utils import *

# Start by simply parsing dimensional tables

# Path definitions
data_dir = os.getenv('DATA_DIR')
mimic_path = os.path.join(data_dir, 'mimic_iv')
hosp = os.path.join(mimic_path, 'hosp')
icu = os.path.join(mimic_path, 'icu')

start_idx = 0
BATCH_SIZE = 500

patient = read_full('hosp', 'patients')

patient['dead'] = patient['dod'].notna().astype(int)

# Use anchor year group to calculate real year offset
# anchor_year_group gives you the real period, anchor_year is the shifted year
# Extract the start year of the group and compare
patient['real_year_start'] = patient['anchor_year_group'].str[:4].astype(int)
patient['year_offset'] = patient['real_year_start'] - patient['anchor_year']

patient['dod'] = patient['dod'].apply(lambda x: int(x[:4]) if not pd.isna(x) else x)

patient['dead_year'] = patient['dod'] + patient['year_offset']

patient = patient[['subject_id', 'gender', 'anchor_age', 'real_year_start', 'year_offset', 'dead', 'dead_year']]

patient.columns = ['id', 'gender', 'age', 'first_adm', 'offset', 'dead', 'dead_year']

# Import into Database
    query = """
        UNWIND $rows AS row

        MERGE (d:Patient:Test:MIMIC {id: row.id})
        SET d.gender = row.gender,
            d.nane = row.id,
            d.age = row.age,
            d.first_adm = row.first_adm,
            d.offset = row.offset,
            d.dead = row.dead,
            d.dead_year = row.dead_year
        """

    # Process in batches
    for i in tqdm(range(start_idx, len(patient), BATCH_SIZE), desc="Batch processing"):

        batch = patient.iloc[i:i+BATCH_SIZE]

        rows = []
        for _, row in batch.iterrows():
            rows.append({
                "id": row["id"],
                "gender": row["gender"],
                'age': row['age'],
                'first_adm': row['first_adm'],
                'offset': row['offset'],
                'dead': row['dead'],
                'dead_year': row['dead_year'] if row['dead_year'] else None
            })

        dml_ddl_neo4j(
            query,
            progress=False,
            rows=rows
        )
