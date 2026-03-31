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

# Definitions
    patient = read_full('hosp', 'patients')

    patient['dead'] = patient['dod'].notna().astype(int)

    # Use anchor year group to calculate real year offset
    patient['real_year_start'] = patient['anchor_year_group'].str[:4].astype(int)
    patient['year_offset'] = patient['real_year_start'] - patient['anchor_year']

    patient['dod'] = patient['dod'].apply(lambda x: int(x[:4]) if not pd.isna(x) else x)

    patient['dead_year'] = patient['dod'] + patient['year_offset']

    patient = patient[['subject_id', 'gender', 'anchor_age', 'real_year_start', 'year_offset', 'dead', 'dead_year']]

    patient.columns = ['id', 'gender', 'age', 'first_adm', 'offset', 'dead', 'dead_year']

## Import into Database
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

### Enrich with admission data

    adm = read_full('hosp', 'admissions')

    subject = adm[['subject_id', 'insurance', 'marital_status', 'language', 'race']]

    def get_mode(x):
        m = x.dropna().mode()
        return m.iloc[0] if len(m) > 0 else np.nan

    cols = ['insurance', 'marital_status', 'language', 'race']

    nunique_per_subject = subject.groupby('subject_id')[cols].nunique().max(axis=1)

    # Split consistent (one value per id for all attributes) and inconsistent (multiple values per id for at least one attribute)
    consistent = subject[subject['subject_id'].isin(nunique_per_subject[nunique_per_subject == 1].index)]
    consistent = consistent.drop_duplicates()

    # For inconsistent subjects, take the mode for each attribute
    inconsistent = subject[subject['subject_id'].isin(nunique_per_subject[nunique_per_subject > 1].index)]
    inconsistent = (
        inconsistent.groupby('subject_id')[cols]
        .agg(get_mode)
        .reset_index()
    )

    patient_add = pd.concat([consistent, inconsistent], axis = 0)

    patient_add[['insurance', 'marital_status', 'language', 'race']] = (
        patient_add[['insurance', 'marital_status', 'language', 'race']].apply(lambda col: col.str.title())
    )

    patient_add = patient_add.drop_duplicates(subset = 'subject_id')
    patient_add['race'] = patient_add['race'].apply(lambda x: 'Other' if x in ['Unknown', 'Unable To Obtain', 'Patient Declined To Answer'] else x)

    # Enrich current nodes
        query = """
            UNWIND $rows AS row

            MATCH (d:Patient:Test:MIMIC {id: row.id})
            SET d.insurance = row.insurance,
                d.marital_status = row.marital_status,
                d.language = row.language,
                d.race = row.race
            """

        # Process in batches
        for i in tqdm(range(start_idx, len(patient_add), BATCH_SIZE), desc="Batch processing"):

            batch = patient_add.iloc[i:i+BATCH_SIZE]

            rows = []
            for _, row in batch.iterrows():
                rows.append({
                    "id": row["subject_id"],
                    "insurance": row["insurance"] if pd.notna(row["insurance"]) else 'Other',
                    "marital_status": row["marital_status"] if pd.notna(row["marital_status"]) else 'Other',
                    "language": row["language"] if pd.notna(row["language"]) else 'Other',
                    'race': row['race'] if pd.notna(row["race"]) else 'Other'
                })

            dml_ddl_neo4j(
                query,
                progress=False,
                rows=rows
            )

