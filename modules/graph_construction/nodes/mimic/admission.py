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
adm = read_full('hosp', 'admissions')
adm = adm.drop(['insurance', 'marital_status', 'race', 'language', 'admit_provider_id', 'edregtime', 'edouttime'], axis = 1)

to_date(adm, 'admittime')
to_date(adm, 'dischtime')
to_date(adm, 'deathtime')

adm['length_of_stay'] = (adm['dischtime'] - adm['admittime']).dt.total_seconds()

## For year offset alignment
    adm = pd.merge(adm, patient, on = 'subject_id', how = 'left')

    date_cols = ['admittime', 'dischtime', 'deathtime']
    for col in date_cols:
        adm[col] = pd.to_datetime(adm[col], errors='coerce')

    for col in date_cols:
        adm[col] = shift_year(adm[col], adm['offset'])

### Title case for categorical columns
    cols = ['admission_type', 'admission_location', 'discharge_location']

    for col in cols:
        adm[col] = adm[col].apply(lambda x: x.title() if isinstance(x, str) else x)

    mapping = {
        'Ew Emer.': 'Emergency Ward Emergency',
        'Eu Observation': 'Emergency Unit Observation',
        'Direct Emer.': 'Direct Emergency'
    }

    adm['admission_type'] = adm['admission_type'].replace(mapping)
    adm['admission_location'] = adm['admission_location'].replace({'Pacu': 'Post-Anesthesia Care Unit'}) 

#### Calculate delay between Admission and set Readmission for Admission with next admission
    adm = adm.sort_values(['subject_id', 'admittime'])

    adm['next_hadm'] = adm.groupby('subject_id')['hadm_id'].shift(-1)
    adm['next_admittime'] = adm.groupby('subject_id')['admittime'].shift(-1)

    adm['time_delay'] = (
        adm['next_admittime'] - adm['dischtime']
    ).dt.total_seconds()

##### Create a whole distinct dataframe for Admission-Admission delay
    delay_df = adm[['subject_id', 'hadm_id', 'next_hadm', 'time_delay']].copy()

    delay_df = delay_df.rename(columns={
        'hadm_id': 'previous_hadm'
    })

    # Drop rows where there is no next admission
    delay_df = delay_df.dropna(subset=['next_hadm'])

    delay_df['readmission'] = (delay_df['time_delay'] > 0).astype(int)

    adm['readmission'] = 0  

    adm.loc[
        adm['hadm_id'].isin(
            delay_df.loc[delay_df['time_delay'] > 0, 'previous_hadm']
        ),
        'readmission'
    ] = 1

###### Create Admission node and connect with Patient node
    query = """
    UNWIND $rows AS row

    MERGE (d:Admission:Test:MIMIC {id: row.hadm_id})
    MATCH (p:Patient:Test:MIMIC {id: row.subject_id})

    SET d.admittime = datetime(row.admittime),
        d.dischtime = datetime(row.dischtime),
        d.deathtime = CASE 
            WHEN row.deathtime IS NULL THEN NULL 
            ELSE datetime(row.deathtime) 
        END,
        d.name = row.hadm_id,
        d.readmission = row.readmission,
        d.admission_type = row.admission_type,
        d.admission_location = row.admission_location,
        d.discharge_location = row.discharge_location,
        d.length_of_stay = row.length_of_stay,
        d.inhospital_dead = row.hospital_expire_flag

    MERGE (p)-[:ADMISSION]->(d)
    """

    for i in tqdm(range(start_idx, len(adm), BATCH_SIZE), desc="Batch processing"):

        batch = adm.iloc[i:i+BATCH_SIZE]

        rows = []
        for _, row in batch.iterrows():
            rows.append({
                "subject_id": row["subject_id"],
                "hadm_id": row["hadm_id"],
                "admittime": row["admittime"].isoformat() if pd.notna(row["admittime"]) else None,
                "dischtime": row["dischtime"].isoformat() if pd.notna(row["dischtime"]) else None,
                "deathtime": row["deathtime"].isoformat() if pd.notna(row["deathtime"]) else None,
                "readmission": int(row["readmission"]),
                "admission_type": row["admission_type"] if pd.notna(row["admission_type"]) else "Unknown",
                "admission_location": row["admission_location"] if pd.notna(row["admission_location"]) else "Unknown",
                "discharge_location": row["discharge_location"] if pd.notna(row["discharge_location"]) else "Unknown",
                "length_of_stay": float(row["length_of_stay"]) if pd.notna(row["length_of_stay"]) else None,
                "time_delay": float(row["time_delay"]) if pd.notna(row["time_delay"]) else None,
                "hospital_expire_flag": int(row["hospital_expire_flag"])
            })

        dml_ddl_neo4j(
            query,
            progress=False,
            rows=rows
        )

###### Create Admission-Admission Readmission relationship using the delay_df
    query = """
    UNWIND $rows AS row

    MATCH (prev:Admission:Test:MIMIC {id: row.previous_hadm})
    MATCH (next:Admission:Test:MIMIC {id: row.next_hadm})

    MERGE (prev)-[r:READMISSION]->(next)
    SET r.time_delay = row.time_delay
    """

    for i in tqdm(range(start_idx, len(delay_df), BATCH_SIZE), desc="Batch processing"):

        batch = delay_df.iloc[i:i+BATCH_SIZE]

        rows = []
        for _, row in batch.iterrows():
            rows.append({
                "previous_hadm": row["previous_hadm"],
                "next_hadm": row["next_hadm"],
                "readmission": int(row["readmission"]),
                "time_delay": float(row["time_delay"]) if pd.notna(row["time_delay"]) else None
            })

        dml_ddl_neo4j(
            query,
            progress=False,
            rows=rows
        )

####### Enrich Admission node with drg codes
    drgcode = read_full('hosp', 'drgcodes')

    title(drgcode, 'description')
    drgcode['description'] = drgcode['description'].apply(lambda x: x.replace(',', ', '))

    # Before that, deal with Admission with both APR and HACF code, prioritise APR
        apr = drgcode[drgcode['drg_type'] == 'APR'].drop_duplicates(subset='hadm_id', keep='first')
        hcfa = drgcode[drgcode['drg_type'] == 'HCFA'].drop_duplicates(subset='hadm_id', keep='first')

        # APR takes priority, HCFA fills in the rest
        drg = pd.concat([
            apr,
            hcfa[~hcfa['hadm_id'].isin(apr['hadm_id'])]  # only HCFA-only admissions
        ])

    query = """
        UNWIND $rows AS row

        MATCH(d:Admission:Test {id: row.hadm_id})
        SET d.drg_type = row.drg_type, 
            d.drg_code = row.drg_code,
            d.drg_description = row.description,
            d.drg_severity = row.drg_severity,
            d.drg_mortality = row.drg_mortality
        """

    # Process in batches
    for i in tqdm(range(start_idx, len(drg), BATCH_SIZE), desc="Batch processing"):

        batch = drg.iloc[i:i+BATCH_SIZE]

        rows = []
        for _, row in batch.iterrows():
            rows.append({
                "hadm_id": row["hadm_id"],
                'drg_type': row['drg_type'],
                'drg_code': row['drg_code'],
                'description': row['description'],
                'drg_severity': row['drg_severity'] if row['drg_severity'] else None,
                'drg_mortality': row['drg_mortality'] if row['drg_mortality'] else None
            })

        dml_ddl_neo4j(
            query,
            progress=False,
            rows=rows
        )

######## Start adding Procedure to each Admission 
    procedure = read_full('hosp', 'procedures_icd')

    query = """
        UNWIND $rows AS row

        MATCH(d:Admission:Test:MIMIC {id: row.hadm_id})
        MATCH(p:Procedure:MIMIC:Test {id: row.icd_id})
        
        MERGE (d)-[:HAS_PROCEDURE]->(p)
        """

    # Process in batches
    for i in tqdm(range(start_idx, len(procedure), BATCH_SIZE), desc="Batch processing"):

        batch = procedure.iloc[i:i+BATCH_SIZE]

        rows = []
        for _, row in batch.iterrows():
            rows.append({
                "hadm_id": row["hadm_id"],
                'icd_id': row['icd_code']
            })

        dml_ddl_neo4j(
            query,
            progress=False,
            rows=rows
        )

######## Start adding Diagnosis to each Admission 
    diagnosis = read_full('hosp', 'diagnoses_icd')

    query = """
        UNWIND $rows AS row

        MATCH(d:Admission:Test:MIMIC {id: row.hadm_id})
        MATCH(p:Diagnosis:MIMIC:Test {id: row.icd_id})
        
        MERGE (d)-[:HAS_DIAGNOSIS]->(p)
        """

    # Process in batches
    for i in tqdm(range(start_idx, len(diagnosis), BATCH_SIZE), desc="Batch processing"):

        batch = diagnosis.iloc[i:i+BATCH_SIZE]

        rows = []
        for _, row in batch.iterrows():
            rows.append({
                "hadm_id": row["hadm_id"],
                'icd_id': row['icd_code']
            })

        dml_ddl_neo4j(
            query,
            progress=False,
            rows=rows
        )

  