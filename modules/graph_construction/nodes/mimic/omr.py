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
omr = read_full('hosp', 'omr')

## Reshift date offset
    omr = pd.merge(omr, patient, on = 'subject_id', how = 'left')
    date_cols = ['chartdate']
    for col in date_cols:
        omr[col] = pd.to_datetime(omr[col], errors='coerce')
    for col in date_cols:
        omr[col] = shift_year(omr[col], omr['offset'])

### Pivot related metrics into columns

    omr['result_value_raw'] = omr['result_value']

    # Normalize names
    omr_mapping = {
        'BMI (kg/m2)'                     : 'BMI',
        'Weight (Lbs)'                    : 'Weight',
        'Height (Inches)'                 : 'Height',
        'Blood Pressure Sitting'          : 'Blood Pressure',
        'Blood Pressure Lying'            : 'Blood Pressure',
        'Blood Pressure Standing'         : 'Blood Pressure',
        'Blood Pressure Standing (1 min)' : 'Blood Pressure',
        'Blood Pressure Standing (3 mins)': 'Blood Pressure',
    }
    omr['result_name'] = omr['result_name'].replace(omr_mapping)

    # Handle BP separately
    bp = omr[omr['result_name'] == 'Blood Pressure'].copy()
    bp[['Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic']] = (
        bp['result_value_raw']
        .str.split('/', expand=True)
        .astype(float)
    )
    bp_pivot = (
        bp.groupby(['subject_id', 'chartdate'])[['Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic']]
        .mean()
        .reset_index()
    )

    # Pivot non-BP
    omr['result_value'] = pd.to_numeric(omr['result_value'], errors='coerce')
    omr_no_bp = omr[omr['result_name'] != 'Blood Pressure']

    omr_pivot = (
        omr_no_bp
        .groupby(['subject_id', 'chartdate', 'result_name'])['result_value']
        .mean()
        .unstack('result_name')
        .reset_index()
    )
    omr_pivot.columns.name = None

    # Merge BP back
    omr_final = omr_pivot.merge(bp_pivot, on=['subject_id', 'chartdate'], how='left')

    omr = omr_final

#### Fillna for Weight, Height, BMI based on the Formula

    omr = omr.sort_values(['subject_id', 'chartdate']).reset_index(drop=True)

    # Fill Height and cumulative max to ensure consistency
    omr['Height'] = omr.groupby('subject_id')['Height'].transform(lambda x: x.ffill().cummax())

    # Compute missing Weight and BMI using filled Height
    omr['Weight'] = omr['Weight'].fillna(round(omr['BMI'] * (omr['Height'] ** 2) / 703, 2))
    omr['BMI']    = omr['BMI'].fillna(round(703 * omr['Weight'] / (omr['Height'] ** 2), 1))

##### Create Cumulative OMR index based on patient id

    omr = omr.sort_values(['subject_id', 'chartdate']).reset_index(drop=True)

    # Create cumulative count within each subject as index
    omr['omr_id'] = omr.groupby('subject_id').cumcount() + 1

    # Combine subject_id + index as string ID
    omr['omr_id'] = omr['subject_id'].astype(str) + '_' + omr['omr_id'].astype(str)

###### Create Nodes and connect with Patient
    query = """
        UNWIND $rows AS row

        MERGE (r:OMR:Test:MIMIC {id: row.omr_id})
        MATCH (p:Patient:Test:MIMIC {id: row.patient_id})

        SET r.name                     = row.omr_id,
            r.date                     = row.date,
            r.BMI                      = row.BMI,
            r.Height                   = row.Height,
            r.Weight                   = row.Weight,
            r.eGFR                     = row.eGFR,
            r.blood_pressure_systolic  = row.blood_pressure_systolic,
            r.blood_pressure_diastolic = row.blood_pressure_diastolic

        MERGE (p)-[:OUTPATIENT_TEST]->(r)
        """

    for i in tqdm(range(start_idx, len(omr), BATCH_SIZE), desc="Batch processing"):

        batch = omr.iloc[i:i+BATCH_SIZE]

        rows = []
        for _, row in batch.iterrows():
            rows.append({
                "omr_id"                    : row["omr_id"],
                "patient_id"                : int(row["subject_id"]),
                "date"                      : row["chartdate"].isoformat(),
                "BMI"                       : float(row["BMI"]) if pd.notna(row["BMI"]) else None,
                "Height"                    : float(row["Height"]) if pd.notna(row["Height"]) else None,
                "Weight"                    : float(row["Weight"]) if pd.notna(row["Weight"]) else None,
                "eGFR"                      : float(row["eGFR"]) if pd.notna(row["eGFR"]) else None,
                "blood_pressure_systolic"   : float(row["Blood_Pressure_Systolic"]) if pd.notna(row["Blood_Pressure_Systolic"]) else None,
                "blood_pressure_diastolic"  : float(row["Blood_Pressure_Diastolic"]) if pd.notna(row["Blood_Pressure_Diastolic"]) else None,
            })

        dml_ddl_neo4j(
            query,
            progress=False,
            rows=rows
        )

        