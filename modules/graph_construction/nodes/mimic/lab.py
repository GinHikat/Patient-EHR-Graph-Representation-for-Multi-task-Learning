import pandas as pd
import numpy as np
import json
from collections import defaultdict
import duckdb
from tqdm import tqdm

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from shared_functions.gg_sheet_drive import *
from shared_functions.global_functions import *
from modules.dataset_preprocessing.utils import *
from modules.dataset_preprocessing.utils import shift_year

from dotenv import load_dotenv
load_dotenv() 

# Paths setup
data_dir = 'F:/Din/Study/Education/Projects/Thesis/data'
mimic_path = os.path.join(data_dir, 'mimic_iv')
hosp_path = os.path.join(mimic_path, 'hosp')

input_path = os.path.join(hosp_path, 'lab_cleaned.csv')
patient_path = os.path.join(base_data_dir, 'patient.csv')
output_path = os.path.join(hosp_path, 'lab_cleaned_v2.csv')

print("Loading patient data...")
patient = pd.read_csv(patient_path)

type = 'hosp'
con = duckdb.connect()

# Due to the large size of the Lab table, use duckdb for stream processing
def initial_clean_input():

    name = 'labevents'
    sample_path = os.path.join(mimic_path, type, f'{name}.csv')
    output_path = os.path.join(mimic_path, type, 'lab_cleaned.csv')
    
    query = f"""
    SELECT subject_id, hadm_id, itemid, charttime, valuenum, valueuom, flag
    FROM read_csv_auto('{sample_path}',
        ignore_errors = True
    )
    WHERE valuenum IS NOT NULL
    """
    result = con.execute(query)
    chunk_size = 50000 
    first_chunk = True

    while True:
        try:
            chunk = result.fetch_df_chunk(chunk_size)
        except Exception as e:
            print(f"Finished or encountered error: {e}")
            break
            
        if chunk.empty:
            break
        
        # Process the chunk
        chunk['flag'] = chunk['flag'] == 'abnormal'
        
        chunk.to_csv(
            output_path, 
            mode='a', 
            index=False, 
            header=first_chunk
        )
        
        if first_chunk:
            print(f"Streaming started... writing to {output_path}")
            first_chunk = False
    print(f"Processing complete! Results saved to {output_path}")

## Match time offset for each patient to get the unified time
def clean_input_lab():
    
    query = f"""
    SELECT subject_id, hadm_id, itemid, charttime, valuenum, valueuom, flag
    FROM read_csv_auto('{input_path}', ignore_errors = True)
    """

    print(f"Starting stream from {input_path}...")
    result = con.execute(query)

    chunk_size = 1000
    first_chunk = True

    while True:
        try:
            chunk = result.fetch_df_chunk(chunk_size)
        except Exception as e:
            break
            
        if chunk.empty:
            break
        
        # Merge with patient data for the 'offset' column
        chunk = pd.merge(chunk, patient, on='subject_id', how='left')

        date_cols = ['charttime']
        for col in date_cols:
            chunk[col] = pd.to_datetime(chunk[col], errors='coerce')

        if 'offset' in chunk.columns:
            for col in date_cols:
                chunk[col] = shift_year(chunk[col], chunk['offset'])

        chunk = chunk.drop(['dead', 'dead_year', 'offset'], axis=1, errors='ignore')
        
        chunk.to_csv(
            output_path, 
            mode='a', 
            index=False, 
            header=first_chunk
        )
        
        if first_chunk:
            first_chunk = False
            print(f"Streaming to {output_path}...")

    print("Zeroing complete! Results saved to lab_cleaned.csv")

### Match Lab item from the Item check table
def match_lab_item():

    # Filtered_df is the item check table
    filtered_df = filtered_df.rename(columns = {'id': 'itemid'})
    filtered_df['itemid'] = filtered_df['itemid'].astype(int)

    sample_path = os.path.join(mimic_path, type, 'lab_cleaned_v2.csv')
    output_path = os.path.join(mimic_path, type, 'lab_clean.csv') 

    query = f"SELECT * FROM read_csv_auto('{sample_path}')"
    print(f"Streaming from {sample_path}...")
    result = con.execute(query)
    chunk_size = 5000
    first_chunk = True

    while True:
        try:
            lab_chunk = result.fetch_df_chunk(chunk_size)
        except Exception as e:
            break
            
        if lab_chunk.empty:
            break
        
        lab_chunk = lab_chunk.merge(filtered_df[['itemid', 'name']], on='itemid', how='left')
        
        lab_chunk = lab_chunk.drop(['itemid'], axis=1)
        lab_chunk = lab_chunk.rename(columns={'name': 'stat'})

        lab_chunk.to_csv(
            output_path, 
            mode='a', 
            index=False, 
            header=first_chunk
        )
        
        if first_chunk:
            print(f"First chunk written to {output_path}...")
            first_chunk = False
    print(f"Processing complete! Results saved to {output_path}")

