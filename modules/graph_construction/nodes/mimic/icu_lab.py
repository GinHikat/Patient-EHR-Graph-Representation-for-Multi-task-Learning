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
icu_path = os.path.join(mimic_path, 'icu')

import pyarrow as pa
import pyarrow.parquet as pq

# 1. Connect to DuckDB
con = duckdb.connect()

# 2. Register stay_ids list as a temporary DuckDB table
stay_ids_df = pd.DataFrame({'stay_id': stay_ids})
con.register('stay_ids_df', stay_ids_df)

# 3. Configure file paths (pre-formatted to avoid backslash in f-string)
chartevents_csv = os.path.join(mimic_path, 'icu', 'chartevents.csv')
chartevents_csv_clean = chartevents_csv.replace('\\', '/')

icu_folder = os.path.join(mimic_path, 'icu')
output_parquet = os.path.join(icu_folder, 'chartevents_aggregated.parquet').replace('\\', '/')

print(f"Reading from: {chartevents_csv}")
print(f"Streaming and aggregating into: {output_parquet}...")

# 4. Formulate the streaming query
query = f"""
WITH filtered_events AS (
    -- Stream CSV and filter rows with stay_id in target stays
    SELECT 
        c.subject_id, 
        c.hadm_id, 
        c.stay_id, 
        c.charttime, 
        c.itemid, 
        c.valuenum
    FROM read_csv_auto('{chartevents_csv_clean}', ignore_errors=True) c
    INNER JOIN stay_ids_df s ON c.stay_id = s.stay_id
    WHERE c.valuenum IS NOT NULL
),
clean_events AS (
    -- Average duplicate records of same itemid at same charttime
    SELECT 
        subject_id, 
        hadm_id, 
        stay_id, 
        charttime, 
        itemid, 
        avg(valuenum) as valuenum
    FROM filtered_events
    GROUP BY subject_id, hadm_id, stay_id, charttime, itemid
)
-- Group by stay_id and charttime, and create key-value dictionary
SELECT 
    subject_id, 
    hadm_id, 
    stay_id, 
    charttime,
    to_json(map(list(CAST(itemid AS VARCHAR)), list(valuenum))) as value
FROM clean_events
GROUP BY subject_id, hadm_id, stay_id, charttime
"""

# Execute the query to open a streaming cursor
result = con.execute(query)

chunk_size = 50000  # Number of aggregated records to process in each chunk
writer = None

# Initialize tqdm progress bar
pbar = tqdm(desc="Streaming & Aggregating ICU Chartevents", unit=" chunks")

try:
    while True:
        try:
            # Stream/fetch one chunk from DuckDB
            chunk = result.fetch_df_chunk(chunk_size)
        except Exception:
            break
            
        if chunk.empty:
            break
            
        # Convert pandas DataFrame to PyArrow Table
        arrow_table = pa.Table.from_pandas(chunk)
        
        # Initialize Parquet Writer on the very first chunk
        if writer is None:
            writer = pq.ParquetWriter(output_parquet, arrow_table.schema)
            
        # Write chunk incrementally to disk
        writer.write_table(arrow_table)
        
        # Update progress bar
        pbar.update(1)
        
finally:
    # Safely close the file and progress bar
    if writer is not None:
        writer.close()
    pbar.close()
    
print(f"\nSUCCESS: All aggregated records successfully saved to {output_parquet}")
