import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

import re
import pandas as pd
from tqdm import tqdm
import ast
import duckdb

from modules.processing import Extractor
extractor = Extractor()

data_dir = os.path.join(project_root, 'data', 'Note')
cleaned_data_dir = os.path.join(data_dir, 'cleaned')
discharge_path = os.path.join(data_dir, 'discharge.csv')
output_path = os.path.join(cleaned_data_dir, 'discharge_processed.csv')

os.makedirs(cleaned_data_dir, exist_ok=True)

processed_ids = set()
if os.path.exists(output_path):
    print(f"Loading checkpoint from {output_path}...")
    # Using duckdb to quickly get unique hadm_ids from existing output
    processed_ids = set(duckdb.query(f"SELECT DISTINCT hadm_id FROM '{output_path}'").to_df()['hadm_id'])
    print(f"Resuming: {len(processed_ids)} records already processed.")

con = duckdb.connect()

csv_options = "ignore_errors=True, null_padding=True, parallel=False"

print("Starting full dataset processing. Excluding metadata: subject_id, note_id.")
rel = con.sql(f"SELECT * EXCLUDE (subject_id, note_id) FROM read_csv_auto('{discharge_path}', {csv_options})")

batch_size = 1000
total_to_process = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{discharge_path}', {csv_options})").fetchone()[0]

pbar_total = tqdm(total=total_to_process, desc="Overall Progress", unit="row")
pbar_total.update(len(processed_ids))

# Process in batches using LIMIT and OFFSET for better control
offset = 0
while offset < total_to_process:
    # Query only the next batch
    query = f"""
        SELECT * EXCLUDE (subject_id, note_id) 
        FROM read_csv_auto('{discharge_path}', {csv_options})
        LIMIT {batch_size} OFFSET {offset}
    """
    batch_df = con.execute(query).df()
    
    if batch_df.empty:
        break
    
    current_batch_ids = set(batch_df['hadm_id'])
    
    # Check if this batch needs processing (at least one ID not in processed_ids)
    if not current_batch_ids.issubset(processed_ids):
        # Filter out already processed within the batch
        to_process_df = batch_df[~batch_df['hadm_id'].isin(processed_ids)]
        
        if not to_process_df.empty:
            # Process extraction
            extracted_df = extractor.batch_extracting_discharge(to_process_df)
            
            if not extracted_df.empty:
                header = not os.path.exists(output_path)
                # Save and force flush to disk
                extracted_df.to_csv(output_path, mode='a', index=False, header=header)
                
                # Update processed set
                processed_ids.update(to_process_df['hadm_id'])
    
    batch_len = len(batch_df)
    offset += batch_len
    pbar_total.update(batch_len)

pbar_total.close()
print(f"\nProcessing complete. Output saved/updated at: {output_path}")
