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
diagnosis_path = os.path.join(cleaned_data_dir, 'diagnosis_clean.csv')
output_path = os.path.join(cleaned_data_dir, 'discharge_diagnosis_combined.csv')

os.makedirs(cleaned_data_dir, exist_ok=True)

print("Loading diagnosis data...")
diagnosis = pd.read_csv(diagnosis_path)
diagnosis['category'] = diagnosis['category'].apply(ast.literal_eval)

processed_ids = set()
if os.path.exists(output_path):
    print(f"Loading checkpoint from {output_path}...")
    processed_ids = set(duckdb.query(f"SELECT DISTINCT hadm_id FROM '{output_path}'").to_df()['hadm_id'])
    print(f"Resuming: {len(processed_ids)} records already processed.")

con = duckdb.connect()

csv_options = "ignore_errors=True, null_padding=True, parallel=False"

print("Starting full dataset processing. Excluding metadata: subject_id, note_id.")
rel = con.sql(f"SELECT * EXCLUDE (subject_id, note_id) FROM read_csv_auto('{discharge_path}', {csv_options})")

batch_size = 1000
result_set = rel.execute()

total_to_process = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{discharge_path}', {csv_options})").fetchone()[0]
pbar_total = tqdm(total=total_to_process, desc="Overall Progress", unit="row")

while True:
    batch_df = result_set.fetch_df_chunk(batch_size)
    if batch_df is None or batch_df.empty:
        break
    
    original_batch_len = len(batch_df)
    batch_df = batch_df[~batch_df['hadm_id'].isin(processed_ids)]
    
    if batch_df.empty:
        pbar_total.update(original_batch_len)
        continue
    
    extracted_df = extractor.batch_extracting_discharge(batch_df)
    joined_df = extracted_df.merge(diagnosis, on='hadm_id', how='inner')
    
    if not joined_df.empty:
        header = not os.path.exists(output_path)
        joined_df.to_csv(output_path, mode='a', index=False, header=header)
        processed_ids.update(joined_df['hadm_id'])

    pbar_total.update(original_batch_len)

pbar_total.close()
print(f"\nProcessing complete. Output saved/updated at: {output_path}")
