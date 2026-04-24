import duckdb
import os
import pandas as pd
from tqdm import tqdm

# Paths
input_file = '/home/hngoc/gin/data/diagnosis_final.csv'
output_file = '/home/hngoc/gin/data/diagnosis_finals.csv'

file_size = os.path.getsize(input_file)

print(f"Starting batch processing {input_file}...")

con = duckdb.connect()

query = f"""
    SELECT 
        hadm_id, 
        radiology, 
        discharge_input AS discharge, 
        diagnosis
    FROM read_csv_auto('{input_file}')
"""

rel = con.sql(query)
reader = rel.fetch_record_batch(rows_per_batch=100_000)

first_chunk = True

with tqdm(desc="Processing chunks", unit="chunk") as pbar:
    while True:
        try:
            chunk = reader.read_next_batch()
            df = chunk.to_pandas()
            
            # Write to CSV
            if first_chunk:
                df.to_csv(output_file, index=False, mode='w', header=True)
                first_chunk = False
            else:
                df.to_csv(output_file, index=False, mode='a', header=False)
            
            pbar.update(1)
        except StopIteration:
            break

print(f"\nSuccessfully created {output_file}")
size_gb = os.path.getsize(output_file) / (1024**3)
print(f"New file size: {size_gb:.2f} GB")
