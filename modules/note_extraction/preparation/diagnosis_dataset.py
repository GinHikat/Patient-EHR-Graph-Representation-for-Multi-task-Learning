import sys, os
import pandas as pd
import sqlite3
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

from processing import Extractor

# Constant Paths
DIAGNOSIS_TRAIN = os.path.join(script_dir, 'diagnosis_train.csv')
DISCHARGE_CSV = os.path.join(script_dir, 'discharge.csv')
OUTPUT_CSV = os.path.join(script_dir, 'diagnosis_final.csv')
DB_PATH = os.path.join(script_dir, 'temp_lookup.db')

def setup_lookup_db():
    """Import discharge records into a temporary SQLite DB for fast indexed lookup."""
    print("Setting up temporary lookup database (this may take a few minutes)...")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE discharge (hadm_id INTEGER PRIMARY KEY, text TEXT)")
    
    # Read discharge.csv in chunks and insert into SQLite
    chunk_iter = pd.read_csv(DISCHARGE_CSV, usecols=['hadm_id', 'text'], chunksize=100000)
    for chunk in tqdm(chunk_iter, desc="Indexing discharge records"):
        chunk.dropna(subset=['hadm_id', 'text'], inplace=True)
        chunk.to_sql('discharge', conn, if_exists='append', index=False)
    
    print("Creating index for faster lookups...")
    # PRIMARY KEY already creates an index, but we ensure it's optimized
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_hadm_id ON discharge(hadm_id)")
    conn.commit()
    return conn

def process_streaming():
    extractor = Extractor()
    conn = setup_lookup_db()
    
    print(f"Starting streaming process to {OUTPUT_CSV}...")
    
    # Write header for the first chunk
    first_chunk = True
    
    # Read diagnosis_train.csv in chunks
    # 500 is a safe chunk size to avoid SQLite's parameter limit (often 999)
    chunk_iter = pd.read_csv(DIAGNOSIS_TRAIN, usecols=['hadm_id', 'diagnosis', 'Note'], chunksize=500)
    
    for chunk in tqdm(chunk_iter, desc="Processing diagnosis chunks"):
        # Rename 'Note' to 'radiology' to match extractor expectations
        chunk.columns = ['hadm_id', 'diagnosis', 'radiology']
        
        # Merge with discharge text via SQL
        hadm_ids = tuple(chunk['hadm_id'].tolist())
        query = f"SELECT hadm_id, text as discharge FROM discharge WHERE hadm_id IN {hadm_ids}"
        discharge_text = pd.read_sql_query(query, conn)
        
        # Merge chunk with the retrieved discharge text
        df_chunk = chunk.merge(discharge_text, on='hadm_id', how='inner')
        
        if df_chunk.empty:
            continue
            
        # Run extractor on the combined chunk
        df_chunk = extractor.batch_diagnosis_input(df_chunk)
        
        # Append to output CSV
        df_chunk.to_csv(OUTPUT_CSV, mode='a', index=False, header=first_chunk)
        first_chunk = False
        
    conn.close()
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    print("Processing complete!")

if __name__ == "__main__":
    process_streaming()

