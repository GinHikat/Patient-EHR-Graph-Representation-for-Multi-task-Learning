import pandas as pd
import numpy as np
import duckdb
import re
from tqdm import tqdm

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from shared_functions.global_functions import *
from modules.dataset_preprocessing.utils import *
from modules.models.models import EmbeddingModels

from dotenv import load_dotenv
load_dotenv() 

embedder = EmbeddingModels(1)

prescription = read_full('hosp', 'prescription_clean')

def normalize_dose_unit(unit):
    if pd.isna(unit):
        return unit
    
    unit = str(unit).strip()
    
    # mcg first (before mg, to avoid partial match)
    if re.search(r'mcg|nanogram', unit, re.IGNORECASE):
        return 'mcg'
    
    # mg variants
    if re.search(r'mg', unit, re.IGNORECASE):
        return 'mg'
    
    # g variants (gm, gtt excluded)
    if re.search(r'\bgm\b|\bg\b', unit, re.IGNORECASE):
        return 'g'
    
    return unit  # leave others as-is

prescription['dose_unit_normalized'] = prescription['dose_unit'].apply(normalize_dose_unit)

# g -> mg
mask = prescription['dose_unit_normalized'] == 'g'
prescription.loc[mask, 'dose_value'] = prescription.loc[mask, 'dose_value'] * 1000
prescription.loc[mask, 'dose_unit_normalized'] = 'mg'

# Remove drugs that have rare unusual units (occurence < 50)
counts = prescription['dose_unit_normalized'].value_counts()
valid_units = counts[counts >= 50].index
prescription = prescription[prescription['dose_unit_normalized'].isin(valid_units)]

prescription = prescription.drop(['dose_unit', 'ndc'], axis = 1)
prescription = prescription.rename(columns = {'dose_unit_normalized':'unit'})

# Batch Embedding drug name for DrugBank matching
list_drug = prescription[['drug']].drop_duplicates()
list_drug['drug_embedding'] = list_drug['drug'].apply(lambda x: embedder.encode_text(x))

prescription = prescription.merge(list_drug, on = 'drug', how = 'left')

## Apply Similarity Drug matching
prescription = pd.read_parquet(os.path.join(base_data_dir, 'utils', 'prescription_embedding.parquet'))
prescription['drug_embedding'] = prescription['drug_embedding'].apply(
    lambda x: np.fromstring(x.replace('\n', ' ').strip('[]'), sep=' ') if isinstance(x, str) else x
)
external = pd.read_parquet(os.path.join(base_data_dir, 'utils', 'external.parquet'))

def drug_matching(prescription, external):
    # Convert embeddings to 2D numpy arrays for matrix operations
    prescription_embeddings = np.vstack(prescription['drug_embedding'].values) 
    external_embeddings = np.vstack(external['drug_embedding'].values)          

    # Normalize for cosine similarity
    prescription_norm = prescription_embeddings / np.linalg.norm(prescription_embeddings, axis=1, keepdims=True)
    external_norm = external_embeddings / np.linalg.norm(external_embeddings, axis=1, keepdims=True)

    # Compute cosine similarity in batches to avoid memory overflow
    batch_size = 256
    best_matches = []
    best_scores = []

    for i in tqdm(range(0, len(prescription_norm), batch_size), desc = 'Finding best match'):
        batch = prescription_norm[i:i+batch_size]
        scores = batch @ external_norm.T  
        
        best_idx = np.argmax(scores, axis=1)
        best_score = scores[np.arange(len(batch)), best_idx]
        
        best_matches.extend(external['name'].iloc[best_idx].values)
        best_scores.extend(best_score)
        
        if i % 1000 == 0:
            print(f"Processed {i}/{len(prescription_norm)}")

    prescription['match'] = best_matches
    prescription['match_similarity'] = best_scores

    return prescription

prescription = drug_matching(prescription, external)
prescription = prescription.merge(external.drop('drug_embedding', axis = 1).rename(columns = {'name': 'match'}), on = 'match', how = 'left')

prescription = prescription[prescription['match_similarity'] > 0.65].sort_values('match_similarity')
prescription = prescription.drop(['drug_embedding', 'match_similarity'], axis = 1)

### Now map back to Prescription for Edge mapping
merged = prescription_base.merge(prescription, on = 'drug', how = 'right')

# Deal with range values
def parse_dose(val):
    if isinstance(val, str) and '-' in val:
        parts = val.split('-')
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except:
            return np.nan
    try:
        return float(val)
    except:
        return np.nan

merged['dose_value'] = merged['dose_value'].apply(parse_dose)

#### Import into Neo4j
start_idx = 0
BATCH_SIZE = 500

query = """
    UNWIND $rows AS row

    MATCH (a:Admission:Test:MIMIC {id: row.id})
    MATCH (d:Drug:External:Test {id:row.drug_id})

    MERGE (a)-[:PRESCRIBED]->(d)
    """

# Process in batches
for i in tqdm(range(start_idx, len(merged), BATCH_SIZE), desc="Batch processing"):

    batch = merged.iloc[i:i+BATCH_SIZE]

    rows = []
    for _, row in batch.iterrows():
        rows.append({
            "id": row["hadm_id"],
            'drug_id':row['id']
        })

    dml_ddl_neo4j(
        query,
        progress=False,
        rows=rows
    )