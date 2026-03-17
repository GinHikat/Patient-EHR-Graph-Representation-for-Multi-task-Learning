import pandas as pd 
import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from shared_functions.global_functions import *

# Drugbank

df = pd.read_csv(os.path.join(drugbank_path, 'full.csv'))

def safe_eval(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return []
    return x

cols = ['atc_codes', 'interactions']
for col in cols:
    df[col] = df[col].apply(safe_eval)

df['len'] = df['interactions'].apply(lambda x: len(x))

CHECKPOINT_FILE = "checkpoint.txt"
BATCH_SIZE = 50

start_idx = 0
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r") as f:
        content = f.read().strip()
        if content:
            start_idx = int(content)

print(f"Resuming from index: {start_idx}")

query = """
UNWIND $rows AS row

MERGE (d:Drug:DB:Test {id: row.drugbank_id})
SET d.name = row.name,
    d.description = row.description

WITH d, row

UNWIND row.interactions AS inter_id
MERGE (d2:Drug:DB:Test {id: inter_id})
MERGE (d)-[:INTERACTS_WITH]->(d2)
"""

# Process in batches
for i in tqdm(range(start_idx, len(df), BATCH_SIZE), desc="Batch processing"):

    batch = df.iloc[i:i+BATCH_SIZE]

    rows = []
    for _, row in batch.iterrows():
        rows.append({
            "drugbank_id": row["drugbank_id"],
            "name": row["name"],
            "description": row["description"] if pd.notna(row["description"]) else None,
            "atc_codes": row["atc_codes"] if row["atc_codes"] else [],
            "interactions": row["interactions"] if row["interactions"] else []
        })

    dml_ddl_neo4j(
        query,
        progress=False,
        rows=rows
    )

    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(i + BATCH_SIZE))

# Add with Snap
snap = pd.read_csv(os.path.join(snap_path, 'snap.tsv'), sep='\t')
snap.columns = ['id1', 'id2']

query = """
UNWIND $rows AS row

MATCH (d1:Drug:DB:Test {id: row.id1})
MATCH (d2:Drug:DB:Test {id: row.id2})

WITH d1, d2
WHERE d1.id < d2.id   // prevents duplicate reverse edges

MERGE (d1)-[:INTERACTS_WITH]-(d2)
"""

start_idx = 0
BATCH_SIZE = 1000

for i in tqdm(range(start_idx, len(snap), BATCH_SIZE), desc="Batch processing"):

    batch = snap.iloc[i:i+BATCH_SIZE]

    rows = []
    for _, row in batch.iterrows():
        rows.append({
            "id1": row["id1"],
            "id2": row["id2"]
        })

    dml_ddl_neo4j(
        query,
        progress=False,
        rows=rows
    )