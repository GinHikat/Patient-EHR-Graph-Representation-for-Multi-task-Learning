import pandas as pd
import numpy as np
from tqdm import tqdm
import os, sys
from dotenv import load_dotenv
from langdetect import detect, LangDetectException

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

load_dotenv(os.path.join(project_root, '.env'))

from enrichment import DatabaseExtract

# Instantiate the parser
uml_parser = DatabaseExtract()

base_data_dir = os.path.join(project_root, 'data')
edge_dir = os.path.join(base_data_dir, 'edges.csv')
node_dir = os.path.join(base_data_dir, 'nodes.csv')

nodes = pd.read_csv(node_dir)
nodes = nodes[nodes['description'].isna()]
nodes = nodes[~nodes['uml_id'].isna()]

CHECKPOINT_FILE = 'nodes_checkpoint.csv'
CHECKPOINT_IDX_FILE = 'checkpoint_idx.txt'

# Resume from checkpoint
if os.path.exists(CHECKPOINT_IDX_FILE) and os.path.exists(CHECKPOINT_FILE):
    nodes = pd.read_csv(CHECKPOINT_FILE)
    with open(CHECKPOINT_IDX_FILE, 'r') as f:
        start_idx = int(f.read().strip()) + 1
    print(f"Resuming from index {start_idx}")
else:
    nodes['description'] = None
    start_idx = 0

for i in tqdm(range(start_idx, len(nodes)), desc='Retrieving related description'):
    try:
        result = uml_parser.fetch_umls_description(nodes['uml_id'].iloc[i])
        nci_text = next((item['text'] for item in result['definitions'] if item['source'] == 'NCI'), None)
        
        if nci_text is None:
            nci_text = result['definitions'][0]['text'] if result['definitions'] else ''
    
    except Exception as e:
        print(f"\nError at index {i}, uml_id={nodes['uml_id'].iloc[i]}: {e}")
        nci_text = ''

    nodes.loc[nodes.index[i], 'description'] = nci_text

    nodes.to_csv(CHECKPOINT_FILE, index=False)
    with open(CHECKPOINT_IDX_FILE, 'w') as f:
        f.write(str(i))

# Keep only English description

def safe_detect(text):
    try:
        text = str(text).strip()
        if len(text) < 3:  
            return None
        return detect(text)
    except LangDetectException:
        return None

df = pd.read_csv('nodes_checkpoint.csv')

df['lang'] = df['description'].apply(safe_detect)

for i in range(len(df)):
    if df['lang'].iloc[i] == 'cs':
        df.loc[i, 'description'] = None

df = df[df['lang'] == 'en']

# Update to the Graph

query = """
    UNWIND $rows AS row
    MATCH (drug:Test {id: row.id})
    SET drug.description = row.description
    """

rows = [
    {"id": row["id"], "description": row["description"]}
    for _, row in df.iterrows()
]

BATCH_SIZE = 500
for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Processing"):
    dml_ddl_neo4j(query, progress=False, rows=rows[i:i+BATCH_SIZE])