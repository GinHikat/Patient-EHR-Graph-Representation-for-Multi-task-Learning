import pandas as pd
import numpy as np
import duckdb
import re
import torch
from tqdm import tqdm
import gc

import warnings
import logging
import transformers
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from underthesea import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

from shared_functions.gg_sheet_drive import *
from modules.models import EmbeddingModels, model_dict
from modules.processing import *

data_dir = os.path.join(project_root, 'data', 'Note')

cleaned_data_dir = os.path.join(data_dir, 'cleaned')
training_path = os.path.join(cleaned_data_dir, 'train_set.csv')

# Use DuckDB to load and filter NA rows efficiently
target_cols = ['hadm_id', 'Examination', 'Indication', 'Technique', 'Comparison', 'Findings', 'Impression', 'procedure']
where_clause = " AND ".join([f'"{col}" IS NOT NULL' for col in target_cols])
query = f"""
    SELECT {', '.join([f'"{c}"' for c in target_cols])} 
    FROM read_csv_auto('{training_path}') 
    WHERE {where_clause}
"""
train = duckdb.query(query).df()
train['hadm_id'] = train['hadm_id'].astype(int)

procedure_train = train.copy()

procedure_train['Examination'] = procedure_train['Examination'].str.title()

procedure_train['Indication'] = procedure_train['Indication'].apply(lambda x: str(x).replace('//', ','))

procedure_train['Combined_Note'] = (
    procedure_train['Examination'] + " - " + 
    procedure_train['Indication'] + " - " +
    procedure_train['Technique'] + " - " +
    procedure_train['Comparison'] + " - " + 
    procedure_train['Findings'] + " - " + 
    procedure_train['Impression']
)

print("Preparing full dataset...")
procedure_train['Combined_Note'] = procedure_train['Combined_Note'].astype(str)
procedure_train['procedure'] = procedure_train['procedure'].astype(str)

embedder = EmbeddingModels(4) 

notes_list = procedure_train['Combined_Note'].tolist()
procs_list = procedure_train['procedure'].tolist()

chunk_size = 256
v1_chunks = []
v2_chunks = []

for i in tqdm(range(0, len(notes_list), chunk_size)):
    chunk = notes_list[i : i + chunk_size]
    v1_chunks.append(embedder.encode_text(chunk, batch_size=256))

for i in tqdm(range(0, len(procs_list), chunk_size)):
    chunk = procs_list[i : i + chunk_size]
    v2_chunks.append(embedder.encode_text(chunk, batch_size=256))

v1 = np.vstack(v1_chunks)
v2 = np.vstack(v2_chunks)
del embedder; gc.collect(); torch.cuda.empty_cache()

dots = (v1 * v2).sum(axis=1)
norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)

procedure_train['sim_score_fulltext'] = np.divide(dots, norms, out=np.zeros_like(dots), where=norms!=0)

procedure_train.to_csv(os.path.join(cleaned_data_dir, 'procedure_train.csv'), index=False)