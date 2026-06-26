import pandas as pd
import numpy as np
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
from modules.models import Models, model_dict
from modules.processing import *

data_dir = os.path.join(project_root, 'data', 'Note')

cleaned_data_dir = os.path.join(data_dir, 'cleaned')
training_path = os.path.join(cleaned_data_dir, 'train_set.csv')

train = pd.read_csv(training_path)
train['hadm_id'] = train['hadm_id'].astype(int)

procedure_train = train[['hadm_id', 'Technique', 'Examination', 'Indication', 'Impression', 'procedure']]

procedure_train['Examination'] = procedure_train['Examination'].str.title()

procedure_train['Indication'] = procedure_train['Indication'].apply(lambda x: str(x).replace('//', ','))

procedure_train['Combined_Note'] = (
    procedure_train['Examination'] + " - " + 
    procedure_train['Indication'] + " - " +
    procedure_train['Technique'] + " - " +
    procedure_train['Impression']
)

procedure_train = procedure_train.dropna(axis = 0)

procedure_train = procedure_train.drop_duplicates()

# Start Filtering
tech_counts = procedure_train.groupby(['procedure', 'Technique']).size().reset_index(name='t_count')
best_tech_idx = tech_counts.groupby('procedure')['t_count'].idxmax()
canonical_tech = tech_counts.loc[best_tech_idx][['procedure', 'Technique']]

# Filter the raw data to ONLY include rows that match those winning Techniques
filtered_df = procedure_train.merge(canonical_tech, on=['procedure', 'Technique'])

# Now, find the most common Examination within that protected subset
exam_counts = filtered_df.groupby(['procedure', 'Technique', 'Examination']).size().reset_index(name='final_count')
best_exam_idx = exam_counts.groupby('procedure')['final_count'].idxmax()
final_canonical_pairs = exam_counts.loc[best_exam_idx].sort_values('final_count', ascending=False)

evaluator_df = final_canonical_pairs[final_canonical_pairs['final_count'] >= 5].reset_index(drop=True)

proven_tuples = evaluator_df[['procedure', 'Technique', 'Examination']]

matched_original_rows = procedure_train.merge(
    proven_tuples, 
    on=['procedure', 'Technique', 'Examination'], 
    how='inner'
)

valid_hadm_ids = matched_original_rows['hadm_id'].unique()
final_eval_dataset = procedure_train[procedure_train['hadm_id'].isin(valid_hadm_ids)].copy()
proven_tuples = evaluator_df[['procedure', 'Technique', 'Examination']].copy()
proven_tuples['is_golden'] = 1

final_eval_dataset = final_eval_dataset.merge(
    proven_tuples, 
    on=['procedure', 'Technique', 'Examination'], 
    how='left'
)

# Fill the blank non-matches with 0 (These are the noisy Cartesian rows!)
final_eval_dataset['is_golden'] = final_eval_dataset['is_golden'].fillna(0).astype(int)

# Test Subset
all_eval_hadms = final_eval_dataset['hadm_id'].unique()
sample_hadms = pd.Series(all_eval_hadms).sample(1000, random_state=42).tolist()
final_eval_dataset = final_eval_dataset[final_eval_dataset['hadm_id'].isin(sample_hadms)].copy()

# Start testing
final_eval_dataset['Combined_Note'] = final_eval_dataset['Combined_Note'].astype(str)
final_eval_dataset['procedure'] = final_eval_dataset['procedure'].astype(str)

def calculate_full_pipeline(model_idx, name):
    
    print(f"\n[{name}] Loading model...")
    embedder = Models(model_idx)
    
    notes_list = final_eval_dataset['Combined_Note'].tolist()
    procs_list = final_eval_dataset['procedure'].tolist()
    
    v1_chunks = []
    v2_chunks = []
    chunk_size = 256  
    
    # Encode Combined Notes with Progress Bar
    print(f"[{name}] Encoding Combined Notes:")
    for i in tqdm(range(0, len(notes_list), chunk_size)):
        chunk = notes_list[i : i + chunk_size]
        v1_chunks.append(embedder.encode_text(chunk, batch_size=256))
    v1 = np.vstack(v1_chunks)
    
    # Encode Procedures with Progress Bar
    print(f"[{name}] Encoding Procedures:")
    for i in tqdm(range(0, len(procs_list), chunk_size)):
        chunk = procs_list[i : i + chunk_size]
        v2_chunks.append(embedder.encode_text(chunk, batch_size=256))
    v2 = np.vstack(v2_chunks)
    
    del embedder
    gc.collect()
    torch.cuda.empty_cache()
    # Calculate 1-to-1 Cosine Similarity for each row
    dots = (v1 * v2).sum(axis=1)
    norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    
    sim_col_name = f'sim_{name}'
    final_eval_dataset[sim_col_name] = np.divide(dots, norms, out=np.zeros_like(dots), where=norms!=0)
    print(f"[{name}] Similarity calculation complete! Saved to column: {sim_col_name}")
    # Rank the scores descending WITHIN each hospital admission
    final_eval_dataset['intra_admission_rank'] = final_eval_dataset.groupby('hadm_id')[sim_col_name].rank(method='first', ascending=False)
    
    # Isolate only the true Golden Standard of Care pairings
    golden_rows = final_eval_dataset[final_eval_dataset['is_golden'] == 1]
    
    total_golden_pairs = len(golden_rows)
    if total_golden_pairs == 0:
        print("Error: No golden pairs found. Make sure 'is_golden' is calculated correctly!")
        return
        
    recall_at_1 = (golden_rows['intra_admission_rank'] == 1).sum() / total_golden_pairs * 100
    recall_at_5 = (golden_rows['intra_admission_rank'] <= 5).sum() / total_golden_pairs * 100
    mrr = (1.0 / golden_rows['intra_admission_rank']).mean()
    
    print("\n[ Final Metrics ]")
    print(f"Total Golden Pairs Evaluated: {total_golden_pairs}")
    print(f"Recall@1: {recall_at_1:.1f}%")
    print(f"Recall@5: {recall_at_5:.1f}%")
    print(f"MRR:      {mrr:.4f}")
    
calculate_full_pipeline(1, "BGE-Large")
calculate_full_pipeline(2, "PubMedBERT")
calculate_full_pipeline(3, "MedCPT-Query-Encoder")
calculate_full_pipeline(4, "NeuML-PubMedBert")