import pandas as pd
import numpy as np
import dotenv
import duckdb
import csv
from tqdm import tqdm
dotenv.load_dotenv()
import random
from rapidfuzz import process, fuzz # Efficient string matching

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from shared_functions.gg_sheet_drive import *

def benchmark_dataset():
    '''
       Benchmark dataset creation based on DrugBank alias
       - Positive samples: DrugBank alias pairs
       - Negative samples: Random and lexical negatives (each 50%)
    '''
    df = gs_to_df_pandas('check')

    pos_df = df[['name', 'drug_alias']].copy() # Your 35,981 rows
    all_drug_names = df['name'].unique() # The 5,705 unique drugs

    # Generate 50% Random Negatives (Easy)
    # Shuffle the 'name' column so each alias is paired with a random wrong name
    random_neg_df = pos_df.copy()
    random_neg_df['name'] = np.random.permutation(random_neg_df['name'].values)

    # Safety check: ensure we didn't accidentally shuffle it back to its original name
    mask = random_neg_df['name'] == pos_df['name']
    random_neg_df.loc[mask, 'name'] = np.random.choice(all_drug_names, size=mask.sum())
    random_neg_df = random_neg_df.sample(n=len(pos_df)//2)
    random_neg_df['label'] = 0

    # Generate 50% Lexical Negatives (Hard)
    hard_negatives = []
    targets_for_hard = pos_df.sample(n=len(pos_df)//2)

    print("Generating Hard Negatives (this may take a minute)...")
    for idx, row in targets_for_hard.iterrows():
        alias = row['drug_alias']
        correct_name = row['name']
        
        # Use fuzzy matching to find the top 2 names similar to the alias
        # One will be the correct name, the other will be our "Hard Negative"
        best_matches = process.extract(alias, all_drug_names, limit=3, scorer=fuzz.WRatio)
        
        # Pick the first match that ISN'T the correct name
        hard_name = None
        for match_name, score, _ in best_matches:
            if match_name != correct_name:
                hard_name = match_name
                break
        
        if hard_name:
            hard_negatives.append({'name': hard_name, 'drug_alias': alias, 'label': 0})

    hard_neg_df = pd.DataFrame(hard_negatives)

    # Final Assemblage
    benchmark_df = pd.concat([
        pos_df.assign(label=1),
        random_neg_df,
        hard_neg_df
    ]).sample(frac=1).reset_index(drop=True)

    print(f"Total dataset size: {len(benchmark_df)}")
    print(benchmark_df['label'].value_counts())
    
    return benchmark_df
