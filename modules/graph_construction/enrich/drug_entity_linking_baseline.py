import pandas as pd
import numpy as np
import dotenv
import duckdb
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
dotenv.load_dotenv()

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from modules.models.models import EmbeddingModels
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def batch_sim(df, chunk_size=1000):
    df = df.copy()
    all_scores = []
    
    # Iterate in large chunks to maintain GPU batching efficiency
    for i in tqdm(range(0, len(df), chunk_size), desc='Calculating sim score'):
        chunk = df.iloc[i : i + chunk_size]
        
        # Batch encode the names and aliases in this chunk
        name_emb = embedder.encode_text(chunk['name'].tolist())
        alias_emb = embedder.encode_text(chunk['drug_alias'].tolist())
        
        # Vectorized calculation for the chunk
        n_emb = normalize(name_emb)
        a_emb = normalize(alias_emb)
        chunk_scores = np.sum(n_emb * a_emb, axis=1)
        
        all_scores.append(chunk_scores)
        
    df['sim_score'] = np.concatenate(all_scores)
    return df

def plot_sim(df, name):
    plot_df = df.copy()
    plot_df['label_name'] = plot_df['label'].map({1: 'Positive (1)', 0: 'Negative (0)'})

    plt.figure(figsize=(8, 4))
    sns.histplot(
        data=plot_df, 
        x='sim_score', 
        hue='label_name', # Seaborn will now use the correct names
        bins=50, 
        kde=True, 
        element="step", 
        alpha=0.5,
        common_norm=False
    )

    plt.title(f'Corrected Similarity Score Distributions for model {name}', fontsize=15)
    plt.show()

def evaluate_threshold(df, threshold):
    """
    Calculates metrics by converting sim_score into binary predictions 
    based on the given threshold.
    """
    y_true = df['label']
    y_pred = (df['sim_score'] >= threshold).astype(int)
    
    metrics = {
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

    metrics = pd.DataFrame([metrics])

    plot_sim(df, '')
    
    print(metrics)

benchmark_df = pd.read_csv('entity_linking_df.csv')

embedder = EmbeddingModels(1)
df_1 = batch_sim(benchmark_df.sample(20000, random_state=42), chunk_size=500)

evaluate_threshold(df_1, 0.7)

