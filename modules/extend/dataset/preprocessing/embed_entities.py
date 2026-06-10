import os
import pandas as pd
import numpy as np
import pickle
from modules.models.models import EmbeddingModels

def main():
    csv_path = "data/viettel/vietnamese_ner/ground_truth_vn_entity.csv"
    output_pkl = "data/viettel/vietnamese_ner/mapped_entities_embedded.pkl"
    output_csv = "data/viettel/vietnamese_ner/mapped_entities_embedded.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find '{csv_path}'. Please run this from the project root.")
        return
        
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Clean columns: keep only "entity", "mapped_cui", "mapped_type", and "original_type"
    cols_to_keep = ["entity", "original_type", "mapped_cui", "mapped_type"]
    missing = [c for c in cols_to_keep if c not in df.columns]
    if missing:
        print(f"Error: Missing columns {missing} in the CSV.")
        return
        
    df = df[cols_to_keep]
    print(f"DataFrame filtered to columns: {list(df.columns)}")
    print(f"Total rows: {len(df)}")
    
    # Clean up empty entities just in case
    df = df.dropna(subset=['entity']).reset_index(drop=True)
    
    # 2. Embed each entity using the specified SapBERT model
    model_name = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
    print(f"\nLoading embedding model: {model_name}...")
    
    # Using your existing EmbeddingModels wrapper from modules.models.models
    embedder = EmbeddingModels(model_choice=model_name)
    
    entities = df['entity'].tolist()
    print(f"\nEmbedding {len(entities)} entities (this might take a minute)...")
    
    # encode_text returns a numpy array of shape (N, embedding_dim)
    embeddings = embedder.encode_text(entities, batch_size=256, show_progress=True)
    
    # 3. Save that embedding in the column "embedding"
    # Convert numpy array matrix to a list of arrays so it fits cleanly in a Pandas Column
    df['embedding'] = list(embeddings)
    
    print(f"\nSaving results...")
    # Pickle is the safest format for DataFrames containing numpy arrays
    df.to_pickle(output_pkl)
    print(f"Saved Pickle (Recommended for Python loading): {output_pkl}")
    
    # We also save a CSV version, converting the arrays to native Python lists
    # so they serialize properly to strings in the CSV
    df_csv = df.copy()
    df_csv['embedding'] = df_csv['embedding'].apply(lambda x: list(x))
    df_csv.to_csv(output_csv, index=False)
    print(f"Saved CSV (For easy viewing): {output_csv}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
