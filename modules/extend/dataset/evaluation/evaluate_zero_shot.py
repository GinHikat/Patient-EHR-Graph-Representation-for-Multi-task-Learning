import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys, os

# Add Thesis root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from modules.models.models import EmbeddingModels
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score

def batch_sim(df, embedder, chunk_size=500):
    df = df.copy()
    all_scores = []
    
    # Iterate in chunks to maintain GPU batching efficiency
    for i in tqdm(range(0, len(df), chunk_size), desc='Calculating sim score'):
        chunk = df.iloc[i : i + chunk_size]
        
        # Batch encode the query and matching term
        query_emb = embedder.encode_text(chunk['query_term'].tolist())
        match_emb = embedder.encode_text(chunk['matching_term'].tolist())
        
        # Vectorized calculation for the chunk
        q_emb = normalize(query_emb)
        m_emb = normalize(match_emb)
        chunk_scores = np.sum(q_emb * m_emb, axis=1)
        
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
        hue='label_name',
        bins=50, 
        kde=True, 
        element="step", 
        alpha=0.5,
        common_norm=False
    )

    plt.title(f'Similarity Score Distributions for {name}', fontsize=12)
    plt.tight_layout()
    
    # Save the plot securely
    safe_name = name.replace('/', '_').replace('\\', '_')
    plot_path = os.path.join(os.path.dirname(__file__), f"sim_dist_{safe_name}.png")
    plt.savefig(plot_path)
    print(f"Histogram saved to: {plot_path}")
    plt.close()

def evaluate_threshold(df, name):
    """
    Calculates metrics across multiple thresholds and returns the best F1.
    Also calculates mAP.
    """
    y_true = df['label']
    probs = df['sim_score']
    
    # Calculate Mean Average Precision (mAP)
    map_score = average_precision_score(y_true, probs)
    
    best_f1 = 0
    best_metrics = {}
    
    # Search for optimal threshold
    for t in np.arange(0.1, 1.0, 0.05):
        y_pred = (probs >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {
                'model': name,
                'best_threshold': round(t, 2),
                'accuracy': round(accuracy_score(y_true, y_pred), 4),
                'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
                'recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
                'f1': round(f1, 4),
                'mAP': round(map_score, 4)
            }
            
    plot_sim(df, name)
    return pd.DataFrame([best_metrics])

def main():
    dataset_path = r"data\viettel\combine\icd_pairwise_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return
        
    benchmark_df = pd.read_csv(dataset_path)
    print(f"Loaded evaluation dataset with {len(benchmark_df)} pairs.")
    
    # We test multiple highly relevant zero-shot models
    models_to_test = [
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",       # English SapBERT (Baseline to prove language gap)
        "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR", # Multilingual SapBERT (The Gold Standard)
        "democratized-nlp/vihealthbert",                       # Vietnamese Medical BERT
        "vinai/phobert-base",                                  # General Vietnamese BERT
        "BAAI/bge-m3",                                         # State of the art general multilingual embedding
        "AITeamVN/Vietnamese_Embedding",                       # SOTA Vietnamese embedding (BGE-M3 fine-tuned)
        "bkai-foundation-models/vietnamese-bi-encoder",        # Top Vietnamese semantic search encoder
        "keepitreal/vietnamese-sbert"                          # Vietnamese SBERT
    ]
    
    all_results = []
    
    for model_name in models_to_test:
        print(f"\n{'='*70}\nEvaluating Model: {model_name}\n{'='*70}")
        try:
            # Initialize using your custom EmbeddingModels wrapper
            embedder = EmbeddingModels(model_choice=model_name)
            
            # Calculate similarities
            df_scored = batch_sim(benchmark_df, embedder, chunk_size=500)
            
            # Evaluate metrics and plot
            short_name = model_name.split('/')[-1]
            metrics_df = evaluate_threshold(df_scored, short_name)
            
            print("\nResults:")
            print(metrics_df.to_string(index=False))
            all_results.append(metrics_df)
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        print("\n\nFINAL ZERO-SHOT COMPARISON:")
        print(final_df.to_string(index=False))
        
        output_csv = r"data\viettel\combine\zero_shot_evaluation_results.csv"
        final_df.to_csv(output_csv, index=False)
        print(f"\nSaved final metrics to: {output_csv}")

if __name__ == "__main__":
    main()
