import os, sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.cluster import KMeans
import argparse

# Import model components
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from EHR_model import EHRDataset, ehr_collate_fn, EHRTransformer

def parse_args():
    parser = argparse.ArgumentParser(description='EHR Direct K-Means Clustering')
    parser.add_argument('--model_path', type=str, required=True, help='Path to best_model.pt')
    parser.add_argument('--k', type=int, default=10, help='Number of clusters')
    parser.add_argument('--batch_size', type=int, default=128)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup Output
    output_dir = os.path.dirname(args.model_path)
    print(f"Results will be saved to: {output_dir}")
    
    # Load Supervised Model
    print(f"Loading model from {args.model_path}...")
    model = EHRTransformer().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Data Loading
    downstream_data_path = os.path.join(project_root, 'data', 'Timeline')
    TIMELINE_DIR = os.path.join(project_root, 'data', 'Timelines')
    ADMISSION_NODES_PATH = os.path.join(downstream_data_path, 'admission_nodes.json')
    DIAG_VOCAB_PATH      = os.path.join(downstream_data_path, 'top200_diag_vocab.json')
    TRAIN_DF_PATH        = os.path.join(downstream_data_path, 'models', 'train_df.csv')
    PATIENT_CACHE_PATH   = os.path.join(downstream_data_path, 'setup', 'patient_cache.pt')
    ADMISSION_CACHE_PATH = os.path.join(downstream_data_path, 'setup', 'admission_cache.pt')
    DRUG_VOCAB_PATH      = os.path.join(downstream_data_path, 'top50_drug_vocab.json')

    print('Loading metadata...')
    train_df = pd.read_csv(TRAIN_DF_PATH, dtype={'id': str, 'patient_id': str})
    with open(ADMISSION_NODES_PATH) as f: admission_nodes = json.load(f)
    with open(DIAG_VOCAB_PATH) as f: diag_to_idx = json.load(f)
    with open(DRUG_VOCAB_PATH) as f: drug_to_idx = json.load(f)
    patient_cache = torch.load(PATIENT_CACHE_PATH, map_location='cpu')
    admission_cache = torch.load(ADMISSION_CACHE_PATH, map_location='cpu')

    dataset = EHRDataset(
        admissions_df=train_df, timeline_dir=TIMELINE_DIR, admission_nodes=admission_nodes,
        diag_to_idx=diag_to_idx, drug_to_idx=drug_to_idx, patient_cache=patient_cache,
        admission_cache=admission_cache, max_len=512
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ehr_collate_fn)

    # Extract Embeddings (Last Hidden State)
    embeddings = []
    metadata = []
    print("Extracting last hidden states...")
    with torch.no_grad():
        for batch in tqdm(loader):
            if batch is None: continue
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(batch_gpu)
            
            # Extract shared_repr (the last hidden state hybrid)
            reprs = out['shared_repr'].cpu().numpy()
            embeddings.append(reprs)
            
            for i in range(len(reprs)):
                metadata.append({
                    'pid': batch['pids'][i],
                    'adm_id': batch['adm_ids'][i]
                })

    X = np.concatenate(embeddings, axis=0)
    
    # K-Means Clustering
    print(f"Running K-Means (K={args.k})...")
    kmeans = KMeans(n_clusters=args.k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # Save Results
    results_df = pd.DataFrame(metadata)
    results_df['cluster'] = clusters
    
    output_path = os.path.join(output_dir, f'direct_clusters_k{args.k}.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Clustering complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
