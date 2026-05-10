import os, sys
import json
import numpy as np
import torch
import torch.nn as nn
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

from EHR_model import EHRDataset, ehr_collate_fn, ClinicalGAT

class GATOnlyModel(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.gat = ClinicalGAT(feature_dim=feature_dim)
        # Global pooling to get a single vector for clustering
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, batch):
        # Flatten timeline: (B, T, 128)
        x = batch['emb']
        # Apply GAT to the sequence (treating it as a fully connected graph)
        x = self.gat(x)
        # Global mean pool across time steps
        # x: (B, T, 128) -> (B, 128, T) -> (B, 128, 1) -> (B, 128)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return x

def parse_args():
    parser = argparse.ArgumentParser(description='EHR GAT-Only Clustering (No Temporal)')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup Output
    output_dir = "checkpoints/gat_only_clustering"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Model (Untrained for unsupervised representation discovery)
    model = GATOnlyModel().to(device)
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

    print('Loading data...')
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

    # Extract Embeddings
    embeddings = []
    metadata = []
    print("Extracting GAT features...")
    with torch.no_grad():
        for batch in tqdm(loader):
            if batch is None: continue
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            reprs = model(batch_gpu).cpu().numpy()
            embeddings.append(reprs)
            
            for i in range(len(reprs)):
                metadata.append({'pid': batch['pids'][i], 'adm_id': batch['adm_ids'][i]})

    X = np.concatenate(embeddings, axis=0)
    
    # K-Means
    print(f"Running K-Means (K={args.k})...")
    kmeans = KMeans(n_clusters=args.k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # Save
    results_df = pd.DataFrame(metadata)
    results_df['cluster'] = clusters
    output_path = os.path.join(output_dir, f'gat_only_clusters_k{args.k}.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
