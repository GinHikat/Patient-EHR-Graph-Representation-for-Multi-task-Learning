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

from EHR_model import EHRDataset, ehr_collate_fn, EHRModel, EHRTransformer, EHRTransformerBase

from sklearn.metrics import silhouette_score
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='EHR Direct Clustering Sweep')
    parser.add_argument('--model_path', type=str, required=True, help='Path to best_model.pt')
    parser.add_argument('--model_type', type=str, default='transformer', choices=['transformer', 'transformer_base'], help='Model type')
    parser.add_argument('--k_list', type=int, nargs='+', default=[2, 4, 6, 8, 10, 12], help='List of K values to test')
    parser.add_argument('--level', type=str, default='admission', choices=['admission', 'patient'], help='Cluster level: admission or patient')
    parser.add_argument('--batch_size', type=int, default=128)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = os.path.dirname(args.model_path)
    
    # 1. Load Model
    print(f"Loading {args.model_type} model from {args.model_path}...")
    if args.model_type == 'transformer_base':
        model = EHRTransformerBase().to(device)
    else:
        model = EHRTransformer().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # 2. Extract Embeddings
    downstream_data_path = os.path.join(project_root, 'data', 'Timeline')
    TIMELINE_DIR = os.path.join(project_root, 'data', 'Timelines')
    ADMISSION_NODES_PATH = os.path.join(downstream_data_path, 'admission_nodes.json')
    DIAG_VOCAB_PATH      = os.path.join(downstream_data_path, 'top200_diag_vocab.json')
    TRAIN_DF_PATH        = os.path.join(downstream_data_path, 'models', 'train_df.csv')
    PATIENT_CACHE_PATH   = os.path.join(downstream_data_path, 'setup', 'patient_cache.pt')
    ADMISSION_CACHE_PATH = os.path.join(downstream_data_path, 'setup', 'admission_cache.pt')
    DRUG_VOCAB_PATH      = os.path.join(downstream_data_path, 'top50_drug_vocab.json')

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

    all_data = []
    print(f"Extracting representations ({args.level} level)...")
    with torch.no_grad():
        for batch in tqdm(loader):
            if batch is None: continue
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(batch_gpu)
            reprs = out['shared_repr'].cpu().numpy()
            for i in range(len(reprs)):
                all_data.append({
                    'pid': batch['pids'][i],
                    'adm_id': batch['adm_ids'][i],
                    'repr': reprs[i]
                })

    full_df = pd.DataFrame(all_data)
    
    # Aggregate if patient-level
    if args.level == 'patient':
        print("Aggregating embeddings per patient...")
        patient_data = full_df.groupby('pid')['repr'].apply(lambda x: np.mean(np.stack(x.values), axis=0))
        X = np.stack(patient_data.values)
        metadata_pids = patient_data.index.tolist()
    else:
        X = np.stack(full_df['repr'].values)
        metadata = full_df[['pid', 'adm_id']].to_dict('records')

    # 3. Clustering Sweep
    sweep_stats = []
    for k in args.k_list:
        print(f"\n--- Running K-Means (K={k}, level={args.level}) ---")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        # Calculate silhouette
        if len(X) > 20000:
            sub_idx = np.random.choice(len(X), 20000, replace=False)
            sil = silhouette_score(X[sub_idx], clusters[sub_idx])
        else:
            sil = silhouette_score(X, clusters)
            
        print(f"  Inertia: {kmeans.inertia_:.2f}")
        print(f"  Silhouette: {sil:.4f}")
        sweep_stats.append({'K': k, 'Inertia': kmeans.inertia_, 'Silhouette': sil})
        
        # Save results
        if args.level == 'patient':
            results_df = pd.DataFrame({'patient_id': metadata_pids, 'cluster': clusters})
        else:
            results_df = pd.DataFrame(metadata)
            results_df['cluster'] = clusters
            
        results_df.to_csv(os.path.join(output_dir, f'clusters_{args.level}_k{k}.csv'), index=False)

    pd.DataFrame(sweep_stats).to_csv(os.path.join(output_dir, f'clustering_sweep_{args.level}_summary.csv'), index=False)
    print(f"\nSweep complete. Summary saved to {output_dir}")

if __name__ == "__main__":
    main()
