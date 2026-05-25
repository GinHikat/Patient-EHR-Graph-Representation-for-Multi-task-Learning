import os, sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import argparse

# Import model components
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from EHR_model import EHRDataset, ehr_collate_fn, EHRModel, EHRTransformer, EHRTransformerBase

def parse_args():
    parser = argparse.ArgumentParser(description='Generate High-Resolution t-SNE Plot')
    parser.add_argument('--model_path', type=str, required=True, help='Path to best_model.pt')
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'transformer', 'transformer_base'], help='Model type')
    parser.add_argument('--level', type=str, default='patient', choices=['admission', 'patient'], help='Cluster level: admission or patient')
    parser.add_argument('--assignment_path', type=str, required=True, help='Path to cluster assignments CSV')
    parser.add_argument('--sample_size', type=int, default=3000, help='Number of points to sample for clean t-SNE')
    parser.add_argument('--perplexity', type=int, default=50, help='t-SNE perplexity (higher values group clusters more tightly)')
    parser.add_argument('--full', action='store_true', help='Use the full patient cohort without downsampling')
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
    elif args.model_type == 'lstm':
        model = EHRModel().to(device)
    else:
        model = EHRTransformer().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # 2. Extract Embeddings
    downstream_data_path = os.path.join(project_root, 'data', 'Timeline')
    TIMELINE_DIR = os.path.join(project_root, 'data', 'Timeline_new')
    if not os.path.exists(TIMELINE_DIR):
        TIMELINE_DIR = os.path.join(project_root, 'data', 'Timeline')
        
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

    # Cohort filtering (supports full extraction or sub-sampling):
    df_assign = pd.read_csv(args.assignment_path, dtype=str)
    join_key = 'patient_id' if args.level == 'patient' else 'adm_id'
    
    if not args.full:
        sample_ids = df_assign[join_key].sample(n=min(len(df_assign), int(args.sample_size * 1.2)), random_state=42).tolist()
        if args.level == 'patient':
            train_df = train_df[train_df['patient_id'].astype(str).isin(sample_ids)]
        else:
            train_df = train_df[train_df['id'].astype(str).isin(sample_ids)]
        print(f"Optimized: Filtered to {len(train_df)} matching admissions for fast t-SNE extraction...")
    else:
        print(f"Full run: Processing all {len(train_df)} admissions for full cohort t-SNE...")

    dataset = EHRDataset(
        admissions_df=train_df, timeline_dir=TIMELINE_DIR, admission_nodes=admission_nodes,
        diag_to_idx=diag_to_idx, drug_to_idx=drug_to_idx, patient_cache=patient_cache,
        admission_cache=admission_cache, max_len=None
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
    
    # 3. Aggregate if patient-level
    if args.level == 'patient':
        print("Aggregating embeddings per patient...")
        patient_data = full_df.groupby('pid')['repr'].apply(lambda x: np.mean(np.stack(x.values), axis=0))
        X = np.stack(patient_data.values)
        metadata_ids = patient_data.index.tolist()
        df_features = pd.DataFrame({'patient_id': metadata_ids})
    else:
        X = np.stack(full_df['repr'].values)
        df_features = full_df[['pid', 'adm_id']].copy()
        df_features = df_features.rename(columns={'pid': 'patient_id'})

    # 4. Load Assignments and Merge
    print(f"Loading cluster assignments from {args.assignment_path}...")
    df_assign = pd.read_csv(args.assignment_path, dtype=str)
    
    # Check join keys
    join_key = 'patient_id' if args.level == 'patient' else 'adm_id'
    df_features[join_key] = df_features[join_key].astype(str)
    df_assign[join_key] = df_assign[join_key].astype(str)
    
    df_merged = df_features.merge(df_assign, on=join_key)
    print(f"Merged features with assignments. Total matched records: {len(df_merged)}")

    # 5. Downsample for plotting clarity (bypassed if --full is set)
    if not args.full and len(df_merged) > args.sample_size:
        print(f"Sampling {args.sample_size} records for clean t-SNE visualization...")
        df_sampled = df_merged.sample(n=args.sample_size, random_state=42).copy()
        sampled_indices = df_sampled.index.values
        X_sampled = X[sampled_indices]
    else:
        print(f"Full cohort run: Plotting all {len(df_merged)} matches without downsampling...")
        df_sampled = df_merged.copy()
        X_sampled = X

    # 6. Run t-SNE
    print(f"Computing 2D t-SNE embeddings (perplexity={args.perplexity}) using all CPU cores...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity, n_jobs=-1, random_state=42)
    X_embedded = tsne.fit_transform(X_sampled)
    
    df_sampled['tsne_1'] = X_embedded[:, 0]
    df_sampled['tsne_2'] = X_embedded[:, 1]

    # 7. Map Taxonomy Labels beautifully
    taxonomy_labels = {
        '0': 'Cluster 0: Cardiometabolic Early-Symptoms',
        '1': 'Cluster 1: Psychiatric Crisis',
        '2': 'Cluster 2: End-stage Cardiometabolic',
        '3': 'Cluster 3: Cardiorenal Failure'
    }
    df_sampled['taxonomy'] = df_sampled['cluster'].astype(str).map(lambda x: taxonomy_labels.get(x, f"Cluster {x}"))
    
    # Sort order to match cluster numbering in legend
    df_sampled = df_sampled.sort_values('cluster')

    # 8. Plot t-SNE beautifully
    print("Generating high-resolution t-SNE scatter plot...")
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    
    # Custom rich, deep, high-contrast, professional publication-grade palette
    custom_palette = {
        'Cluster 0: Cardiometabolic Early-Symptoms': '#2B5C8F',    # Deep Slate/Steel Blue
        'Cluster 1: Psychiatric Crisis': '#9E2A2B',               # Deep Burgundy/Crimson Red
        'Cluster 2: End-stage Cardiometabolic': '#D98A00',         # Deep Golden Amber (High contrast with blue)
        'Cluster 3: Cardiorenal Failure': '#6A3D9A'               # Deep Royal Purple (Replaced Orange)
    }
    
    # Fallback to standard palette if K != 4
    all_taxonomies = df_sampled['taxonomy'].unique()
    palette = {tax: custom_palette.get(tax, sns.color_palette("Set2")[i % 8]) for i, tax in enumerate(sorted(all_taxonomies))}

    scatter = sns.scatterplot(
        data=df_sampled,
        x='tsne_1',
        y='tsne_2',
        hue='taxonomy',
        palette=palette,
        alpha=0.75,
        s=55,
        edgecolor='w',
        linewidth=0.3
    )

    # Styling labels and title
    plt.title('t-SNE Projection Of Patient-level Clustering', 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('t-SNE Dimension 1', fontsize=11, fontweight='bold', labelpad=8)
    plt.ylabel('t-SNE Dimension 2', fontsize=11, fontweight='bold', labelpad=8)
    
    # Clean grid and spines
    plt.grid(True, linestyle='--', alpha=0.3)
    sns.despine(top=True, right=True)
    
    # Legend styling
    plt.legend(
        title='Identified Patient Subgroups',
        title_fontsize='11',
        loc='best',
        frameon=True,
        shadow=True,
        fontsize='10'
    )

    plt.tight_layout()
    
    # Save output
    output_path = os.path.join(output_dir, f"tsne_{args.level}_k{len(df_sampled['cluster'].unique())}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Beautiful t-SNE plot generated and saved to: {output_path}")

if __name__ == '__main__':
    main()
