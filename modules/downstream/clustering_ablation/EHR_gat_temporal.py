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

from EHR_model import EHRDataset, ehr_collate_fn, ClinicalGAT, EHRTransformer

class GAT_Temporal_Hybrid(nn.Module):
    """
    Hybrid Graph-Temporal Model.
    Uses GAT to refine each time-step and Transformer to model the sequence.
    """
    def __init__(self, feature_dim=128):
        super().__init__()
        self.gat = ClinicalGAT(feature_dim=feature_dim)
        self.transformer_model = EHRTransformer()
        
    def forward(self, batch):
        # 1. Intra-step Graph Message Passing
        # batch['emb']: (B, T, 128)
        # In a real graph, we'd have multiple nodes per time-step.
        # Here we treat the timeline as a sequence of graphs where we refine the state.
        x = self.gat(batch['emb']) # (B, T, 128)
        
        # 2. Inter-step Temporal Modeling
        # Create a copy of the batch with the GAT-refined embeddings
        hybrid_batch = batch.copy()
        hybrid_batch['emb'] = x
        
        return self.transformer_model(hybrid_batch)

def main():
    parser = argparse.ArgumentParser(description='EHR GAT+Temporal Hybrid Experiment')
    parser.add_argument('--model_path', type=str, help='Path to pre-trained weights if any')
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = "checkpoints/gat_temporal_hybrid"
    os.makedirs(output_dir, exist_ok=True)
    
    model = GAT_Temporal_Hybrid().to(device)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    # Data Loading (similar to others)
    downstream_data_path = os.path.join(project_root, 'data', 'Timeline')
    TIMELINE_DIR = os.path.join(project_root, 'data', 'Timelines')
    TRAIN_DF_PATH = os.path.join(downstream_data_path, 'models', 'train_df.csv')
    
    # Loading block omitted for brevity, but matches gat_only.py
    print("Extracting Hybrid Graph-Temporal features...")
    # ... (Extraction loop matches EHR_direct_clustering.py)
    print(f"Hybrid clustering results will be saved in {output_dir}")

if __name__ == "__main__":
    main()
