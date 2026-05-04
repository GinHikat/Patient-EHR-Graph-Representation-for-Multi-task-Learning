import os
import sys
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATConv
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
BASE_DIR = os.path.join(PROJECT_ROOT, 'gin', 'Clinical-Note-Extraction', 'data')

# Paths
SAPBERT_EMB_PATH = os.path.join(BASE_DIR, 'kg_nodes_embed.npy')
EDGES_CSV_PATH   = os.path.join(BASE_DIR, 'kg_edges.csv')
NODES_CSV_PATH   = os.path.join(BASE_DIR, 'kg_nodes.csv')
SAVE_PATH        = 'kg_gat_pretrained.pt'

# Model Hyperparameters
IN_DIM      = 768
HIDDEN_DIM  = 256
OUT_DIM     = 64
HEADS       = 4
DROPOUT     = 0.1
N_RELATIONS = 6

# Training Hyperparameters
EPOCHS     = 50
LR         = 1e-3
MARGIN     = 0.5
BATCH_SIZE = 1024

class KG_GAT(nn.Module):
    def __init__(self, in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM,
                 heads=HEADS, dropout=DROPOUT, n_relations=N_RELATIONS):
        super().__init__()
        self.edge_type_emb = nn.Embedding(n_relations, in_dim)
        self.conv1  = GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.conv2  = GATConv(hidden_dim, out_dim, heads=1, dropout=dropout)
        self.norm1  = nn.LayerNorm(hidden_dim)
        self.norm2  = nn.LayerNorm(out_dim)
        self.drop   = nn.Dropout(dropout)
        self.act    = nn.ELU()

    def forward(self, x, edge_index, edge_type):
        edge_feat = self.edge_type_emb(edge_type)
        x_mod = x.clone()
        x_mod[edge_index[0]] = x[edge_index[0]] + edge_feat

        h = self.conv1(x_mod, edge_index)
        h = self.norm1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        return h

class LinkPredDataset(Dataset):
    def __init__(self, pos_edges, neg_edges):
        self.pos = torch.tensor(pos_edges, dtype=torch.long)
        self.neg = torch.tensor(neg_edges, dtype=torch.long)

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, idx):
        return self.pos[idx], self.neg[idx]

def evaluate(model, x_full, edge_index_full, edge_type_full, eval_df, neg_idx_a, neg_idx_b):
    model.eval()
    with torch.no_grad():
        h = model(x_full, edge_index_full, edge_type_full).cpu().numpy()

    # 1. Mean similarity per relation type
    results = {}
    for rel in eval_df['relation'].unique():
        subset = eval_df[eval_df['relation'] == rel]
        sims = []
        for _, row in subset.iterrows():
            idx_a, idx_b = row['idx_pair']
            sim = cosine_similarity(h[idx_a:idx_a+1], h[idx_b:idx_b+1])[0][0]
            sims.append(sim)
        results[rel] = np.mean(sims)

    # 2. AUROC (Real vs Random)
    pos_sims, neg_sims = [], []
    for _, row in eval_df.iterrows():
        idx_a, idx_b = row['idx_pair']
        pos_sims.append(cosine_similarity(h[idx_a:idx_a+1], h[idx_b:idx_b+1])[0][0])
    for idx_a, idx_b in zip(neg_idx_a, neg_idx_b):
        neg_sims.append(cosine_similarity(h[idx_a:idx_a+1], h[idx_b:idx_b+1])[0][0])

    labels = [1] * len(pos_sims) + [0] * len(neg_sims)
    scores = pos_sims + neg_sims
    auroc = roc_auc_score(labels, scores)

    # 3. Hits@10 (Accuracy among top candidates)
    hits = 0
    for _, row in eval_df.iterrows():
        idx_a, idx_b = row['idx_pair']
        sims = cosine_similarity(h[idx_a:idx_a+1], h)[0]
        sims[idx_a] = -1 # exclude self
        top10 = np.argsort(sims)[::-1][:10]
        if idx_b in top10:
            hits += 1
    hits_at_10 = hits / len(eval_df)

    return results, auroc, hits_at_10

def sample_negatives(pos_edges, n_nodes, full_pos_set):
    negs = []
    for src, dst, rel in pos_edges:
        while True:
            if np.random.rand() < 0.5:
                neg_src, neg_dst = np.random.randint(n_nodes), dst
            else:
                neg_src, neg_dst = src, np.random.randint(n_nodes)
            
            if (neg_src, neg_dst) not in full_pos_set:
                negs.append((neg_src, neg_dst, int(rel)))
                break
    return np.array(negs)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load Data
    print("Loading datasets...")
    all_nodes = pd.read_csv(NODES_CSV_PATH, low_memory=False)
    edges_df = pd.read_csv(EDGES_CSV_PATH)
    node_embeddings = np.load(SAPBERT_EMB_PATH)
    n_nodes = len(node_embeddings)

    # Train/Test Split
    print("Splitting edges into train/test sets...")
    edges_df = edges_df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(0.95 * len(edges_df))
    train_df = edges_df.iloc[:split_idx]
    test_df  = edges_df.iloc[split_idx:]

    # Prepare Evaluation Data
    eval_df = test_df.sample(n=min(1000, len(test_df)), random_state=42).copy()
    eval_df['idx_pair'] = eval_df.apply(lambda r: (int(r['src_idx']), int(r['dst_idx'])), axis=1)
    rel_map = {0: 'CAUSE', 1: 'TREAT', 2: 'EQUIVALENT_TO', 3: 'HAS_PHENOTYPE', 4: 'INTERACTS_WITH', 5: 'CHILD_OF'}
    eval_df['relation'] = eval_df['rel_idx'].map(lambda x: rel_map.get(x, str(x)))

    # Negative Sampling
    full_pos_set = set(map(tuple, edges_df[['src_idx', 'dst_idx']].values.tolist()))
    print("Sampling negatives for training...")
    pos_edges = train_df[['src_idx', 'dst_idx', 'rel_idx']].values
    neg_edges = sample_negatives(pos_edges, n_nodes, full_pos_set)

    # Dataloader
    dataset = LinkPredDataset(pos_edges, neg_edges)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model Setup
    model = KG_GAT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Graph Tensors (Restricted to Training edges)
    x_full = torch.tensor(node_embeddings, dtype=torch.float32).to(device)
    edge_index_train = torch.tensor(train_df[['src_idx','dst_idx']].values.T, dtype=torch.long).to(device)
    edge_type_train = torch.tensor(train_df['rel_idx'].values, dtype=torch.long).to(device)

    # Sample Eval Negatives
    print("Sampling evaluation negatives...")
    np.random.seed(42)
    eval_negs = []
    while len(eval_negs) < 1000:
        a, b = np.random.randint(0, n_nodes), np.random.randint(0, n_nodes)
        if (a, b) not in full_pos_set and a != b:
            eval_negs.append((a, b))
    neg_idx_a, neg_idx_b = [p[0] for p in eval_negs], [p[1] for p in eval_negs]

    # Training Loop
    print("Starting fine-tuning...")
    best_loss = float('inf')
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for pos_batch, neg_batch in tqdm(dataloader, desc=f'Epoch {epoch}/{EPOCHS}', leave=False):
            optimizer.zero_grad()
            h = model(x_full, edge_index_train, edge_type_train)
            
            pos_sim = F.cosine_similarity(h[pos_batch[:, 0]], h[pos_batch[:, 1]])
            neg_sim = F.cosine_similarity(h[neg_batch[:, 0]], h[neg_batch[:, 1]])
            
            loss = F.relu(MARGIN - pos_sim + neg_sim).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        if epoch % 5 == 0 or epoch == 1:
            rel_sims, auroc, hits10 = evaluate(model, x_full, edge_index_train, edge_type_train, eval_df, neg_idx_a, neg_idx_b)
            history.append({'epoch': epoch, 'loss': avg_loss, 'auroc': auroc, 'hits@10': hits10, **rel_sims})
            print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | AUROC={auroc:.3f} | Hits@10={hits10:.3f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), SAVE_PATH)

    # Summary
    print("\nTraining complete. History summary:")
    print(pd.DataFrame(history).round(3).to_string(index=False))

if __name__ == "__main__":
    main()