import pandas as pd
import numpy as np
import json
import pickle 
from pathlib import Path
from tqdm import tqdm

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from shared_functions.global_functions import *
from modules.models.models import EmbeddingModels

embedder  = EmbeddingModels(1)

SAPBERT_BATCH_SIZE = 256
CHECKPOINT_EVERY   = 5
CHECKPOINT_PATH    = 'sapbert_checkpoint.pkl'

data_dir = f"{project_root}/data"
all_nodes = pd.read_csv(f"{data_dir}/downstream/embedding/kg_nodes.csv")

def encode_nodes_with_checkpoint(df, embedder, batch_size=SAPBERT_BATCH_SIZE,
                                  checkpoint_path=CHECKPOINT_PATH):
    names = df['name'].fillna('unknown').tolist()
    n_batches = (len(names) + batch_size - 1) // batch_size

    #  Resume checkpoint 
    if Path(checkpoint_path).exists():
        with open(checkpoint_path, 'rb') as f:
            ckpt = pickle.load(f)
        embeddings  = ckpt['embeddings']
        start_batch = ckpt['next_batch']
        print(f'Resuming from batch {start_batch}/{n_batches} '
              f'({start_batch * batch_size}/{len(names)} nodes done)')
    else:
        embeddings  = []
        start_batch = 0
        print(f'Starting fresh — {len(names)} nodes, {n_batches} batches')

    #  Encode 
    for i in tqdm(range(start_batch, n_batches), desc='SapBERT encoding',
                  initial=start_batch, total=n_batches):
        batch = names[i * batch_size:(i + 1) * batch_size]
        emb   = embedder.encode_text(batch)
        embeddings.append(emb)

        # Save checkpoint every N batches
        if (i + 1) % CHECKPOINT_EVERY == 0:
            tmp = checkpoint_path + '.tmp'
            with open(tmp, 'wb') as f:
                pickle.dump({'embeddings': embeddings, 'next_batch': i + 1}, f)
            os.replace(tmp, checkpoint_path)

    result = np.vstack(embeddings).astype(np.float32)

    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        print('Checkpoint cleared.')

    return result

node_embeddings = encode_nodes_with_checkpoint(all_nodes, embedder)
print(f'Done. Shape: {node_embeddings.shape}')  # (N, 768)

# Save to file 
np.savez(f"{data_dir}/downstream/embedding/node_embeddings.npz", embeddings=node_embeddings)
print("Node embeddings saved to node_embeddings.npz")
