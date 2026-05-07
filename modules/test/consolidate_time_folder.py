import os
import torch
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
data_dir = os.getenv('DATA_DIR')
TIMELINE_DIR = Path(os.path.join(data_dir, 'Timelines'))
OUTPUT_FILE = 'patient_timelines.pt'

def consolidate():
    '''
    Compress folder of a lot of small files into 1 .pt file for uploading
    '''
    # Find all unique patient IDs by looking at the files
    pids = set(f.name.split('_')[0] for f in TIMELINE_DIR.glob('*_emb.npy'))
    print(f"Found {len(pids)} patients to consolidate.")

    consolidated_data = {}

    for pid in tqdm(pids, desc="Packing data"):
        try:
            # Load the three components
            emb = np.load(TIMELINE_DIR / f'{pid}_emb.npy')
            dt  = np.load(TIMELINE_DIR / f'{pid}_dt.npy')
            with open(TIMELINE_DIR / f'{pid}_meta.json', 'r') as f:
                meta = json.load(f)

            # Store as tensors for efficiency
            consolidated_data[int(pid)] = {
                'emb': torch.from_numpy(emb),
                'dt': torch.from_numpy(dt),
                'meta': meta
            }
        except Exception as e:
            print(f"Error packing patient {pid}: {e}")

    print(f"Saving to {OUTPUT_FILE}...")
    torch.save(consolidated_data, OUTPUT_FILE)
    print("Done!")

if __name__ == '__main__':
    consolidate()
