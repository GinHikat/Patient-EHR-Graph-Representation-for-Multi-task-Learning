import torch
import numpy as np
import json
import os
from pathlib import Path

# Load the consolidated file
data = torch.load('/home/hngoc/gin/Clinical-Note-Extraction/data/Timeline/patient_timelines.pt')
output_dir = Path('data/Timelines')
output_dir.mkdir(exist_ok=True)

# Loop through and save individual files
for pid, content in data.items():
    # Save embeddings
    np.save(output_dir / f'{pid}_emb.npy', content['emb'].numpy())
    # Save delta times
    np.save(output_dir / f'{pid}_dt.npy', content['dt'].numpy())
    # Save metadata
    with open(output_dir / f'{pid}_meta.json', 'w') as f:
        json.dump(content['meta'], f)

print(f"Successfully extracted data for {len(data)} patients.")
