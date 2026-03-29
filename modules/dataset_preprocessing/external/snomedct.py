import pandas as pd
import numpy as np
from pathlib import Path

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

snomed_root = Path(snomedct_path)

output_root = Path(os.path.join(snomedct_path, 'csv'))
output_root.mkdir(exist_ok=True)

# Find all RF2 txt files
txt_files = list(snomed_root.rglob("*.txt"))

print(f"Found {len(txt_files)} files")

for txt_file in txt_files:
    try:
        # Read RF2 file (tab separated)
        df = pd.read_csv(txt_file, sep="\t", dtype=str)

        # Preserve folder structure
        relative_path = txt_file.relative_to(snomed_root)
        csv_path = output_root / relative_path.with_suffix(".csv")

        csv_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(csv_path, index=False)

        print(f"Converted: {txt_file.name}")

    except Exception as e:
        print(f"Error processing {txt_file}: {e}")

print("Conversion completed.")