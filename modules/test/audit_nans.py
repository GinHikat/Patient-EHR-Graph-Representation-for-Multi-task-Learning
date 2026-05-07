import numpy as np
import os
import json
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def audit_file(file_path):
    try:
        data = np.load(file_path)
        if not np.isnan(data).any():
            return None
        
        total_elements = data.size
        nan_count = np.isnan(data).sum()
        
        # Check for full rows of NaNs
        row_nans = np.isnan(data).all(axis=1)
        full_nan_rows = np.sum(row_nans)
        
        return {
            'file': os.path.basename(file_path),
            'total_elements': total_elements,
            'nan_count': int(nan_count),
            'nan_ratio': float(nan_count / total_elements),
            'full_nan_rows': int(full_nan_rows),
            'total_rows': data.shape[0]
        }
    except Exception as e:
        return {'file': os.path.basename(file_path), 'error': str(e)}

def main():
    timeline_dir = 'data/Timelines'
    files = [os.path.join(timeline_dir, f) for f in os.listdir(timeline_dir) if f.endswith('_emb.npy')]
    print(f"Starting audit of {len(files):,} files...")

    # Use multiprocessing for speed
    with Pool(cpu_count()) as p:
        results = list(tqdm(p.imap(audit_file, files), total=len(files), desc="Auditing"))

    # Filter and summarize
    nan_results = [r for r in results if r is not None and 'error' not in r]
    errors = [r for r in results if r is not None and 'error' in r]
    
    summary = {
        'total_files_checked': len(files),
        'files_with_nans': len(nan_results),
        'files_with_errors': len(errors),
        'nan_file_ratio': len(nan_results) / len(files) if files else 0
    }

    print("\n--- Audit Summary ---")
    print(json.dumps(summary, indent=2))

    if nan_results:
        # Calculate deeper stats
        total_rows_affected = sum(r['full_nan_rows'] for r in nan_results)
        print(f"Total full-NaN rows found: {total_rows_affected:,}")
        
        # Save detailed report
        with open('nan_audit_report.json', 'w') as f:
            json.dump(nan_results, f, indent=2)
        print("\nDetailed report saved to 'nan_audit_report.json'")

if __name__ == "__main__":
    main()
