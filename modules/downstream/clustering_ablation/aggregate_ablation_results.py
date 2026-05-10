import os
import re
import pandas as pd
from pathlib import Path

def parse_metrics(file_path):
    """Parses the last epoch metrics from a metrics.txt file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split by epochs and take the last full one
    epochs = content.split('Epoch')
    if len(epochs) < 2:
        return None
    
    last_epoch = epochs[-1]
    
    metrics = {}
    
    # Extract Mean AUROC
    mean_match = re.search(r"Val mean AUROC: ([\d.]+)", last_epoch)
    if mean_match:
        metrics['Mean AUROC'] = float(mean_match.group(1))
        
    # Extract Individual Task AUROCs
    tasks = {
        'MORTALITY': 'Mortality',
        'LOS > 7 DAYS': 'LOS',
        'READMISSION': 'Readmission',
        'PROGRESSION': 'Progression',
        'DRUG RECOMMENDATION': 'Drug Rec'
    }
    
    for marker, task_name in tasks.items():
        # Look for the AUROC line under each task marker
        task_block = last_epoch.split(f"[{marker}]")
        if len(task_block) > 1:
            auroc_match = re.search(r"AUROC: ([\d.]+)", task_block[1])
            if auroc_match:
                metrics[task_name] = float(auroc_match.group(1))
                
    return metrics

def main():
    checkpoint_dir = Path('checkpoints')
    results = []
    
    print(f"Scanning {checkpoint_dir} for results...")
    
    for folder in sorted(checkpoint_dir.iterdir()):
        if not folder.is_dir():
            continue
            
        metrics_file = folder / 'metrics.txt'
        if metrics_file.exists():
            data = parse_metrics(metrics_file)
            if data:
                data['Folder'] = folder.name
                # Identify ablation type from folder name
                if 'ablation_' in folder.name:
                    data['Type'] = folder.name.split('ablation_')[1].split('_transformer')[0]
                else:
                    data['Type'] = 'Base/Standard'
                    
                results.append(data)
    
    if not results:
        print("No results found!")
        return
        
    df = pd.DataFrame(results)
    # Reorder columns for readability
    cols = ['Type', 'Mean AUROC', 'Mortality', 'LOS', 'Readmission', 'Progression', 'Drug Rec', 'Folder']
    df = df[cols]
    
    output_path = 'ablation_summary_table.csv'
    df.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("ABLATION RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    print(f"\nFull table saved to: {output_path}")

if __name__ == "__main__":
    main()
