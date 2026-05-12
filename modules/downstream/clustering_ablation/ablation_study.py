import os, sys
import argparse
import subprocess
import datetime

# This script is a wrapper around EHR_training.py to run different ablation studies

def run_experiment(mode, epochs=10, batch_size=64, num_workers=4, model_type='transformer', patience=7, task='all', no_pos_weight=False):
    print(f"\n" + "="*50)
    print(f"RUNNING EXPERIMENT: {mode}")
    print(f"="*50)
    
    cmd = [
        "python3", "modules/downstream/training/EHR_training.py",
        "--model_type", model_type,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers),
        "--lr", "1e-4",
        "--patience", str(patience),
        "--task", task,
    ]
    if no_pos_weight:
        cmd.append("--no_pos_weight")
    
    # We will pass the ablation mode as a new argument if we update EHR_training.py 
    # Or we can use environment variables to communicate with the training script
    env = os.environ.copy()
    env["ABLATION_MODE"] = mode
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Experiment {mode} failed with error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EHR Ablation Study Runner')
    parser.add_argument('--group', type=str, choices=['leakage', 'static', 'temporal', 'modality', 'independent', 'equal_loss', 'all'], default='leakage')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    args = parser.parse_args()
    
    experiments = []
    if args.group == 'leakage':
        experiments = ['last_24h', 'first_48h', 'static_only', 'no_last_event', 'no_future']
    elif args.group == 'static':
        experiments = ['no_static', 'no_patient', 'no_admission']
    elif args.group == 'temporal':
        experiments = ['no_temporal']
    elif args.group == 'modality':
        experiments = ['no_labs', 'no_omr']
    elif args.group == 'independent':
        # Run 5 independent models, one for each task
        tasks = ['mortality', 'los_7d', 'readmission', 'progression', 'drug_rec']
        for task in tasks:
            run_experiment(f"independent_{task}", epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers, patience=args.patience, task=task)
        sys.exit(0)
    elif args.group == 'equal_loss':
        run_experiment("equal_loss", epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers, patience=args.patience, no_pos_weight=True)
        sys.exit(0)
    elif args.group == 'all':
        # Standard data/architecture ablations
        experiments = [
            # 'last_24h', 'first_48h', 'static_only', 
            # 'no_static', 'no_patient', 
            'no_admission', 
            'no_last_event',
            'no_temporal', 'no_labs', 
            'no_omr',
            
        ]
        for exp in experiments:
            run_experiment(exp, epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers, patience=args.patience)
        
        # Equal loss ablation
        run_experiment("equal_loss", epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers, patience=args.patience, no_pos_weight=True)
        
        # Independent task training ablations
        tasks = ['mortality', 'los_7d', 'readmission', 'progression', 'drug_rec']
        for task in tasks:
            run_experiment(f"independent_{task}", epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers, patience=args.patience, task=task)
            
        sys.exit(0)
        
    for exp in experiments:
        run_experiment(exp, epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers, patience=args.patience)
