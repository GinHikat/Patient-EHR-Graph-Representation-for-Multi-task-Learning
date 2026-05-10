import os, sys
import argparse
import subprocess
import datetime

# This script is a wrapper around EHR_training.py to run different ablation studies

def run_experiment(mode, epochs=10, model_type='transformer'):
    print(f"\n" + "="*60)
    print(f"RUNNING ABLATION: {mode}")
    print("="*60 + "\n")
    
    cmd = [
        "python3", "modules/downstream/training/EHR_training.py",
        "--model_type", model_type,
        "--epochs", str(epochs),
        "--batch_size", "32",
        "--lr", "1e-4",
    ]
    
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
    parser.add_argument('--group', type=str, choices=['leakage', 'static', 'temporal', 'all'], default='leakage')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    
    experiments = []
    if args.group == 'leakage':
        experiments = ['no_last_note', 'first_48h', 'static_only']
    elif args.group == 'static':
        experiments = ['no_static']
    elif args.group == 'temporal':
        experiments = ['no_dt_decay']
    elif args.group == 'all':
        experiments = ['no_last_note', 'first_48h', 'static_only', 'no_static', 'no_dt_decay']
        
    for exp in experiments:
        run_experiment(exp, epochs=args.epochs)
