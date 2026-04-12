import pandas as pd
import numpy as np
import torch
import ast
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm.auto import tqdm
try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None
import os
import sys
import warnings
import logging
import dotenv
import argparse
import joblib

dotenv.load_dotenv()

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from modules.models import ProcedureModel, RadiologyDataset

data_dir = os.path.join(project_root, 'data', 'Note')
cleaned_data_dir = os.path.join(data_dir, 'cleaned')

def main(load_dir=None, truncation_level=200, epochs=15):
    # Setup run folder for saving
    models_root = os.path.join(project_root, "models")
    os.makedirs(models_root, exist_ok=True)
    
    existing_runs = [d for d in os.listdir(models_root) if d.startswith("diagnosis_run_") and os.path.isdir(os.path.join(models_root, d))]
    next_i = max([int(r.split('_')[2]) for r in existing_runs] + [0]) + 1
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = f"diagnosis_run_{next_i}_{timestamp}"
    save_path = os.path.join(models_root, run_folder_name)
    print(f"Results will be saved to: {save_path}")

    # Load dataset
    print("Loading dataset...")
    csv_path = os.path.join(cleaned_data_dir, 'final_diagnosis.csv') 
    df = pd.read_csv(csv_path)
    df['category'] = df['category'].apply(ast.literal_eval)

    # Combine inputs: Merge Note and Discharge columns
    print("Creating combined inputs (Note + Discharge)...")
    # We use [SEP] to help the model distinguish different sections
    df['text'] = (
        "RADIOLOGY FINDINGS: " + df['Note'].fillna("N/A").astype(str) + 
        " [SEP] CLINICAL HISTORY: " + df['Discharge'].fillna("N/A").astype(str)
    )

    # Label Processing
    print(f"Truncating categories with less than {truncation_level} occurrences...")
    unique_categories = df['category'].explode().value_counts()
    rare_cats = set(unique_categories[unique_categories <= truncation_level].index)
    df['category'] = df['category'].apply(lambda cats: [c if c not in rare_cats else 'Other' for c in cats])

    print("Encoding labels...")
    mlb = MultiLabelBinarizer()
    binary_labels = mlb.fit_transform(df['category'])
    df['labels'] = list(binary_labels.astype(float))
    num_labels = len(mlb.classes_)
    print(f"Total Unique Labels: {num_labels}")

    # Split dataset
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    # Initialize Model
    print("Initializing Model...")
    torch.cuda.empty_cache()
    # Reusing ProcedureModel logic as it is a generic multi-label classifier
    pm = ProcedureModel(num_labels=num_labels) 
    
    if load_dir:
        # Load weights logic (omitted for brevity, same as procedure_training.py)
        pass

    # Enable Gradient Checkpointing for Longformer memory efficiency
    pm.model.gradient_checkpointing_enable()
    
    # Dataloaders - Input uses the new 'text' column containing both sources
    train_dataset = RadiologyDataset(train_df['text'].tolist(), train_df['labels'].tolist(), pm.tokenizer)
    val_dataset = RadiologyDataset(val_df['text'].tolist(), val_df['labels'].tolist(), pm.tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=4, pin_memory=True)

    optimizer = AdamW(pm.model.parameters(), lr=1e-5, weight_decay=0.001)
    if bnb: optimizer = bnb.optim.AdamW8bit(pm.model.parameters(), lr=1e-5)
    
    EPOCHS = epochs
    ACCUMULATION_STEPS = 4
    total_steps = (len(train_loader) // ACCUMULATION_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = torch.nn.BCEWithLogitsLoss()

    #Training loop
    print("Starting Training...")
    for epoch in range(EPOCHS):
        pm.model.train()

        os.makedirs(save_path, exist_ok=True)

    # Save Final Model
    print(f"Saving diagnosis model to {save_path}...")
    pm.model.save_pretrained(save_path)
    pm.tokenizer.save_pretrained(save_path)
    joblib.dump(mlb, os.path.join(save_path, "mlb.pkl"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Diagnosis classifier")
    parser.add_argument("--load_dir", type=str, default=None)
    parser.add_argument("--truncation_level", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    
    main(
        load_dir=args.load_dir, 
        truncation_level=args.truncation_level, 
        epochs=args.epochs
    )
