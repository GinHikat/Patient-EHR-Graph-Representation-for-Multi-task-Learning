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
temp_root = os.path.dirname(script_dir) 
if temp_root not in sys.path:
    sys.path.append(temp_root)

from modules.models import ProcedureModel, RadiologyDataset

data_dir = os.path.join(temp_root, 'data', 'Note')
cleaned_data_dir = os.path.join(data_dir, 'cleaned')

def main(load_dir=None, truncation_level=200, others_limit=None, epochs=15):
    # Setup run folder for saving
    models_root = os.path.join(temp_root, "models")
    os.makedirs(models_root, exist_ok=True)
    
    existing_runs = [d for d in os.listdir(models_root) if d.startswith("diagnosis_run_") and os.path.isdir(os.path.join(models_root, d))]
    next_i = max([int(r.split('_')[2]) for r in existing_runs] + [0]) + 1
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = f"diagnosis_run_{next_i}_{timestamp}_diagnosis"
    save_path = os.path.join(models_root, run_folder_name)
    print(f"Results will be saved to: {save_path}")

    # Load dataset
    print("Loading dataset...")
    csv_path = os.path.join(cleaned_data_dir, 'final_diagnosis.csv') 
    df = pd.read_csv(csv_path)
    df['category'] = df['category'].apply(ast.literal_eval)

    # Use existing 'text' column as input
    print("Using 'text' column as input...")
    if 'text' not in df.columns:
        raise KeyError("Dataset must contain a 'text' column.")

    # Label Processing
    print(f"Truncating categories with less than {truncation_level} occurrences...")
    unique_categories = df['category'].explode().value_counts()
    rare_cats = set(unique_categories[unique_categories <= truncation_level].index)
    
    # Map to 'Other' and deduplicate
    df['category'] = df['category'].apply(lambda cats: list(set([c if c not in rare_cats else 'Other' for c in cats])))

    # Limit 'Other' label to avoid extreme imbalance
    if others_limit is None:
        others_limit = 2 * truncation_level
        
    if others_limit is not None:
        has_other = df['category'].apply(lambda cats: 'Other' in cats)
        other_indices = df[has_other].index
        
        if len(other_indices) > others_limit:
            print(f"Limiting 'Other' instances from {len(other_indices)} to at most {others_limit}...")
            np.random.seed(42)
            keep_indices = set(np.random.choice(other_indices, others_limit, replace=False))
            remove_indices = other_indices.difference(pd.Index(list(keep_indices)))
            
            # Remove 'Other' from rows not selected to keep it
            df.loc[remove_indices, 'category'] = df.loc[remove_indices, 'category'].apply(
                lambda cats: [c for c in cats if c != 'Other']
            )
            
            # Drop rows that now have no labels left
            initial_count = len(df)
            df = df[df['category'].apply(len) > 0].reset_index(drop=True)
            if initial_count > len(df):
                print(f"Dropped {initial_count - len(df)} samples that had no labels left after limiting 'Other'.")

    print("Encoding labels...")
    mlb = MultiLabelBinarizer()
    binary_labels = mlb.fit_transform(df['category'])
    df['labels'] = list(binary_labels.astype(float))
    num_labels = len(mlb.classes_)
    print(f"Total Unique Labels: {num_labels}")

    # Calculate class weights for BCE loss
    num_positives = binary_labels.sum(axis=0)
    num_negatives = len(binary_labels) - num_positives
    class_weights = np.clip(num_negatives / (num_positives + 1e-5), 1.0, 50.0)
    print(f"Loss weights (first 5): {class_weights[:5]}")


    # Split dataset
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    # Initialize Model
    print("Initializing Model...")
    torch.cuda.empty_cache()
    # Reusing ProcedureModel logic as it is a generic multi-label classifier
    pm = ProcedureModel(num_labels=num_labels) 
    
    # Save tokenizer and label encoder once at the start
    print(f"Saving setup files to {save_path}...")
    import joblib
    joblib.dump(mlb, os.path.join(save_path, "mlb.pkl"))
    pm.tokenizer.save_pretrained(save_path)
    
    if load_dir:
        # Load weights logic (omitted for brevity, same as procedure_training.py)
        pass

    # Enable Gradient Checkpointing for Longformer memory efficiency
    pm.model.gradient_checkpointing_enable()
    
    # Dataloaders - Input uses the new 'text' column containing both sources
    train_dataset = RadiologyDataset(train_df['text'].tolist(), train_df['labels'].tolist(), pm.tokenizer, max_length=2048)
    val_dataset = RadiologyDataset(val_df['text'].tolist(), val_df['labels'].tolist(), pm.tokenizer, max_length=2048)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, num_workers=4, pin_memory=True)

    optimizer = AdamW(pm.model.parameters(), lr=1e-5, weight_decay=0.001)
    if bnb: optimizer = bnb.optim.AdamW8bit(pm.model.parameters(), lr=1e-5)
    
    EPOCHS = epochs
    ACCUMULATION_STEPS = 4
    total_steps = (len(train_loader) // ACCUMULATION_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    # Use weighted BCE loss to boost confidence for sparse labels
    pos_weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(pm.device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    scaler = torch.cuda.amp.GradScaler()


    # Training loop
    print("Starting Training...")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1} / {EPOCHS}")
        
        pm.model.train()
        total_train_loss = 0
        train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for i, batch in enumerate(train_iterator):
            input_ids = batch['input_ids'].to(pm.device)
            attention_mask = batch['attention_mask'].to(pm.device)
            labels = batch['labels'].to(pm.device)
            
            with torch.cuda.amp.autocast():
                outputs = pm.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss = loss / ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(pm.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            train_iterator.set_postfix(loss=f"{(loss.item() * ACCUMULATION_STEPS):.4f}")
            total_train_loss += loss.item() * ACCUMULATION_STEPS
            
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        print("Running Validation...")
        pm.model.eval()
        total_val_loss = 0
        all_logits, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(pm.device)
                attention_mask = batch['attention_mask'].to(pm.device)
                labels = batch['labels'].to(pm.device)
                
                with torch.cuda.amp.autocast():
                    outputs = pm.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = loss_fn(logits, labels)
                    
                total_val_loss += loss.item()
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
        avg_val_loss = total_val_loss / len(val_loader)
        EVAL_THRESHOLD = 0.3
        metrics = pm.compute_metrics((np.vstack(all_logits), np.vstack(all_labels)), threshold=EVAL_THRESHOLD)
        
        # Save metrics to run folder
        os.makedirs(save_path, exist_ok=True)
        mode = "w" if epoch == 0 else "a"
        with open(os.path.join(save_path, "training_metrics.txt"), mode) as f_run:
            if epoch == 0:
                f_run.write(f"Starting Training at {pd.Timestamp.now()}\n")
            f_run.write(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")
            for key, val in metrics.items():
                f_run.write(f"{key}: {val:.4f}\n")
            f_run.write("-" * 20 + "\n")
        
        # Save model state dictionary and pretrained weights every epoch
        print(f"Saving checkpoint for Epoch {epoch + 1} to {save_path}...")
        pm.model.save_pretrained(save_path)
        torch.save(pm.model.state_dict(), os.path.join(save_path, "model_state.pt"))

        
        for key, val in metrics.items():
            print(f"{key}: {val:.4f}")

    print("Training Complete!")
    print(f"Final diagnosis model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Diagnosis classifier")
    parser.add_argument("--load_dir", type=str, default=None)
    parser.add_argument("--truncation_level", type=int, default=200)
    parser.add_argument("--others_limit", type=int, default=None, help="Maximum number of 'Other' labels to keep (defaults to 2x truncation_level)")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    
    main(
        load_dir=args.load_dir, 
        truncation_level=args.truncation_level, 
        others_limit=args.others_limit,
        epochs=args.epochs
    )
