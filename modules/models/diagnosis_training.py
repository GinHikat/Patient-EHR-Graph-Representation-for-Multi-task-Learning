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
import joblib
import gc
import argparse
import dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
temp_root = os.path.dirname(script_dir) 
if temp_root not in sys.path:
    sys.path.append(temp_root)

from modules.models import ProcedureModel, PLMICDModel, MSMNModel, RadiologyDataset

data_dir = os.path.join(temp_root, 'data')
cleaned_data_dir = os.path.join(data_dir, 'cleaned')

def main(load_dir=None, truncation_level=200, others_limit=None, epochs=15, model_type='longformer'):
    # Setup models root
    models_root = os.path.join(temp_root, "models")
    os.makedirs(models_root, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    csv_path = os.path.join(data_dir, 'final_diagnosis.csv') 
    df = pd.read_csv(csv_path)
    df['category'] = df['category'].apply(ast.literal_eval)

    # Combine 'discharge' and 'radiology' columns for input
    print("Combining 'discharge' and 'radiology' columns for input...")
    required_cols = ['discharge', 'radiology']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Dataset must contain a '{col}' column.")
    
    # Fill NaN with empty strings and concatenate
    df['text'] = "Discharge Summary: " + df['discharge'].fillna('') + "\n\nRadiology Report: " + df['radiology'].fillna('')

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
    # num_labels is the number of labels AFTER truncation and mapping to 'Other'
    num_labels = len(mlb.classes_)
    print(f"Total Unique Labels (after truncation): {num_labels}")

    # Setup run folder for saving
    run_folder_name = f"diagnosis_run_{truncation_level}_{num_labels}_{model_type}"
    save_path = os.path.join(models_root, run_folder_name)
    os.makedirs(save_path, exist_ok=True)
    print(f"Results will be saved to: {save_path}")

    # Calculate class weights for BCE loss
    num_positives = binary_labels.sum(axis=0)
    num_negatives = len(binary_labels) - num_positives
    class_weights = np.clip(num_negatives / (num_positives + 1e-5), 1.0, 50.0)
    print(f"Loss weights (first 5): {class_weights[:5]}")

    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    # Initialize Model
    print("Initializing Model...")
    torch.cuda.empty_cache()

    if model_type == 'plmicd':
        print("Using PLM-ICD Model (Label Attention)...")
        pm = PLMICDModel(num_labels=num_labels)
    elif model_type == 'msmn':
        print("Using MSMN Model (Multi-Synonym Attention)...")
        pm = MSMNModel(num_labels=num_labels)
    else:
        print("Using Standard ProcedureModel (Clinical-Longformer)...")
        pm = ProcedureModel(num_labels=num_labels)
    
    # Save tokenizer and label encoder once at the start
    print(f"Saving setup files to {save_path}...")
    import joblib
    joblib.dump(mlb, os.path.join(save_path, "mlb.pkl"))
    pm.tokenizer.save_pretrained(save_path)
    
    if load_dir:
        target_load_dir = os.path.join(models_root, load_dir)
        if not os.path.exists(target_load_dir):
            if os.path.exists(load_dir):
                target_load_dir = load_dir
            else:
                print(f"Error: Directory {load_dir} not found.")
                target_load_dir = None
        
        if target_load_dir:
            weights_files = ["model_state.pt", "model.safetensors", "pytorch_model.bin"]
            found = False
            for wf in weights_files:
                wf_path = os.path.join(target_load_dir, wf)
                if os.path.exists(wf_path):
                    print(f"Loading weights from {wf_path}")
                    if wf.endswith(".safetensors"):
                        from safetensors.torch import load_file
                        state_dict = load_file(wf_path, device=str(pm.device))
                        state_dict = {k.replace(".gamma", ".weight").replace(".beta", ".bias"): v for k, v in state_dict.items()}
                        pm.model.load_state_dict(state_dict)
                        found = True
                    else:
                        state_dict = torch.load(wf_path, map_location=pm.device)
                        state_dict = {k.replace(".gamma", ".weight").replace(".beta", ".bias"): v for k, v in state_dict.items()}
                        pm.model.load_state_dict(state_dict)
                        found = True
                    if found: break
            if not found:
                print(f"Warning: No weights found in {target_load_dir}")
        print("Starting training from scratch.")
    
    checkpoint_path = os.path.join(save_path, "checkpoint.pt")
    start_epoch = 0
    start_step = 0
    
    if load_dir and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=pm.device)
        pm.model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_step = checkpoint['iteration']
        print(f"Resuming from Epoch {start_epoch + 1}, Step {start_step}")

    # Enable Gradient Checkpointing for Longformer memory efficiency
    pm.model.gradient_checkpointing_enable()
    
    # Dataloaders 
    train_dataset = RadiologyDataset(train_df['text'].tolist(), train_df['labels'].tolist(), pm.tokenizer, max_length=2048)
    val_dataset = RadiologyDataset(val_df['text'].tolist(), val_df['labels'].tolist(), pm.tokenizer, max_length=2048)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, num_workers=0, pin_memory=True)

    optimizer = AdamW(pm.model.parameters(), lr=1e-5, weight_decay=0.0001)
    if bnb: optimizer = bnb.optim.AdamW8bit(pm.model.parameters(), lr=1e-5)
    
    EPOCHS = epochs
    ACCUMULATION_STEPS = 4
    total_steps = (len(train_loader) // ACCUMULATION_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    # Use weighted BCE loss to boost confidence for sparse labels
    pos_weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(pm.device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    scaler = torch.cuda.amp.GradScaler()

    if load_dir and os.path.exists(checkpoint_path):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # Training loop
    print("Starting Training...")
    for epoch in range(start_epoch, EPOCHS):
        print(f"Epoch {epoch + 1} / {EPOCHS}")
        
        pm.model.train()
        total_train_loss = 0
        train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for i, batch in enumerate(train_iterator):
            # Skip steps if resuming
            if epoch == start_epoch and i < start_step:
                continue
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
            
            # Periodic Mid-Epoch Checkpointing
            if (i + 1) % 5000 == 0:
                print(f"\nSaving Mid-Epoch Checkpoint at Step {i+1}...")
                torch.save({
                    'epoch': epoch,
                    'iteration': i + 1,
                    'model_state_dict': pm.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                }, checkpoint_path)
            
            # Periodic Memory Cleanup
            if (i + 1) % 500 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            
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

        EVAL_THRESHOLD = 0.7
        
        metrics = pm.compute_metrics((np.vstack(all_logits), np.vstack(all_labels)), threshold=EVAL_THRESHOLD)
        
        # Log metrics to file inside the run folder
        log_file_path = os.path.join(save_path, "training_metrics.txt")
        mode = "a" if (epoch > 0 or load_dir) else "w"
        with open(log_file_path, mode) as f_run:
            if epoch == 0:
                f_run.write(f"Starting Training at {pd.Timestamp.now()}\n")
                f_run.write(f"Evaluation Threshold: {EVAL_THRESHOLD}\n")
                f_run.write("="*40 + "\n")
            
            f_run.write(f"Epoch {epoch + 1}/{EPOCHS}\n")
            f_run.write(f"Average Training Loss: {avg_train_loss:.4f}\n")
            f_run.write(f"Validation Loss: {avg_val_loss:.4f}\n")
            for key, val in metrics.items():
                f_run.write(f"{key}: {val:.4f}\n")
                print(f"{key}: {val:.4f}")
            f_run.write("-" * 20 + "\n")
        
        # Save model state dictionary and pretrained weights every epoch
        print(f"Saving checkpoint for Epoch {epoch + 1} to {save_path}...")
        pm.model.save_pretrained(save_path)
        torch.save(pm.model.state_dict(), os.path.join(save_path, "model_state.pt"))

    print("Training Complete!")
    print(f"Final diagnosis model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Diagnosis classifier")
    parser.add_argument("--load_dir", type=str, default=None)
    parser.add_argument("--truncation_level", type=int, default=200)
    parser.add_argument("--others_limit", type=int, default=None, help="Maximum number of 'Other' labels to keep (defaults to 2x truncation_level)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--model_type", type=str, default='longformer', choices=['longformer', 'plmicd', 'msmn'], 
                        help="Model architecture to use: 'longformer' (default), 'plmicd', or 'msmn'")
    args = parser.parse_args()
    
    main(
        load_dir=args.load_dir, 
        truncation_level=args.truncation_level, 
        others_limit=args.others_limit,
        epochs=args.epochs,
        model_type=args.model_type
    )
