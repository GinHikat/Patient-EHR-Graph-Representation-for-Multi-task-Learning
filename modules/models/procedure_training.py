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
import datetime
import argparse

dotenv.load_dotenv()

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Define paths relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
temp_root = os.path.dirname(script_dir) # i.e., /home/hngoc/gin/Temp
from modules.models import ProcedureModel, RadiologyDataset

data_dir = os.path.join(temp_root, 'data', 'Note')
cleaned_data_dir = os.path.join(data_dir, 'cleaned')

def main(load_dir=None, truncation_level=200, epochs=15):
    # Determine the run folder
    gin_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_root = os.path.join(gin_root, "models")
    os.makedirs(models_root, exist_ok=True)
    existing_runs = [d for d in os.listdir(models_root) if d.startswith("run_") and os.path.isdir(os.path.join(models_root, d))]
    
    def get_run_idx(folder_name):
        parts = folder_name.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1])
        return 0
    
    if existing_runs:
        last_i = max(get_run_idx(r) for r in existing_runs)
        next_i = last_i + 1
    else:
        next_i = 1
        
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = f"run_{next_i}_{timestamp}"
    save_path = os.path.join(models_root, run_folder_name)
    print(f"Results will be saved to: {save_path}")

    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(os.path.join(cleaned_data_dir, 'procedure_final.csv'))
    df['category'] = df['category'].apply(ast.literal_eval)

    # truncation_level is now passed from args
    print(f"Truncating categories with less than {truncation_level} occurrences...")

    unique_categories = df['category'].explode().value_counts()
    rare_cats = set(unique_categories[unique_categories <= truncation_level].index)

    df['category'] = df['category'].apply(
        lambda cats: [c if c not in rare_cats else 'Other' for c in cats]
    )

    # Encode labels
    print("Encoding labels...")
    mlb = MultiLabelBinarizer()
    binary_labels = mlb.fit_transform(df['category'])
    
    label_dictionary = {idx: class_name for idx, class_name in enumerate(mlb.classes_)}
    print(f"Total Unique Labels Discovered: {len(label_dictionary)}")
    
    df['labels'] = list(binary_labels.astype(float))

    # Split dataset
    print("Splitting datasets...")
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
    
    print(f"Train/Val/Test Sizes: {len(train_df)} / {len(val_df)} / {len(test_df)}")

    # Initialize model and dataloaders
    print("Initializing Model and Dataloaders...")
    torch.cuda.empty_cache()
    num_labels = len(label_dictionary)
    pm = ProcedureModel(num_labels=num_labels)
    
    trainable_params = sum(p.numel() for p in pm.model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {trainable_params:,}")

    # Load state_dict if load_dir is provided
    if load_dir:
        target_load_dir = os.path.join(models_root, load_dir)
        if not os.path.exists(target_load_dir):
            # Try as absolute or direct relative path if models_root concatenation doesn't exist
            if os.path.exists(load_dir):
                target_load_dir = load_dir
            else:
                print(f"Error: Directory {load_dir} not found in {models_root} or as direct path.")
                target_load_dir = None
        
        if target_load_dir:
            # Try to find a weights file
            weights_files = ["model_state.pt", "model.safetensors", "pytorch_model.bin"]
            found = False
            for wf in weights_files:
                wf_path = os.path.join(target_load_dir, wf)
                if os.path.exists(wf_path):
                    print(f"Loading state_dict from {wf_path}")
                    if wf.endswith(".safetensors"):
                        try:
                            from safetensors.torch import load_file
                            state_dict = load_file(wf_path, device=str(pm.device))
                            # Handle potential key mismatches (gamma/beta vs weight/bias)
                            state_dict = {k.replace(".gamma", ".weight").replace(".beta", ".bias"): v for k, v in state_dict.items()}
                            pm.model.load_state_dict(state_dict)
                            found = True
                        except ImportError:
                            print("safetensors library not found. Please install it to load .safetensors files.")
                    else:
                        state_dict = torch.load(wf_path, map_location=pm.device)
                        # Handle potential key mismatches (gamma/beta vs weight/bias)
                        state_dict = {k.replace(".gamma", ".weight").replace(".beta", ".bias"): v for k, v in state_dict.items()}
                        pm.model.load_state_dict(state_dict)
                        found = True
                    
                    if found:
                        break
            
            if not found:
                print(f"Warning: No weights file found in {target_load_dir}. Expected one of {weights_files}")
        else:
            print("Warning: load_dir was provided but target directory could not be located.")
    else:
        print("Starting training from scratch (no load_dir provided).")
    
    # Enable Gradient Checkpointing (Huge memory saving for Longformer)
    pm.model.gradient_checkpointing_enable()
    
    train_dataset = RadiologyDataset(train_df['text'].tolist(), train_df['labels'].tolist(), pm.tokenizer)
    val_dataset = RadiologyDataset(val_df['text'].tolist(), val_df['labels'].tolist(), pm.tokenizer)
    test_dataset = RadiologyDataset(test_df['text'].tolist(), test_df['labels'].tolist(), pm.tokenizer)

    BATCH_SIZE = 8
    ACCUMULATION_STEPS = 4  
    
    # Optimized DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=4, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=4, 
        pin_memory=True
    )

    EPOCHS = epochs
    
    # Use 8-bit Optimizer if available (saves VRAM on optimizer states)
    if bnb:
        print("Using 8-bit AdamW Optimizer...")
        optimizer = bnb.optim.AdamW8bit(pm.model.parameters(), lr=1e-5, weight_decay=0.001)
    else:
        optimizer = AdamW(pm.model.parameters(), lr=1e-5, weight_decay=0.001)
        
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    # Calculate total steps based on accumulation
    total_steps = (len(train_loader) // ACCUMULATION_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    # Initialize Mixed Precision Scaler
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
            
            # Forward pass with AMP autocast
            with torch.cuda.amp.autocast():
                outputs = pm.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, labels)
                
                # Normalize loss for accumulation
                loss = loss / ACCUMULATION_STEPS
            
            # Backward pass with scaled gradients (gradients are accumulated)
            scaler.scale(loss).backward()
            
            # Only update weights every ACCUMULATION_STEPS
            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(pm.model.parameters(), 1.0)
                
                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            # Log the full loss (re-multiply for logging)
            train_iterator.set_postfix(loss=f"{(loss.item() * ACCUMULATION_STEPS):.4f}")
            total_train_loss += loss.item() * ACCUMULATION_STEPS
            
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        print("Running Validation...")
        pm.model.eval()
        total_val_loss = 0
        all_logits = []
        all_labels = []
        
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
        print(f"Validation Loss: {avg_val_loss:.4f}")

        EVAL_THRESHOLD = 0.3
        metrics = pm.compute_metrics((np.vstack(all_logits), np.vstack(all_labels)), threshold=EVAL_THRESHOLD)
        
        # Log metrics to file
        log_file_path = "training_metrics.txt"
        mode = "w" if epoch == 0 else "a"
        with open(log_file_path, mode) as f:
            if epoch == 0:
                f.write(f"Starting Training at {pd.Timestamp.now()}\n")
                f.write(f"Evaluation Threshold: {EVAL_THRESHOLD}\n")
                f.write("="*40 + "\n")
            
            f.write(f"Epoch {epoch + 1}/{EPOCHS}\n")
            f.write(f"Average Training Loss: {avg_train_loss:.4f}\n")
            f.write(f"Validation Loss: {avg_val_loss:.4f}\n")
            for key, val in metrics.items():
                f.write(f"{key}: {val:.4f}\n")
                print(f"{key}: {val:.4f}")
            f.write("-" * 20 + "\n")
        
        # Also copy metrics to the run folder
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "training_metrics.txt"), mode) as f_run:
            if epoch == 0:
                f_run.write(f"Starting Training at {pd.Timestamp.now()}\n")
                f_run.write(f"Evaluation Threshold: {EVAL_THRESHOLD}\n")
                f_run.write("="*40 + "\n")
            f_run.write(f"Epoch {epoch + 1}/{EPOCHS}\n")
            f_run.write(f"Average Training Loss: {avg_train_loss:.4f}\n")
            f_run.write(f"Validation Loss: {avg_val_loss:.4f}\n")
            for key, val in metrics.items():
                f_run.write(f"{key}: {val:.4f}\n")
            f_run.write("-" * 20 + "\n")
        
    print("Training Complete!")

    # Save results
    print(f"Saving model and tokenizer to {save_path}...")
    os.makedirs(save_path, exist_ok=True)
    
    pm.model.save_pretrained(save_path)
    pm.tokenizer.save_pretrained(save_path)
    torch.save(pm.model.state_dict(), os.path.join(save_path, "model_state.pt"))
    
    # Save label encoder
    import joblib
    joblib.dump(mlb, os.path.join(save_path, "mlb.pkl"))
    print(f"Model and state_dict saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training procedure classifier")
    parser.add_argument("--load_dir", type=str, default=None, help="Name of the folder in ./models to load state_dict from")
    parser.add_argument("--truncation_level", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    
    main(
        load_dir=args.load_dir, 
        truncation_level=args.truncation_level, 
        epochs=args.epochs
    )
