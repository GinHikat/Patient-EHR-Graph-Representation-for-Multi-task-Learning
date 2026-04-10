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
import os
import sys
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
from modules.models import ProcedureModel, RadiologyDataset

data_dir = os.path.join(project_root, 'gin', 'data', 'Note')
cleaned_data_dir = os.path.join(data_dir, 'cleaned')

def main():
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(os.path.join(cleaned_data_dir, 'procedure_final.csv'))
    df['category'] = df['category'].apply(ast.literal_eval)

    truncation_level = 20
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
    
    train_dataset = RadiologyDataset(train_df['text'].tolist(), train_df['labels'].tolist(), pm.tokenizer)
    val_dataset = RadiologyDataset(val_df['text'].tolist(), val_df['labels'].tolist(), pm.tokenizer)
    test_dataset = RadiologyDataset(test_df['text'].tolist(), test_df['labels'].tolist(), pm.tokenizer)

    BATCH_SIZE = 4
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Training configuration
    EPOCHS = 5
    optimizer = AdamW(pm.model.parameters(), lr=1e-5, weight_decay=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training loop
    print("Starting Training...")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1} / {EPOCHS}")
        
        pm.model.train()
        total_train_loss = 0
        train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch in train_iterator:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(pm.device)
            attention_mask = batch['attention_mask'].to(pm.device)
            labels = batch['labels'].to(pm.device)
            
            outputs = pm.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            loss = loss_fn(logits, labels)
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pm.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_iterator.set_postfix(loss=f"{loss.item():.4f}")
            
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
        with open(log_file_path, "a") as f:
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
        
    print("Training Complete!")

    # Save results
    print("Saving model and tokenizer...")
    save_path = "./models/procedure_classifier"
    os.makedirs(save_path, exist_ok=True)
    
    pm.model.save_pretrained(save_path)
    pm.tokenizer.save_pretrained(save_path)
    
    # Save label encoder
    import joblib
    joblib.dump(mlb, os.path.join(save_path, "mlb.pkl"))
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
