import pandas as pd
import numpy as np
import torch
import ast
import os
import sys
import warnings
import logging
import joblib
import argparse
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from modules.models import ProcedureModel, PLMICDModel, MSMNModel, RadiologyDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

def main():
    parser = argparse.ArgumentParser(description="Testing Diagnosis classifier")
    parser.add_argument("--load_dir", type=str, required=True, help="Path to the model directory (e.g., models/diagnosis_run_600_267_plmicd)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for metrics")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_file", type=str, default="test_results.txt")
    args = parser.parse_args()

    # Automatically resolve path if it's just the folder name or relative to project root
    load_path = args.load_dir
    if not os.path.exists(load_path):
        # Try relative to project root
        temp_path = os.path.join(project_root, args.load_dir)
        if os.path.exists(temp_path):
            load_path = temp_path
        else:
            # Try looking inside project_root/models/
            temp_path = os.path.join(project_root, "models", os.path.basename(args.load_dir))
            if os.path.exists(temp_path):
                load_path = temp_path
            else:
                raise FileNotFoundError(f"Model directory not found: {args.load_dir}")

    print(f"Using model directory: {load_path}")
    
    # Automatically parse model_type and truncation_level from the folder name
    # Expected format: diagnosis_run_{truncation_level}_{num_labels}_{model_type}
    folder_name = os.path.basename(load_path.rstrip('/'))
    parts = folder_name.split('_')
    
    try:
        if len(parts) >= 5:
            truncation_level = int(parts[2])
            model_type = parts[4]
            print(f"Auto-detected from folder name: model_type={model_type}, truncation_level={truncation_level}")
        else:
            raise ValueError("Folder name does not follow the expected format.")
    except (ValueError, IndexError):
        print(f"Warning: Could not parse folder name '{folder_name}'. Defaulting to model_type='longformer', truncation_level=200.")
        model_type = 'longformer'
        truncation_level = 200

    data_dir = os.path.join(project_root, 'data')
    
    # 1. Load dataset
    print("Loading dataset...")
    csv_path = os.path.join(data_dir, 'final_diagnosis.csv') 
    df = pd.read_csv(csv_path)
    df['category'] = df['category'].apply(ast.literal_eval)

    # Combine 'discharge' and 'radiology' columns for input
    df['text'] = "Discharge Summary: " + df['discharge'].fillna('') + "\n\nRadiology Report: " + df['radiology'].fillna('')

    # 2. Label Processing (Must be identical to training)
    print(f"Applying truncation level {truncation_level}...")
    unique_categories = df['category'].explode().value_counts()
    rare_cats = set(unique_categories[unique_categories <= truncation_level].index)
    df['category'] = df['category'].apply(lambda cats: list(set([c if c not in rare_cats else 'Other' for c in cats])))

    # others_limit is derived from truncation_level as in training
    others_limit = 2 * truncation_level
        
    has_other = df['category'].apply(lambda cats: 'Other' in cats)
    other_indices = df[has_other].index
    
    if len(other_indices) > others_limit:
        print(f"Limiting 'Other' instances to {others_limit}...")
        np.random.seed(42)
        keep_indices = set(np.random.choice(other_indices, others_limit, replace=False))
        remove_indices = other_indices.difference(pd.Index(list(keep_indices)))
        df.loc[remove_indices, 'category'] = df.loc[remove_indices, 'category'].apply(
            lambda cats: [c for c in cats if c != 'Other']
        )
        df = df[df['category'].apply(len) > 0].reset_index(drop=True)

    # 3. Load MultiLabelBinarizer
    mlb_path = os.path.join(load_path, "mlb.pkl")
    if not os.path.exists(mlb_path):
        raise FileNotFoundError(f"mlb.pkl not found in {load_path}")
    mlb = joblib.load(mlb_path)
    num_labels = len(mlb.classes_)
    print(f"Loaded MultiLabelBinarizer with {num_labels} classes.")

    binary_labels = mlb.transform(df['category'])
    df['labels'] = list(binary_labels.astype(float))

    # 4. Split data (Must be identical to training)
    from sklearn.model_selection import train_test_split
    _, temp_df = train_test_split(df, test_size=0.20, random_state=42)
    _, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
    
    print(f"Test set size: {len(test_df)}")

    # 5. Initialize Model
    print(f"Initializing {model_type} model...")
    if model_type == 'plmicd':
        pm = PLMICDModel(num_labels=num_labels)
    elif model_type == 'msmn':
        pm = MSMNModel(num_labels=num_labels)
    else:
        pm = ProcedureModel(num_labels=num_labels)

    # 6. Load Weights
    weights_path = os.path.join(load_path, "model_state.pt")
    if not os.path.exists(weights_path):
        # Check for alternative weight names if model_state.pt doesn't exist
        for wf in ["pytorch_model.bin", "model.safetensors"]:
            if os.path.exists(os.path.join(load_path, wf)):
                weights_path = os.path.join(load_path, wf)
                break
    
    print(f"Loading weights from {weights_path}...")
    if weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(weights_path, device=str(pm.device))
    else:
        state_dict = torch.load(weights_path, map_location=pm.device)
    
    # Handle possible key mismatches
    state_dict = {k.replace(".gamma", ".weight").replace(".beta", ".bias"): v for k, v in state_dict.items()}
    pm.model.load_state_dict(state_dict)
    pm.model.eval()

    # 7. Create DataLoader
    test_dataset = RadiologyDataset(test_df['text'].tolist(), test_df['labels'].tolist(), pm.tokenizer, max_length=2048)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True)

    # 8. Evaluation
    print("Running evaluation on test set...")
    all_logits, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(pm.device)
            attention_mask = batch['attention_mask'].to(pm.device)
            labels = batch['labels'].to(pm.device)
            
            with torch.cuda.amp.autocast():
                outputs = pm.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 9. Compute Metrics
    print(f"Computing metrics with threshold={args.threshold}...")
    logits_v = np.vstack(all_logits)
    labels_v = np.vstack(all_labels)
    
    # Calculate probabilities for distribution summary
    probs_v = 1 / (1 + np.exp(-logits_v))
    print("\nProbability Distribution Summary:")
    print(f"  Min: {np.min(probs_v):.4f}, Max: {np.max(probs_v):.4f}, Mean: {np.mean(probs_v):.4f}")
    for q in [10, 25, 50, 75, 90]:
        print(f"  {q}th percentile: {np.percentile(probs_v, q):.4f}")
    print("")

    metrics = pm.compute_metrics((logits_v, labels_v), threshold=args.threshold)

    # 10. Save and Display Results
    output_path = os.path.join(load_path, args.output_file)
    print(f"\nTest Results for {load_path}:")
    with open(output_path, "w") as f:
        f.write(f"Test Results for Model: {load_path}\n")
        f.write(f"Classification Threshold: {args.threshold}\n")
        f.write("="*40 + "\n")
        for key, val in metrics.items():
            result_str = f"{key}: {val:.4f}"
            print(result_str)
            f.write(result_str + "\n")

    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()

