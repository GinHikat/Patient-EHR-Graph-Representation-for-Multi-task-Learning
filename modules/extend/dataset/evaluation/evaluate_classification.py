import os
import sys
import json
import torch
import argparse
from datasets import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, average_precision_score, accuracy_score
from transformers import AutoTokenizer, EvalPrediction, Trainer, TrainingArguments
from safetensors.torch import load_file
import warnings

# Add root project dir to path so we can import our modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.extend.model.plmicd_model import PLMICDModel

def load_data(filepath):
    texts = []
    labels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            texts.append(data.get("text", ""))
            
            gold_ent = data.get("gold_entities", [])
            if isinstance(gold_ent, dict):
                # Load ONLY Disease/Symptom
                disease_cuis = gold_ent.get("Disease/Symptom", [])
                labels.append(list(set(disease_cuis)))
            else:
                labels.append(gold_ent)
    return texts, labels

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = torch.sigmoid(torch.tensor(preds)).numpy()
    
    y_pred = (preds >= 0.5).astype(int)
    y_true = p.label_ids

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        try:
            from sklearn.metrics import average_precision_score
            mAP = average_precision_score(y_true=y_true, y_score=preds, average='macro')
        except:
            mAP = 0

    return {
        "mAP": mAP,
        "accuracy": accuracy
    }

script_dir = os.path.dirname(os.path.abspath(__file__))

def evaluate_model(
    model_name="vihealthbert",
    model_dir=os.path.join(project_root, "modules", "extend", "training", "results", "final_model"),
    classes_file=os.path.join(project_root, "modules", "extend", "training", "results", "classes.json"),
    data_file="data/viettel/vietnamese_ner/training/vietnamese/document_classification/cui_mapped_doc_class_test.jsonl",
    batch_size=16
):
    print(f"--- Starting Local Evaluation for {model_name} ---")
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Cannot find model directory at {model_dir}. Please download it from Kaggle and place it here.")
    if not os.path.exists(classes_file):
        raise FileNotFoundError(f"Cannot find classes.json at {classes_file}. Please download it from Kaggle.")

    print("Loading MultiLabelBinarizer vocabulary...")
    with open(classes_file, "r", encoding="utf-8") as f:
        classes = json.load(f)
    
    mlb = MultiLabelBinarizer()
    mlb.fit([classes]) # Fit to the exact vocabulary used during training
    num_labels = len(classes)
    print(f"Total unique CUI classes: {num_labels}")
    
    print(f"Loading data from {data_file}...")
    texts, labels = load_data(data_file)
    print(f"Evaluation size: {len(texts)}")
    
    y_true = mlb.transform(labels)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    def tokenize_function(texts, labels):
        encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=256)
        items = []
        for i in range(len(texts)):
            item = {key: val[i] for key, val in encodings.items()}
            item['labels'] = labels[i].tolist()
            items.append(item)
        return items
        
    print("Tokenizing datasets...")
    eval_dataset = Dataset.from_list(tokenize_function(texts, y_true))
    
    print(f"Initializing Model Architecture for {model_name}...")
    # Initialize the base architecture so the tensors are the right size
    model = PLMICDModel(num_labels=num_labels, model_name=model_name)
    
    print("Loading trained weights from safetensors...")
    weights_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(weights_path):
        state_dict = load_file(weights_path)
    else:
        # Fallback to pytorch_model.bin if safetensors isn't used
        state_dict = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu")
        
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v
            
    model.load_state_dict(clean_state_dict)
    
    eval_args = TrainingArguments(
        output_dir="models/doc_classifier/eval_temp",
        per_device_eval_batch_size=batch_size,
        report_to="none",
        fp16=torch.cuda.is_available()
    )
    
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    
    print("\nRunning evaluation on the dataset...\n")
    pred_output = trainer.predict(eval_dataset)
    metrics = pred_output.metrics
    
    # Calculate explicit string-based TP, FP, FN against unfiltered raw labels
    preds = torch.sigmoid(torch.tensor(pred_output.predictions)).numpy()
    y_pred = (preds >= 0.5).astype(int)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    import numpy as np
    
    for i in range(len(y_pred)):
        pred_indices = np.where(y_pred[i] == 1)[0]
        pred_cuis = set([classes[idx] for idx in pred_indices])
        gold_cuis = set(labels[i])
        
        total_tp += len(gold_cuis.intersection(pred_cuis))
        total_fp += len(pred_cuis - gold_cuis)
        total_fn += len(gold_cuis - pred_cuis)
        
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    
    print("================ EVALUATION RESULTS ================")
    print(f"Validation Loss : {metrics.get('test_loss', 0):.4f}")
    print(f"mAP             : {metrics.get('test_mAP', 0):.4f}")
    print(f"Accuracy (EMR)  : {metrics.get('test_accuracy', 0):.4f}")
    print("----------------------------------------------------")
    print(f"OVERALL METRICS (Unfiltered Set-based TP/FP/FN):")
    print(f"  True Positives  : {total_tp}")
    print(f"  False Positives : {total_fp}")
    print(f"  False Negatives : {total_fn}")
    print("----------------------------------------------------")
    print(f"  Precision       : {micro_p:.4f}")
    print(f"  Recall          : {micro_r:.4f}")
    print(f"  F1-Score        : {micro_f1:.4f}")
    print("====================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Document Classification Model")
    parser.add_argument("--model_name", type=str, default="vihealthbert", help="Backbone model used during training (vihealthbert, phobert, sapbert)")
    parser.add_argument("--data_file", type=str, default=None, help="Path to the JSONL dataset to evaluate on")
    args = parser.parse_args()
    
    kwargs = {"model_name": args.model_name}
    if args.data_file:
        kwargs["data_file"] = args.data_file
        
    evaluate_model(**kwargs)
