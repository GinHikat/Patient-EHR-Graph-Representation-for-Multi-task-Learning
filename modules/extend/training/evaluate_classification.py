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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
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
            labels.append(data.get("gold_entities", []))
    return texts, labels

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = torch.sigmoid(torch.tensor(preds)).numpy()
    
    y_pred = (preds >= 0.5).astype(int)
    y_true = p.label_ids

    macro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        mAP = average_precision_score(y_true=y_true, y_score=preds, average='macro')

    return {
        "mAP": mAP,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "accuracy": accuracy
    }

script_dir = os.path.dirname(os.path.abspath(__file__))

def evaluate_model(
    model_name="vihealthbert",
    model_dir=os.path.join(script_dir, "results", "final_model"),
    classes_file=os.path.join(script_dir, "results", "classes.json"),
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
        
    # Remove "module." prefix if saved via nn.DataParallel in Kaggle
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
    metrics = trainer.evaluate()
    
    print("================ EVALUATION RESULTS ================")
    print(f"Validation Loss : {metrics.get('eval_loss', 0):.4f}")
    print(f"mAP             : {metrics.get('eval_mAP', 0):.4f}")
    print(f"Macro F1        : {metrics.get('eval_macro_f1', 0):.4f}")
    print(f"Micro F1        : {metrics.get('eval_micro_f1', 0):.4f}")
    print(f"Accuracy (EMR)  : {metrics.get('eval_accuracy', 0):.4f}")
    print("====================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Document Classification Model")
    parser.add_argument("--model_name", type=str, default="vihealthbert", help="Backbone model used during training (vihealthbert, phobert, sapbert)")
    args = parser.parse_args()
    
    evaluate_model(model_name=args.model_name)
