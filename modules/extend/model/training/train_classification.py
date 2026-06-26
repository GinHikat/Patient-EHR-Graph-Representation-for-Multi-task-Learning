import os
import sys
import json
import torch
import argparse
import numpy as np
from datasets import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, average_precision_score, accuracy_score
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction
)

# Add root project dir to path so we can import our modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.extend.model.plmicd_model import PLMICDModel, MODEL_CHOICES

def load_data(filepath):
    texts = []
    labels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            # Structure: {"text": "...", "gold_entities": ["C123", "C456"]}
            texts.append(data.get("text", ""))
            labels.append(data.get("labels", []))
    return texts, labels

import warnings

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    # Apply sigmoid since model outputs raw logits
    preds = torch.sigmoid(torch.tensor(preds)).numpy()
    
    # Threshold at 0.5 for multi-label
    y_pred = (preds >= 0.5).astype(int)
    y_true = p.label_ids

    macro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    
    # Suppress the "No positive class found in y_true" warning for mAP
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        # mAP (Mean Average Precision) uses raw probabilities
        mAP = average_precision_score(y_true=y_true, y_score=preds, average='macro')

    return {
        "mAP": mAP,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "accuracy": accuracy
    }

def train_classifier(
    model_name="vihealthbert", 
    epochs=10, 
    batch_size=16, 
    lr=5e-5, 
    base_dir="data/viettel/vietnamese_ner/training/vietnamese/document_classification",
    output_dir="models/doc_classifier"
):
    print(f"--- Starting Document Classification Training ---")
    print(f"Model: {model_name} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    
    # Determine HF Path for Tokenizer
    model_path = MODEL_CHOICES.get(model_name.lower(), model_name)
    
    train_file = os.path.join(base_dir, "doc_class_train.jsonl")
    dev_file = os.path.join(base_dir, "doc_class_dev.jsonl")
    test_file = os.path.join(base_dir, "doc_class_test.jsonl")
    
    print(f"Loading data from {base_dir}...")
    train_texts, train_labels = load_data(train_file)
    dev_texts, dev_labels = load_data(dev_file)
    test_texts, test_labels = load_data(test_file)
    
    print(f"Train size: {len(train_texts)} | Dev size: {len(dev_texts)} | Test size: {len(test_texts)}")
    
    # Fit MultiLabelBinarizer but DO NOT transform the whole dataset yet to avoid RAM explosion
    mlb = MultiLabelBinarizer()
    mlb.fit(train_labels + dev_labels)
    num_labels = len(mlb.classes_)
    print(f"Total unique CUI classes: {num_labels}")
    
    # Save MLB classes for later inference
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(mlb.classes_.tolist(), f)
    
    # Tokenization
    print(f"Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("Converting to HuggingFace Datasets and Tokenizing in batches (Memory Efficient)...")
    
    # We pass the raw string labels into the dataset. 
    # They take almost zero memory compared to dense multi-hot vectors.
    train_dataset = Dataset.from_dict({'text': train_texts, 'raw_labels': train_labels})
    dev_dataset = Dataset.from_dict({'text': dev_texts, 'raw_labels': dev_labels})
    test_dataset = Dataset.from_dict({'text': test_texts, 'raw_labels': test_labels})
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
        
    # Map tokenization only. No binarization happens here!
    # This keeps PyArrow memory practically at 0 bytes.
    train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=1000)
    dev_dataset = dev_dataset.map(tokenize_function, batched=True, batch_size=1000)
    test_dataset = test_dataset.map(tokenize_function, batched=True, batch_size=1000)
    
    # Remove raw text, but KEEP raw_labels so our collator can binarize them on the fly!
    train_dataset = train_dataset.remove_columns(["text"])
    dev_dataset = dev_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])
    
    # Dynamic Data Collator: Converts string labels to dense float arrays ONLY for the 16 items in the current batch
    def custom_collate_fn(features):
        batch = {}
        batch["input_ids"] = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        batch["attention_mask"] = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
        if "raw_labels" in features[0]:
            raw_labels = [f["raw_labels"] for f in features]
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dense_labels = mlb.transform(raw_labels)
            batch["labels"] = torch.tensor(dense_labels, dtype=torch.float32)
        return batch
    
    # Initialize Model
    print(f"\nInitializing PLM-ICD Architecture with {model_name} backbone...")
    model = PLMICDModel(num_labels=num_labels, model_name=model_name)
    
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="no", 
        logging_strategy="epoch",
        report_to="none", 
        logging_dir=os.path.join(output_dir, "logs"),
        fp16=torch.cuda.is_available(), # Use Mixed Precision
        remove_unused_columns=False,    # PREVENT TRAINER FROM DELETING raw_labels!
        dataloader_num_workers=4        # SPEED UP BATCHING USING MULTIPLE CPU CORES
    )
    
    # Custom Optimizer: The pretrained backbone gets a small LR (5e-5) to avoid catastrophic forgetting,
    # but the randomly initialized LAAT head needs a massive LR (1e-3) to actually learn anything!
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    
    optimizer = AdamW([
        {'params': model.encoder.parameters(), 'lr': lr},          # Backbone LR (e.g. 5e-5)
        {'params': model.laat.parameters(), 'lr': 1e-3}            # LAAT Head LR
    ])
    
    # Total steps for the scheduler
    total_steps = (len(train_dataset) // batch_size) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    from transformers import TrainerCallback
    class PrintMetricsCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics:
                print(f"Epoch {state.epoch:.2f} | "
                      f"Val Loss: {metrics.get('eval_loss', 0):.4f} | "
                      f"mAP: {metrics.get('eval_mAP', 0):.4f} | "
                      f"Macro F1: {metrics.get('eval_macro_f1', 0):.4f} | "
                      f"Micro F1: {metrics.get('eval_micro_f1', 0):.4f} | "
                      f"Acc: {metrics.get('eval_accuracy', 0):.4f}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        data_collator=custom_collate_fn,
        callbacks=[PrintMetricsCallback()],
        optimizers=(optimizer, scheduler)
    )
    
    print("\nStarting Training Engine...")
    trainer.train()
    
    print("\nEvaluating final model on Dev Set...")
    dev_metrics = trainer.evaluate()
    
    print("\nEvaluating final model on Test Set...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    
    print("\n================ FINAL TEST SET RESULTS ================")
    print(f"Test Loss       : {test_metrics.get('eval_loss', 0):.4f}")
    print(f"Test mAP        : {test_metrics.get('eval_mAP', 0):.4f}")
    print(f"Test Macro F1   : {test_metrics.get('eval_macro_f1', 0):.4f}")
    print(f"Test Micro F1   : {test_metrics.get('eval_micro_f1', 0):.4f}")
    print(f"Test Accuracy   : {test_metrics.get('eval_accuracy', 0):.4f}")
    print("========================================================")
    
    print("\nSaving best final model...")
    final_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Save the metrics to a physical file so they can be viewed after the Kaggle run
    metrics_path = os.path.join(output_dir, "final_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "dev_metrics": dev_metrics,
            "test_metrics": test_metrics
        }, f, indent=4)
        
    print(f"Final Metrics saved to {metrics_path}")
    print("Done!")

if __name__ == "__main__":
    # Example execution
    train_classifier(
        model_name="vihealthbert",
        epochs=10,
        batch_size=16,
        lr=5e-5
    )
