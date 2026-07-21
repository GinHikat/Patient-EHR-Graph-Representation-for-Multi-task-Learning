import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
import warnings

warnings.filterwarnings("ignore")

LABEL_LIST = [
    "O",
    "B-Disease/Symptom",
    "I-Disease/Symptom",
    "B-Procedure/Treatment",
    "I-Procedure/Treatment",
    "B-Drug",
    "I-Drug"
]
LABEL_MAP = {label: i for i, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

def load_conll(file_path):
    sentences = []
    labels = []
    if not os.path.exists(file_path):
        return sentences, labels
    with open(file_path, 'r', encoding='utf-8') as f:
        current_words = []
        current_labels = []
        for line in f:
            line = line.strip()
            if not line:
                if current_words:
                    sentences.append(current_words)
                    labels.append(current_labels)
                    current_words = []
                    current_labels = []
                continue
            parts = line.split('\t') if '\t' in line else line.split(' ')
            if len(parts) >= 2:
                current_words.append(parts[0])
                current_labels.append(parts[-1])
        if current_words:
            sentences.append(current_words)
            labels.append(current_labels)
    return sentences, labels

class NERDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len=256):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sentences)
        
    def __getitem__(self, idx):
        words = self.sentences[idx]
        tags = self.labels[idx]
        
        input_ids = [self.tokenizer.cls_token_id]
        label_ids = [-100] # Ignore index for CLS token
        
        for word, tag in zip(words, tags):
            word_tokens = self.tokenizer.encode(word, add_special_tokens=False)
            if not word_tokens: continue
            
            input_ids.extend(word_tokens)
            
            # Apply the label to the FIRST subword only
            label_id = LABEL_MAP.get(tag, 0)
            label_ids.append(label_id)
            
            # The remaining subwords of this word get -100 so they don't affect the loss calculation
            label_ids.extend([-100] * (len(word_tokens) - 1))
            
        input_ids.append(self.tokenizer.sep_token_id)
        label_ids.append(-100)
        
        # Pad or Truncate
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            label_ids = label_ids[:self.max_len]
        else:
            padding_length = self.max_len - len(input_ids)
            input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
            label_ids.extend([-100] * padding_length)
            
        attention_mask = [1 if id != self.tokenizer.pad_token_id else 0 for id in input_ids]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [ID_TO_LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ID_TO_LABEL[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.gather(0, targets.masked_fill(targets == self.ignore_index, 0))
                focal_loss = alpha_t * focal_loss
            else:
                focal_loss = self.alpha * focal_loss

        mask = targets != self.ignore_index
        return focal_loss[mask].mean() if mask.any() else focal_loss.sum()

class FocalLossTrainer(Trainer):
    def __init__(self, *args, gamma=2.0, alpha=None, **kwargs):
        self.gamma = gamma
        self.alpha = alpha
        super().__init__(*args, **kwargs)
        
        if self.alpha is not None:
            self.alpha = torch.tensor(self.alpha, dtype=torch.float32).to(self.args.device)
            
        self.focal_loss = FocalLoss(gamma=self.gamma, alpha=self.alpha, ignore_index=-100)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        active_loss = inputs.get("attention_mask", None)
        num_labels = logits.shape[-1]
        if active_loss is not None:
            active_loss = active_loss.view(-1) == 1
            active_logits = logits.view(-1, num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
        else:
            active_logits = logits.view(-1, num_labels)
            active_labels = labels.view(-1)
            
        loss = self.focal_loss(active_logits, active_labels)
        return (loss, outputs) if return_outputs else loss

def main():
    model_name = "demdecuong/vihealthbert-base-word"
    output_dir = r"models\vihealthbert_ner"
    base_dir = r"data\viettel\vietnamese_ner\training\vietnamese\ner_train"
    
    print("Loading Data...")
    train_sentences, train_labels = load_conll(os.path.join(base_dir, "ner_train.conll"))
    dev_sentences, dev_labels = load_conll(os.path.join(base_dir, "ner_dev.conll"))
    test_sentences, test_labels = load_conll(os.path.join(base_dir, "ner_test.conll"))
    
    print(f"Train sentences: {len(train_sentences)}")
    print(f"Dev sentences: {len(dev_sentences)}")
    
    print("Loading Tokenizer & Model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        ignore_mismatched_sizes=True
    )
    
    train_dataset = NERDataset(train_sentences, train_labels, tokenizer, max_len=128)
    dev_dataset = NERDataset(dev_sentences, dev_labels, tokenizer, max_len=128)
    test_dataset = NERDataset(test_sentences, test_labels, tokenizer, max_len=128)
    
    print(f"GPUs Available: {torch.cuda.device_count()}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_strategy="epoch",   
        disable_tqdm=True,          
        save_total_limit=1,         # ONLY keep the single best checkpoint on disk, delete the rest!
        push_to_hub=False,
        report_to="none",
        fp16=True,                 
        dataloader_num_workers=2,  
    )
    
    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        gamma=2.0
    )
    
    print("\nStarting Training...")
    trainer.train()
    
    print("\nEvaluating on Test Set...")
    test_results = trainer.evaluate(test_dataset)
    print(test_results)
    
    print(f"\nSaving final model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("DONE!")

if __name__ == "__main__":
    main()
