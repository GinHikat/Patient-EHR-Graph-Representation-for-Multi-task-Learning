import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import json
import time
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class SpanPairREDataset(Dataset):
    def __init__(self, csv_path, tokenizer_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext", max_length=256):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        self.unique_labels = set()
        for _, row in self.data.iterrows():
            relations = json.loads(row['relations_json'])
            for rel in relations:
                self.unique_labels.add(rel[2])
                
        self.label2id = {label: i for i, label in enumerate(sorted(list(self.unique_labels)))}
        self.id2label = {i: label for label, i in self.label2id.items()}

    def __len__(self):
        return len(self.data)

    def _char_to_token_idx(self, char_idx, offset_mapping):
        for token_idx, (start, end) in enumerate(offset_mapping):
            if start <= char_idx < end:
                return token_idx
        return None

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['input']
        relations = json.loads(row['relations_json'])
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        offset_mapping = encoded["offset_mapping"][0].tolist()
        
        span_pairs = []
        labels = []
        
        for rel in relations:
            head_char_start, head_char_end = rel[0]
            target_char_start, target_char_end = rel[1]
            label_str = rel[2]
            
            head_tok_start = self._char_to_token_idx(head_char_start, offset_mapping)
            head_tok_end = self._char_to_token_idx(head_char_end - 1, offset_mapping)
            
            target_tok_start = self._char_to_token_idx(target_char_start, offset_mapping)
            target_tok_end = self._char_to_token_idx(target_char_end - 1, offset_mapping)
            
            if None in (head_tok_start, head_tok_end, target_tok_start, target_tok_end):
                continue
                
            span_pairs.append({
                "head_span": (head_tok_start, head_tok_end),
                "target_span": (target_tok_start, target_tok_end)
            })
            labels.append(self.label2id[label_str])
            
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "span_pairs": span_pairs,
            "labels": labels
        }

def span_pair_collate(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    
    max_pairs = max([len(item["span_pairs"]) for item in batch])
    if max_pairs == 0:
        max_pairs = 1 
        
    batch_size = len(batch)
    span_pairs_tensor = torch.zeros((batch_size, max_pairs, 4), dtype=torch.long)
    labels_tensor = torch.zeros((batch_size, max_pairs), dtype=torch.long)
    pair_mask = torch.zeros((batch_size, max_pairs), dtype=torch.bool)
    
    for b_idx, item in enumerate(batch):
        for p_idx, (span_pair, label) in enumerate(zip(item["span_pairs"], item["labels"])):
            span_pairs_tensor[b_idx, p_idx, 0] = span_pair["head_span"][0]
            span_pairs_tensor[b_idx, p_idx, 1] = span_pair["head_span"][1]
            span_pairs_tensor[b_idx, p_idx, 2] = span_pair["target_span"][0]
            span_pairs_tensor[b_idx, p_idx, 3] = span_pair["target_span"][1]
            labels_tensor[b_idx, p_idx] = label
            pair_mask[b_idx, p_idx] = True
            
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "span_pairs": span_pairs_tensor,
        "pair_mask": pair_mask,
        "labels": labels_tensor
    }

class SpanPairREModel(nn.Module):
    def __init__(self, num_labels, model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"):
        super(SpanPairREModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.encoder.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_labels)
        )

    def _mean_pooling(self, hidden_states, start_idx, end_idx):
        span_embeddings = hidden_states[start_idx:end_idx+1]
        return span_embeddings.mean(dim=0)

    def forward(self, input_ids, attention_mask, span_pairs, pair_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  
        
        batch_size, max_pairs, _ = span_pairs.shape
        
        combined_vectors = []
        valid_labels = []
        
        for b in range(batch_size):
            for p in range(max_pairs):
                if not pair_mask[b, p]:
                    continue 
                
                h_start, h_end, t_start, t_end = span_pairs[b, p]
                
                head_emb = self._mean_pooling(sequence_output[b], h_start, h_end)
                target_emb = self._mean_pooling(sequence_output[b], t_start, t_end)
                interaction_emb = head_emb * target_emb
                
                combined = torch.cat([head_emb, target_emb, interaction_emb], dim=-1)
                combined_vectors.append(combined)
                
                if labels is not None:
                    valid_labels.append(labels[b, p])
                    
        if not combined_vectors:
            logits = torch.zeros((1, self.classifier[-1].out_features), device=input_ids.device)
            if labels is not None:
                loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
                valid_labels_tensor = torch.zeros(1, dtype=torch.long, device=input_ids.device)
                return (loss, logits, valid_labels_tensor)
            return logits
            
        combined_vectors = torch.stack(combined_vectors) 
        logits = self.classifier(combined_vectors)
        
        if labels is not None:
            valid_labels = torch.stack(valid_labels)
            loss_fct = FocalLoss(gamma=2.0)
            loss = loss_fct(logits, valid_labels)
            return (loss, logits, valid_labels)
            
        return logits

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            span_pairs = batch["span_pairs"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask, span_pairs, pair_mask, labels=labels)
            loss, logits, valid_labels = outputs[0], outputs[1], outputs[2]
            
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
                
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(valid_labels.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    if len(all_labels) == 0:
        return avg_loss, 0.0, 0.0, 0.0
        
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, precision, recall, f1

def train_model():
    from torch.utils.data import random_split
    import os
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..', '..', '..', '..'))
    csv_file = os.path.join(project_root, 'data', 'viettel', 'vietnamese_ner', 'training', 'english', 'unified_re_indices_with_negatives.csv')
    
    dataset = SpanPairREDataset(csv_file, tokenizer_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext", max_length=256)
    
    # 80/10/10 train/val/test split
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    batch_size = 16 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=span_pair_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=span_pair_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=span_pair_collate)
    
    model = SpanPairREModel(num_labels=len(dataset.unique_labels))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    epochs = 10
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(train_loop):
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            span_pairs = batch["span_pairs"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask, span_pairs, pair_mask, labels=labels)
            loss = outputs[0]
            
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if step % 10 == 0:
                train_loop.set_postfix(loss=f"{loss.item():.4f}")
                
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation Step
        val_loss, val_p, val_r, val_f1 = evaluate_model(model, val_loader, device)
        
        elapsed = time.time() - start_time
        print(f"-> Epoch {epoch+1} Completed in {elapsed:.2f}s")
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Precision: {val_p:.4f} | Val Recall: {val_r:.4f} | Val F1: {val_f1:.4f}\n")

    #  Final Test Set Evaluation 
    print("\nEvaluating on Test Set...")
    test_loss, test_p, test_r, test_f1 = evaluate_model(model, test_loader, device)
    print(f" FINAL TEST SET METRICS ")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Precision (Macro): {test_p:.4f}")
    print(f"Test Recall (Macro): {test_r:.4f}")
    print(f"Test F1 Score (Macro): {test_f1:.4f}\n")

    #  Save Model Checkpoint 
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, '..', 'saved_re_model')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Saving model checkpoint to {save_dir}...")
    # Extract model from DataParallel if necessary
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
    
    # Save the label map for inference
    with open(os.path.join(save_dir, "label_map.json"), "w") as f:
        json.dump(dataset.label2id, f)
        
    print("Model and label map saved successfully!")

if __name__ == "__main__":
    train_model()
