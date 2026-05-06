import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, AutoModelForSequenceClassification
from typing import List, Union, Optional, Dict
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

model_dict = {
    1: 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
    2: 'pritamdeka/S-PubMedBert-MS-MARCO', # 768
    3: 'ncbi/MedCPT-Query-Encoder', # 768
    4: 'NeuML/pubmedbert-base-embeddings', # 768
    5: 'BAAI/bge-base-en-v1.5' # 768
}

ner_model_dict = {
    1: 'pruas/BENT-PubMedBERT-NER-Disease',
    2: 'G-Med-NLP/BioLinkBERT-large-NER',
    3: 'Helios9/BioMed_NER'
}

class EmbeddingModels:
    def __init__(self, model_choice: Union[int, str] = 1, device: Optional[str] = None):
        self.token = self._load_key()
        if self.token:
            os.environ["HF_TOKEN"] = self.token
            
        model_name = model_dict.get(model_choice, model_choice) if isinstance(model_choice, int) else model_choice

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        
        try:
            if "bluebert" in model_name.lower():
                self.tokenizer = BertTokenizer.from_pretrained(model_name, token=self.token)
                self.model = BertModel.from_pretrained(model_name, token=self.token)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.token, use_fast=False)
                self.model = AutoModel.from_pretrained(model_name, token=self.token)
            
            try:
                self.model.to(self.device)
            except RuntimeError as device_error:
                if "cuda" in str(self.device).lower():
                    print(f"CUDA Error: {device_error}. Falling back to CPU.")
                    self.device = torch.device("cpu")
                    self.model.to(self.device)
                else:
                    raise device_error

        except Exception as e:
            print(f"Fallback Error: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.token, use_fast=False)
            self.model = AutoModel.from_pretrained(model_name, token=self.token)
            self.device = torch.device("cpu")
            self.model.to(self.device)

    def _load_key(self):
        for path in ['.env', '../.env', '../../.env', 'modules/.env']:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    for line in f:
                        if 'HUGGINGFACE_API_KEY' in line:
                            return line.split('=')[1].strip().strip('"').strip("'")
        return os.getenv("HUGGINGFACE_API_KEY")

    def encode_text(self, texts: Union[str, List[str]], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        if isinstance(texts, str): texts = [texts]
        if not texts: return np.array([])
        
        all_embeddings = []
        
        
        pbar = tqdm(range(0, len(texts), batch_size), desc='Encoding text', disable=not show_progress)
        for i in pbar:
            batch_texts = texts[i : i + batch_size]
            # Determine safe max length (default to 512 for most medical BERTs)
            tokenizer_max = self.tokenizer.model_max_length
            safe_max_length = tokenizer_max if tokenizer_max and tokenizer_max < 1e6 else 512
            
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=safe_max_length, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
            
        return np.concatenate(all_embeddings, axis=0) if all_embeddings else np.array([])

    def get_models(self) -> Dict[int, str]:
        return model_dict

class RadiologyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=1024):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }

class ProcedureModel:
    def __init__(self, num_labels: int, model_name: str = "yikuan8/Clinical-Longformer", device: Optional[str] = None):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        self.model.to(self.device)

    def compute_metrics(self, eval_preds, threshold: float = 0.5):
        logits, labels = eval_preds
        probs = 1 / (1 + np.exp(-logits))
        
        # Current fixed threshold metrics
        predictions = (probs > threshold).astype(int)
        
        metrics = {
            'f1_macro': f1_score(labels, predictions, average='macro', zero_division=0),
            'f1_micro': f1_score(labels, predictions, average='micro', zero_division=0),
            'precision_macro': precision_score(labels, predictions, average='macro', zero_division=0),
            'recall_macro': recall_score(labels, predictions, average='macro', zero_division=0),
            'max_p': np.max(probs),
            'avg_truth_p': np.mean(probs[labels == 1]) if np.any(labels == 1) else 0.0,
            'accuracy': accuracy_score(labels, predictions),
            'mAP': average_precision_score(labels, probs, average='macro')
        }

        # Add Top-K Metrics (k=5 and k=10)
        for k in [5, 10]:
            pk_list = []
            rk_list = []
            for i in range(len(labels)):
                # Get indices of top k predictions
                top_k_idx = np.argsort(probs[i])[-k:]
                true_idx = np.where(labels[i] == 1)[0]
                
                if len(true_idx) == 0:
                    continue
                
                # Hits: how many of our top k are in the ground truth
                hits = np.intersect1d(top_k_idx, true_idx)
                
                pk_list.append(len(hits) / k)
                rk_list.append(len(hits) / len(true_idx))
                
            metrics[f'precision@{k}'] = np.mean(pk_list) if pk_list else 0.0
            metrics[f'recall@{k}'] = np.mean(rk_list) if rk_list else 0.0

        try:
            metrics['roc_auc_macro'] = roc_auc_score(labels, probs, average='macro')
            metrics['roc_auc_micro'] = roc_auc_score(labels, probs, average='micro')
        except ValueError:
            metrics['roc_auc_macro'] = 0.0
            metrics['roc_auc_micro'] = 0.0
        
        # Searching for the best global threshold (using F1 Micro)
        best_f1 = 0
        best_threshold = 0.5
        
        # Expanded range to catch better thresholds for confident models
        for t in np.arange(0.05, 0.95, 0.05):
            t_preds = (probs > t).astype(int)
            f1 = f1_score(labels, t_preds, average='micro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        
        metrics['best_threshold'] = best_threshold
        metrics['best_f1_micro'] = best_f1
        
        return metrics

    def predict(self, texts: Union[str, List[str]], threshold: float = 0.5, batch_size: int = 8) -> np.ndarray:
        if isinstance(texts, str): texts = [texts]
        
        self.model.eval()
        all_preds = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.sigmoid(outputs.logits)
                preds = (probs > threshold).cpu().numpy().astype(int)
                all_preds.append(preds)
                
        return np.vstack(all_preds) if all_preds else np.array([])

class LAATLayer(nn.Module):
    """
    Label Attention Layer (LAAT) as described in the paper.
    Computes label-specific document representations using an attention mechanism.
    """
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        # Label query matrix (U in the paper)
        self.U = nn.Parameter(torch.zeros(num_labels, hidden_size))
        # Final weights for classification (W in the paper)
        self.W = nn.Parameter(torch.zeros(num_labels, hidden_size))
        # Final bias
        self.bias = nn.Parameter(torch.zeros(num_labels))
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.W)

    def forward(self, H):
        # H: Document hidden states [batch_size, seq_len, hidden_size]
        # U: Label query vectors [num_labels, hidden_size]
        
        # Prepare Query (U) for SDPA: [B, N, D]
        # We treat labels as queries and document states as keys/values
        Q = self.U.unsqueeze(0).expand(H.size(0), -1, -1)
        
        # Use SDPA for optimized attention computation.
        # scale=1.0 is used to match the original implementation's non-scaled dot product.
        # This will automatically use FlashAttention or memory-efficient kernels.
        V = torch.nn.functional.scaled_dot_product_attention(
            Q, H, H, 
            attn_mask=None, 
            dropout_p=0.0, 
            is_causal=False, 
            scale=1.0
        )
        
        # Final classification logit_n = v_n^T w_n + b_n
        logits = (V * self.W).sum(dim=-1) + self.bias # [B, N]
        
        return logits

class PLMICD_Internal(nn.Module):
    """
    Internal nn.Module for PLM-ICD.
    Combines a PLM encoder with a LAAT head.
    """
    def __init__(self, num_labels, model_name="yikuan8/Clinical-Longformer"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.laat = LAATLayer(self.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        H = outputs.last_hidden_state
        logits = self.laat(H)
        return SequenceClassifierOutput(logits=logits)

    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.encoder.gradient_checkpointing_disable()

    def save_pretrained(self, save_directory):
        self.encoder.save_pretrained(save_directory)
        self.encoder.config.save_pretrained(save_directory)

class PLMICDModel(ProcedureModel):
    """
    PLM-ICD Model wrapper compatible with ProcedureModel.
    Uses Clinical-Longformer and Label Attention (LAAT).
    """
    def __init__(self, num_labels: int, model_name: str = "yikuan8/Clinical-Longformer", device: Optional[str] = None):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Initialize the custom PLM-ICD architecture
        self.model = PLMICD_Internal(num_labels, model_name)
        self.model.to(self.device)

class MultiSynonymAttention(nn.Module):
    """
    Multi-Synonym Attention Layer (MSMN) as described in the paper.
    Uses multiple queries (synonyms) per label and aggregates their document representations.
    """
    def __init__(self, hidden_size, num_labels, num_synonyms=3):
        super().__init__()
        self.num_labels = num_labels
        self.num_synonyms = num_synonyms
        self.hidden_size = hidden_size
        
        # Multiple queries per label representing synonyms [N, M, D]
        self.Q = nn.Parameter(torch.zeros(num_labels, num_synonyms, hidden_size))
        
        # Biaffine weight matrix [D, D]
        self.W = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        # Final bias
        self.bias = nn.Parameter(torch.zeros(num_labels))
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.Q)
        nn.init.xavier_uniform_(self.W)

    def forward(self, H):
        # H: [B, L, D]
        B, L, D = H.shape
        N = self.num_labels
        M = self.num_synonyms
        
        # Flatten Q to treat each synonym as an independent query head: [B, N*M, D]
        Q_flat = self.Q.view(1, N * M, D).expand(B, -1, -1)
        
        # Optimized computation using SDPA
        # This replaces manual scores calculation and softmax
        V_all_flat = torch.nn.functional.scaled_dot_product_attention(
            Q_flat, H, H,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=1.0
        ) # [B, N*M, D]
        
        V_all = V_all_flat.view(B, N, M, D)
        
        # Aggregation: Max-pooling across synonyms to capture the most relevant match
        V, _ = torch.max(V_all, dim=2) # [B, N, D]
        
        # Aggregate synonym queries themselves (mean-pooling)
        Q_agg = self.Q.mean(dim=1) # [N, D]
        
        # Biaffine Transformation: logit_n = v_n^T W q_n + b_n
        # First compute V * W -> [B, N, D]
        VW = torch.matmul(V, self.W)
        # Then (VW) dot Q_agg + bias -> [B, N]
        logits = (VW * Q_agg).sum(dim=-1) + self.bias
        
        return logits

class MSMN_Internal(nn.Module):
    """
    Internal nn.Module for MSMN.
    Combines a PLM encoder with a Multi-Synonym Attention head.
    """
    def __init__(self, num_labels, model_name="yikuan8/Clinical-Longformer", num_synonyms=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.msa = MultiSynonymAttention(self.hidden_size, num_labels, num_synonyms)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        H = outputs.last_hidden_state
        logits = self.msa(H)
        return SequenceClassifierOutput(logits=logits)

    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.encoder.gradient_checkpointing_disable()

    def save_pretrained(self, save_directory):
        self.encoder.save_pretrained(save_directory)
        self.encoder.config.save_pretrained(save_directory)

class MSMNModel(ProcedureModel):
    """
    MSMN Model wrapper compatible with ProcedureModel.
    Uses Clinical-Longformer and Multi-Synonym Attention with Biaffine head.
    """
    def __init__(self, num_labels: int, model_name: str = "yikuan8/Clinical-Longformer", device: Optional[str] = None, num_synonyms: int = 3):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Initialize the custom MSMN architecture
        self.model = MSMN_Internal(num_labels, model_name, num_synonyms=num_synonyms)
        self.model.to(self.device)

class NERModel:
    def __init__(self, model_choice: Union[int, str] = 1, device: Optional[str] = None):
        model_name = ner_model_dict.get(model_choice, model_choice) if isinstance(model_choice, int) else model_choice
        
        if device:
            # transformers pipeline uses 0, 1, etc for CUDA or -1 for CPU
            self.device_idx = 0 if "cuda" in str(device).lower() else -1
        else:
            self.device_idx = 0 if torch.cuda.is_available() else -1
            
        print(f"Loading NER model: {model_name} on device: {'cuda' if self.device_idx >= 0 else 'cpu'}")
        
        self.pipeline = pipeline(
            "ner", 
            model=model_name, 
            aggregation_strategy="simple", 
            device=self.device_idx
        )

    def predict(self, text: Union[str, List[str]]) -> Union[List[Dict], List[List[Dict]]]:
        """
        Extracts clinical entities (Disease, Drug, etc.) from the given text.
        """
        if not text:
            return []
        
        return self.pipeline(text)

    def predict_df(self, text: Union[str, List[str]]) -> pd.DataFrame:
        """
        Converts the NER prediction results into a pandas DataFrame and merges consecutive tokens.
        """
        results = self.predict(text)
        if not results:
            return pd.DataFrame(columns=['text', 'start', 'end', 'score', 'group'])

        # Handle batch results (list of lists)
        if len(results) > 0 and isinstance(results[0], list):
            data = [item for sublist in results for item in sublist]
        else:
            data = results

        if not data:
            return pd.DataFrame(columns=['text', 'start', 'end', 'score', 'group'])

        # Merge logic: if prev.end == next.start AND groups match, concat them
        merged_data = []
        current = data[0].copy()
        
        for i in range(1, len(data)):
            next_entry = data[i]
            # Match boundary AND entity group to avoid merging separate entities like 'HCV' and 'cirrhosis'
            if (current['end'] == next_entry['start'] and 
                current.get('entity_group') == next_entry.get('entity_group')):
                
                next_word = next_entry['word']
                # Handle BERT-style subword markers
                if next_word.startswith("##"):
                    next_word = next_word[2:]
                
                current['word'] += next_word
                current['end'] = next_entry['end']
                current['score'] = (current['score'] + next_entry['score']) / 2
            else:
                merged_data.append(current)
                current = next_entry.copy()
        merged_data.append(current)

        df = pd.DataFrame(merged_data)

        # Map pipeline keys to requested column names
        column_mapping = {
            'word': 'text',
            'entity_group': 'group',
            'start': 'start',
            'end': 'end',
            'score': 'score'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure all requested columns exist
        for col in ['text', 'start', 'end', 'score', 'group']:
            if col not in df.columns:
                df[col] = None
        
        return df[['text', 'score', 'group']]

    def get_models(self) -> Dict[int, str]:
        return ner_model_dict

