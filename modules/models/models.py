import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, AutoModelForSequenceClassification
from typing import List, Union, Optional, Dict
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import pipeline
import pandas as pd

model_dict = {
    1: 'BAAI/bge-large-en-v1.5', # 768
    2: 'pritamdeka/S-PubMedBert-MS-MARCO', # 768
    3: 'ncbi/MedCPT-Query-Encoder', # 768
    4: 'NeuML/pubmedbert-base-embeddings' # 768
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

    def encode_text(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        if isinstance(texts, str): texts = [texts]
        if not texts: return np.array([])
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
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
            'accuracy': accuracy_score(labels, predictions)
        }
        
        # Searching for the best global threshold (using F1 Micro)
        best_f1 = 0
        best_threshold = 0.5
        
        for t in np.arange(0.05, 0.55, 0.05):
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

