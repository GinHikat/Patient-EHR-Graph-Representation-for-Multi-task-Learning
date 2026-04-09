import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from typing import List, Union, Optional, Dict

model_dict = {
    1: 'BAAI/bge-large-en-v1.5', # 768
    2: 'pritamdeka/S-PubMedBert-MS-MARCO', # 768
    3: 'ncbi/MedCPT-Query-Encoder', # 768
    4: 'NeuML/pubmedbert-base-embeddings' # 768
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

class ProcedureModel:
    def __init__(self):
        pass
