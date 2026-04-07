import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from typing import List, Union, Optional, Dict

model_dict = {
    1: 'NeuML/pubmedbert-base-embeddings', # 768
    2: 'medicalai/ClinicalBERT', # 768
    3: 'emilyalsentzer/Bio_ClinicalBERT', # 768
    4: 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', # 768
    5: 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12', # 768
    6: 'bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16' # 1024
}

class Models:
    def __init__(self, model_choice: Union[int, str] = 1):
        self.token = self._load_key()
        if self.token:
            os.environ["HF_TOKEN"] = self.token
            
        model_name = model_dict.get(model_choice, model_choice) if isinstance(model_choice, int) else model_choice
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine model loading method (Use explicit BERT classes for older BlueBERT models)
        try:
            if "bluebert" in model_name.lower():
                self.tokenizer = BertTokenizer.from_pretrained(model_name, token=self.token)
                self.model = BertModel.from_pretrained(model_name, token=self.token).to(self.device)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.token, use_fast=False)
                self.model = AutoModel.from_pretrained(model_name, token=self.token).to(self.device)
        except Exception as e:
            # Final fallback to standard AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.token, use_fast=False)
            self.model = AutoModel.from_pretrained(model_name, token=self.token).to(self.device)

    def _load_key(self):
        for path in ['.env', '../.env', '../../.env', 'modules/.env']:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    for line in f:
                        if 'HUGGINGFACE_API_KEY' in line:
                            return line.split('=')[1].strip().strip('"').strip("'")
        return os.getenv("HUGGINGFACE_API_KEY")

    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str): texts = [texts]
        if not texts: return np.array([])
        
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()

    def get_models(self) -> Dict[int, str]:
        return model_dict
