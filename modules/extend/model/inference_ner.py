import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

TOKENIZER_MAP = {
    "vihealthbert": "vihealthbert",
    "vipubmed-deberta": "vipubmed-deberta-base",
    "phobert": "phobert",
    "xlm-roberta": "xlm-roberta-base"
}

LABEL_LIST = [
    "O",
    "B-Disease/Symptom", "I-Disease/Symptom",
    "B-Procedure/Treatment", "I-Procedure/Treatment",
    "B-Drug", "I-Drug"
]

# Cache to prevent reloading models if function is called repeatedly
_PIPELINES = {}

class NER:
    def __init__(self, model_name):
        self.model_name = model_name.lower()

    def extract_entities(self, text: str) -> list:
        """
        Extracts medical entities from text using the specified fine-tuned model.
        """
        global _PIPELINES
    
        if self.model_name not in _PIPELINES:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "statedict", self.model_name)
        
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model weights not found at: {model_path}")

            # Load Tokenizer & Model from the local folder
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            id2label = {i: label for i, label in enumerate(LABEL_LIST)}
            label2id = {label: i for i, label in enumerate(LABEL_LIST)}

            model = AutoModelForTokenClassification.from_pretrained(
                model_path,
                num_labels=len(LABEL_LIST),
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True
            )
        
            device = 0 if torch.cuda.is_available() else -1
        
            # aggregation_strategy="simple" automatically merges subwords (B/I tags)
            _PIPELINES[self.model_name] = pipeline(
                "ner", 
                model=model, 
                tokenizer=tokenizer, 
                device=device,
                aggregation_strategy="simple" 
            )
        
        raw_results = _PIPELINES[self.model_name](text)
        entities = []
        
        # Chronological pointer to search through the original text
        current_search_idx = 0
    
        for entity in raw_results:
            label = entity['entity_group']
            if label == "O":
                continue
            
            term = entity['word'].strip()
            
            start_idx = text.find(term, current_search_idx)
            
            if start_idx != -1:
                end_idx = start_idx + len(term)
                offset = (start_idx, end_idx)
                current_search_idx = end_idx  # Move pointer forward
            else:
                # Fallback if subword merging aggressively changed the string
                offset = (None, None)
            
            entities.append({
                "term": term,
                "offset": offset,
                "label": label
            })
        
        return entities
