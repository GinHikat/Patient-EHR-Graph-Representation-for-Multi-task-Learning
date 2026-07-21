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

LABEL_LIST_VI = [
    "O",
    "B-Disease/Symptom", "I-Disease/Symptom",
    "B-Procedure/Treatment", "I-Procedure/Treatment",
    "B-Drug", "I-Drug"
]

LABEL_LIST_EN = [
    "O",
    "B-Disease", "I-Disease",
    "B-Chemical", "I-Chemical"
]

# Cache to prevent reloading models if function is called repeatedly
_PIPELINES = {}

class NER:
    def __init__(self, mode='vietnamese', model_name='phobert'):
        self.mode = mode.lower()
        self.model_name = model_name.lower()
        
        if self.mode == 'english' and self.model_name not in ['sapbert', 'pubmedbert', 'biobert']:
            raise ValueError("For English mode, model_name must be one of: 'sapbert', 'pubmedbert', 'biobert'")

    def extract_entities(self, text: str) -> list:
        """
        Extracts medical entities from text using the specified fine-tuned model.
        """
        global _PIPELINES
        
        pipeline_key = f"{self.mode}_{self.model_name}"
    
        if pipeline_key not in _PIPELINES:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            statedict_dir = os.path.join(base_dir, "..", "statedict", "ner")
            if self.mode == 'vietnamese':
                model_path = os.path.join(statedict_dir, self.model_name)
                label_list = LABEL_LIST_VI
            else:
                model_path = os.path.join(statedict_dir, "eng", f"{self.model_name}")
                label_list = LABEL_LIST_EN
        
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model weights not found at: {model_path}")

            # Load Tokenizer & Model from the local folder
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            id2label = {i: label for i, label in enumerate(label_list)}
            label2id = {label: i for i, label in enumerate(label_list)}

            model = AutoModelForTokenClassification.from_pretrained(
                model_path,
                num_labels=len(label_list),
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True
            )
        
            device = 0 if torch.cuda.is_available() else -1
        
            # aggregation_strategy="simple" automatically merges subwords (B/I tags)
            _PIPELINES[pipeline_key] = pipeline(
                "ner", 
                model=model, 
                tokenizer=tokenizer, 
                device=device,
                aggregation_strategy="simple" 
            )
        
        # Chunk text to avoid exceeding max sequence length (typically 512 tokens)
        # which causes CUDA device-side assert errors
        raw_results = []
        for line in text.split('\n'):
            if not line.strip(): 
                continue
            
            line_chunks = []
            while len(line) > 1500:
                split_idx = line.rfind(' ', 0, 1500)
                if split_idx == -1: 
                    split_idx = 1500
                line_chunks.append(line[:split_idx])
                line = line[split_idx:]
            if line:
                line_chunks.append(line)
                
            for chunk in line_chunks:
                if chunk.strip():
                    try:
                        raw_results.extend(_PIPELINES[pipeline_key](chunk))
                    except Exception as e:
                        print(f"Skipping a chunk due to inference error: {e}")
        entities = []
        
        # Chronological pointer to search through the original text
        current_search_idx = 0
    
        for entity in raw_results:
            label = entity['entity_group']
            if label == "O":
                continue
            
            term = entity['word'].strip().replace('@@', '').replace('##', '')
            
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
            
        merged_entities = []
        for ent in entities:
            if not merged_entities:
                merged_entities.append(ent)
                continue
                
            prev_ent = merged_entities[-1]
            if prev_ent['label'] == ent['label']:
                p_start, p_end = prev_ent['offset']
                c_start, c_end = ent['offset']
                
                # Check if subsequent place (adjacent or separated by a single space)
                if p_end is not None and c_start is not None and (c_start - p_end) <= 1:
                    if p_start is not None and c_end is not None:
                        new_term = text[p_start:c_end]
                    else:
                        sep = " " if (c_start - p_end) == 1 else ""
                        new_term = prev_ent['term'] + sep + ent['term']
                        
                    prev_ent['term'] = new_term
                    prev_ent['offset'] = (p_start, c_end)
                else:
                    merged_entities.append(ent)
            else:
                merged_entities.append(ent)
                
        import string
        
        # Post-processing: Clean up terms and remove short entities
        final_entities = []
        for ent in merged_entities:
            clean_term = ent['term'].strip(string.punctuation + " \t\n\r")
            if len(clean_term) >= 3:
                ent['term'] = clean_term
                final_entities.append(ent)
        
        return final_entities
