import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json
import os
import itertools
from .inference_ner import NER

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

    def forward(self, input_ids, attention_mask, span_pairs, pair_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  
        
        batch_size, max_pairs, _ = span_pairs.shape
        combined_vectors = []
        
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
                
        if not combined_vectors:
            return torch.zeros((1, self.classifier[-1].out_features), device=input_ids.device)
            
        combined_vectors = torch.stack(combined_vectors) 
        logits = self.classifier(combined_vectors)
        return logits

class RelationExtractor:
    def __init__(self, model_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Label Map
        with open(os.path.join(model_dir, "label_map.json"), "r") as f:
            self.label2id = json.load(f)
        self.id2label = {int(v): k for k, v in self.label2id.items()}
        
        # Load Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
        self.model = SpanPairREModel(num_labels=len(self.label2id))
        
        state_dict_path = os.path.join(model_dir, "pytorch_model.bin")
        self.model.load_state_dict(torch.load(state_dict_path, map_location=self.device, weights_only=True))
        
        self.model.to(self.device)
        self.model.eval()
        print("Relation Extraction Model Loaded Successfully!")

    def _get_token_span(self, char_start, char_end, offset_mapping):
        tok_start, tok_end = None, None
        
        for token_idx, (start, end) in enumerate(offset_mapping):
            if start == end: # Skip special tokens (0,0)
                continue
                
            # First token that ends after our character start
            if tok_start is None and end > char_start:
                tok_start = token_idx
                
            # Keep updating tok_end for any token that starts before our character end
            if start < char_end:
                tok_end = token_idx
                
        return tok_start, tok_end

    def predict(self, text, pairs):
        """
        pairs: list of dicts, e.g., [{"head_span": (start, end), "target_span": (start, end)}]
        where (start, end) are character indices in the raw text.
        """
        encoded = self.tokenizer(
            text,
            max_length=256,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        offset_mapping = encoded["offset_mapping"][0].tolist()
        
        valid_pairs_tensor = []
        
        for pair in pairs:
            h_char_s, h_char_e = pair["head_span"]
            t_char_s, t_char_e = pair["target_span"]
            
            # Map char spans robustly to token spans
            h_tok_s, h_tok_e = self._get_token_span(h_char_s, h_char_e, offset_mapping)
            t_tok_s, t_tok_e = self._get_token_span(t_char_s, t_char_e, offset_mapping)
            
            if None in (h_tok_s, h_tok_e, t_tok_s, t_tok_e):
                continue
                
            valid_pairs_tensor.append([h_tok_s, h_tok_e, t_tok_s, t_tok_e])
            
        if not valid_pairs_tensor:
            return [] # No valid pairs within tokenization limits
            
        # Format for model (Batch Size = 1)
        span_pairs_tensor = torch.tensor([valid_pairs_tensor], dtype=torch.long).to(self.device)
        pair_mask = torch.ones((1, len(valid_pairs_tensor)), dtype=torch.bool).to(self.device)
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, span_pairs_tensor, pair_mask)
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            
        results = []
        for i, pred_id in enumerate(preds):
            results.append({
                "pair": pairs[i],
                "predicted_relation": self.id2label[pred_id]
            })
            
        return results

class NEREPipeline:
    def __init__(self, ner_model_name='sapbert', re_model_dir=None):
        print("Initializing End-to-End Pipeline...")
        print("Loading NER Model...")
        self.ner = NER(mode='english', model_name=ner_model_name)
        
        print("Loading Relation Extraction Model...")
        if re_model_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            re_model_dir = os.path.join(base_dir, "..", "statedict", "re", "english")
        
        self.re = RelationExtractor(re_model_dir)
        print("Pipeline Initialized Successfully!\n")

    def run(self, text):
        print('='*20)
        print(f"Input Text: '{text}'")
        print('='*20)
        entities = self.ner.extract_entities(text)
        print('='*20)
        print("Found Entities:")
        for ent in entities:
            print(f"   - {ent['term']} [{ent['label']}] at offset {ent['offset']}")
            
        valid_entities = [e for e in entities if e['offset'][0] is not None and e['offset'][1] is not None and e['term'].strip('.,;:!?()"\'') != '']
        
        pairs_to_predict = []
        for head, target in itertools.permutations(valid_entities, 2):
            if head['label'] in ["Chemical", "Drug"] and target['label'] in ["Disease", "Symptom"]:
                pairs_to_predict.append({
                    "head_span": head['offset'],
                    "target_span": target['offset'],
                    "head_term": head['term'],
                    "target_term": target['term']
                })
            
        if not pairs_to_predict:
            print("\nNot enough entities found to form valid relation pairs (needs 1 Drug and 1 Disease).")
            return
            
        print(f"\nFormed {len(pairs_to_predict)} candidate pairs. Running Relation Extraction...")
        results = self.re.predict(text, pairs_to_predict)
        
        print("\nPredicted Relations (Filtered):")
        filtered_results = []
        # Keep track of unique pairs to prevent duplicate printing if there are duplicates
        seen_pairs = set()
        
        for res in results:
            rel = res["predicted_relation"]
            if rel != "None":
                h_term = res["pair"]["head_term"]
                t_term = res["pair"]["target_term"]
                pair_key = (h_term, rel, t_term)
                
                if pair_key not in seen_pairs:
                    print(f"   [{h_term}]  ===({rel})===>  [{t_term}]")
                    seen_pairs.add(pair_key)
                    filtered_results.append(res)
        
        if not filtered_results:
            print("   No meaningful relations ('treat' or 'cause') found.")
            
        return filtered_results

if __name__ == "__main__":
    pipeline = NEREPipeline(ner_model_name='sapbert')
    sample_text = (
        # "The patient is prescribed with Aspirin and Metformin, which is used to treat Hypertension. However, the patient is reported to be suffering from diarrhea, nausea, and vomiting, which are documented as potential side effect of the drugs"
        "A 65-year-old male with a history of persistent atrial fibrillation and symptomatic heart failure was started on amiodarone and carvedilol. While carvedilol effectively managed his heart failure, the amiodarone induced severe pulmonary toxicity and thyroid dysfunction. Consequently, levothyroxine was prescribed to resolve the hypothyroidism, and amiodarone was halted to prevent further respiratory decline."
    )
    pipeline.run(sample_text)
