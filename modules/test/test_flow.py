import json
import torch
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from modules.models.models import EmbeddingModels

import sys

class Logger(object):
    def __init__(self, filename="test_flow_output.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    sys.stdout = Logger("data/VN-Clinical-Text/test_flow_output.txt")
    model_id = "PeterPaker123/Qwen2.5-7B-Vietnamese-Medical-NER"
    input_file = "data/VN-Clinical-Text/gold_standard_benchmark.jsonl"
    
    print(f"Loading tokenizer and model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    tokenizer.chat_template = chat_template
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )
    
    # Load SapBERT for entity linking
    print("Loading SapBERT model for entity linking...")
    sapbert_model_name = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
    embedder = EmbeddingModels(model_choice=sapbert_model_name)
    
    # Load Base Entities DataFrame
    base_df = pd.read_pickle("data/VN-Clinical-Text/mapped_entities_embedded.pkl")
    base_df = base_df.dropna(subset=['mapped_cui']).reset_index(drop=True)
    base_embeddings = np.vstack(base_df['embedding'].values)
    
    # Label mapping dictionaries
    label_mapping = {
        "SYMPTOM_AND_DISEASE": "Disease/Symptom",
        "Symptom": "Disease/Symptom",
        "Disease": "Disease/Symptom",
        "Symptom & Disease": "Disease/Symptom",
        "DRUG": "Drug",
        "Drug": "Drug",
        "MEDICAL_PROCEDURE": "Procedure/Treatment",
        "Procedure": "Procedure/Treatment",
        "Diagnostic Procedure": "Procedure/Treatment",
    }
    
    umls_to_three_classes = {
        "Clinical Attribute": "Disease/Symptom",
        "Disease or Syndrome": "Disease/Symptom",
        "Finding": "Disease/Symptom",
        "Injury or Poisoning": "Disease/Symptom",
        "Mental or Behavioral Dysfunction": "Disease/Symptom",
        "Neoplastic Process": "Disease/Symptom",
        "Pathologic Function": "Disease/Symptom",
        "Physiologic Function": "Disease/Symptom",
        "Sign or Symptom": "Disease/Symptom",
        "Amino Acid, Peptide, or Protein": "Drug",
        "Biologically Active Substance": "Drug",
        "Clinical Drug": "Drug",
        "Immunologic Factor": "Drug",
        "Indicator, Reagent, or Diagnostic Aid": "Drug",
        "Nucleic Acid, Nucleoside, or Nucleotide": "Drug",
        "Organic Chemical": "Drug",
        "Pharmacologic Substance": "Drug",
        "Vitamin": "Drug",
        "Diagnostic Procedure": "Procedure/Treatment",
        "Health Care Activity": "Procedure/Treatment",
        "Laboratory Procedure": "Procedure/Treatment",
        "Laboratory or Test Result": "Procedure/Treatment",
        "Medical Device": "Procedure/Treatment",
        "Therapeutic or Preventive Procedure": "Procedure/Treatment",
    }
    
    base_df['macro_type'] = base_df['mapped_type'].map(umls_to_three_classes)
    
    # Read the dataset and find sentences with all 3 types
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    test_lines = []
    for line in lines:
        data = json.loads(line)
        gold = data.get("gold_entities", {})
        if gold.get("Disease/Symptom") and gold.get("Procedure/Treatment") and gold.get("Drug"):
            test_lines.append(line)
            if len(test_lines) == 2:
                break
                
    # If we couldn't find 2 sentences with all 3, just grab the first 2 that have the most variety
    if len(test_lines) < 2:
        test_lines = lines[:2]
        
    print(f"\nFound {len(test_lines)} sample sentences to test.")
    
    system_prompt = (
        "You are a medical expert. Your task is to identify and extract medical entities from the given text.\n"
        "The entity types to extract are:\n"
        "- SYMPTOM_AND_DISEASE: Symptoms, signs of illness, diseases, or medical conditions.\n"
        "- MEDICAL_PROCEDURE: Medical procedures, surgeries, therapies, or diagnostic methods.\n"
        "- DRUG: Names of medicines, drugs, vitamins, or supplements.\n"
        "Return the result as a JSON list containing dictionaries with \"entity\" and \"type\" keys. "
        "If no relevant entities are found, return an empty list []"
    )
    
    for i, line in enumerate(test_lines):
        data = json.loads(line)
        text = data.get("text", "")
        
        print("\n" + "="*80)
        print(f"SAMPLE {i+1}")
        print("="*80)
        print(f"INPUT TEXT:\n{text}")
        print("-" * 80)
        
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Text: {text}"}
        ]
        
        inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True, return_dict=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        
        predicted_entities = []
        try:
            cleaned_response = response.replace("```json", "").replace("```", "")
            predicted_entities = json.loads(cleaned_response.strip())
        except json.JSONDecodeError:
            pass
            
        print("EXTRACTED RESULTS (Qwen):")
        print(json.dumps(predicted_entities, indent=2, ensure_ascii=False))
        print("-" * 80)
        
        linked_entities = {
            "Disease/Symptom": [],
            "Procedure/Treatment": [],
            "Drug": []
        }
        
        print("MAPPED ENTITIES (SapBERT):")
        
        metrics = {
            "Disease/Symptom": {"tp": 0, "fp": 0, "fn": 0},
            "Procedure/Treatment": {"tp": 0, "fp": 0, "fn": 0},
            "Drug": {"tp": 0, "fp": 0, "fn": 0}
        }
        
        if predicted_entities:
            extracted_texts = [ent.get("entity", "") for ent in predicted_entities]
            extracted_embeddings = embedder.encode_text(extracted_texts, batch_size=32, show_progress=False)
            
            for idx, ent in enumerate(predicted_entities):
                raw_type = ent.get("type", "")
                mapped_type = label_mapping.get(raw_type)
                
                if not mapped_type:
                    continue
                    
                emb = extracted_embeddings[idx:idx+1]
                
                mask = base_df['macro_type'] == mapped_type
                if not mask.any():
                    continue
                    
                type_df = base_df[mask]
                type_embeddings = np.vstack(type_df['embedding'].values)
                
                sims = cosine_similarity(emb, type_embeddings)[0]
                max_idx = np.argmax(sims)
                max_sim = sims[max_idx]
                
                matched_term = type_df.iloc[max_idx]['entity']
                cui = type_df.iloc[max_idx]['mapped_cui']
                original_umls_type = type_df.iloc[max_idx]['mapped_type']
                
                if max_sim > 0.6:
                    print(f"✓ ACCEPTED (Score: {max_sim:.4f})")
                    print(f"  Extracted:  '{ent.get('entity')}' [{raw_type}]")
                    print(f"  Matched To: '{matched_term}'")
                    print(f"  CUI:        {cui}")
                    print(f"  UMLS Type:  {original_umls_type} -> {mapped_type}\n")
                    if pd.notna(cui) and cui != "":
                        linked_entities[mapped_type].append(str(cui))
                else:
                    print(f"✗ REJECTED (Score: {max_sim:.4f} < 0.60)")
                    print(f"  Extracted:  '{ent.get('entity')}' [{raw_type}]")
                    print(f"  Best Match: '{matched_term}' ({cui})\n")
                        
        linked_entities["Disease/Symptom"] = list(set(linked_entities["Disease/Symptom"]))
        linked_entities["Procedure/Treatment"] = list(set(linked_entities["Procedure/Treatment"]))
        linked_entities["Drug"] = list(set(linked_entities["Drug"]))
        
        gold_entities = data.get('gold_entities', {})
        gold_sets = {
            "Disease/Symptom": set(gold_entities.get("Disease/Symptom", [])),
            "Procedure/Treatment": set(gold_entities.get("Procedure/Treatment", [])),
            "Drug": set(gold_entities.get("Drug", []))
        }
        pred_sets = {
            "Disease/Symptom": set(linked_entities["Disease/Symptom"]),
            "Procedure/Treatment": set(linked_entities["Procedure/Treatment"]),
            "Drug": set(linked_entities["Drug"])
        }
        
        for cls in ["Disease/Symptom", "Procedure/Treatment", "Drug"]:
            gold = gold_sets[cls]
            pred = pred_sets[cls]
            metrics[cls]["tp"] += len(gold.intersection(pred))
            metrics[cls]["fp"] += len(pred - gold)
            metrics[cls]["fn"] += len(gold - pred)
            
        print("-" * 80)
        print("EVALUATION FOR THIS SENTENCE:")
        for cls in ["Disease/Symptom", "Procedure/Treatment", "Drug"]:
            tp = metrics[cls]["tp"]
            fp = metrics[cls]["fp"]
            fn = metrics[cls]["fn"]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            print(f"[{cls}]")
            print(f"  Gold CUIs:      {list(gold_sets[cls])}")
            print(f"  Predicted CUIs: {list(pred_sets[cls])}")
            print(f"  Metrics:        Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
            print(f"  (TP: {tp}, FP: {fp}, FN: {fn})\n")
            
if __name__ == "__main__":
    main()
