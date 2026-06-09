import os
import sys
import json
import pandas as pd
from tqdm import tqdm
import unicodedata
import re

project_root = r"d:\Study\Education\Projects\Thesis"
if project_root not in sys.path:
    sys.path.append(project_root)

from modules.dataset_preprocessing.external.uml import spacy_quickumls

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("_", " ")
    text = text.strip(" .,;!?\"'()[]{}")
    text = re.sub(r'\s+', ' ', text)
    return text

def clean_dataset_bruteforce():
    base_dir = r"d:\Study\Education\Projects\Thesis\data\viettel\vietnamese_ner"
    # We will use the already partially cleaned one since the original was removed.
    input_jsonl = os.path.join(base_dir, "qwen_dataset.jsonl")
    output_jsonl = os.path.join(base_dir, "cleaned_qwen_dataset.jsonl")
    
    drug_semantic_types = {
        "Pharmacologic Substance", "Organic Chemical", "Clinical Drug", 
        "Antibiotic", "Biomedical or Dental Material", "Hormone",
        "Vitamin", "Immunologic Factor"
    }
    
    # Dictionary to cache QuickUMLS results so we don't scan the same word twice
    # mapping: cleaned_entity_string -> bool (is_drug)
    cache = {}
    
    print(f"Brute-force cleaning dataset: {input_jsonl}...")
    fixed_count = 0
    total_procedures_checked = 0
    
    with open(input_jsonl, 'r', encoding='utf-8') as f_in, open(output_jsonl, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in, desc="Fixing Annotations"):
            data = json.loads(line)
            
            messages = data.get('messages', [])
            if len(messages) < 3: 
                f_out.write(line)
                continue
                
            try:
                raw_entities = json.loads(messages[2]['content'])
            except:
                f_out.write(line)
                continue
                
            has_changes = False
            for ent_dict in raw_entities:
                raw_str = ent_dict.get('entity', "")
                ent_type = ent_dict.get('type', "")
                
                # We strictly check EVERY single Procedure/Treatment using QuickUMLS directly!
                if ent_type == "Procedure/Treatment":
                    total_procedures_checked += 1
                    cleaned_str = clean_text(raw_str).lower()
                    
                    if cleaned_str in cache:
                        is_drug = cache[cleaned_str]
                    else:
                        norm_text = unicodedata.normalize('NFC', cleaned_str)
                        df_results = spacy_quickumls(norm_text)
                        
                        is_drug = False
                        if not df_results.empty:
                            df_results = df_results.sort_values(by=['similarity'], ascending=[False])
                            max_sim = df_results.iloc[0]['similarity']
                            
                            # Check all matches that share the maximum similarity
                            best_matches = df_results[df_results['similarity'] == max_sim]
                            for _, row in best_matches.iterrows():
                                if row['type'] in drug_semantic_types:
                                    is_drug = True
                                    break
                        
                        cache[cleaned_str] = is_drug
                        
                    if is_drug:
                        ent_dict['type'] = "Drug"
                        has_changes = True
                        fixed_count += 1
                        
            # If we fixed something, update the JSON and write it
            if has_changes:
                messages[2]['content'] = json.dumps(raw_entities, ensure_ascii=False)
                data['messages'] = messages
                
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
            
    print(f"\nSUCCESS! Scanned {total_procedures_checked} total procedures using QuickUMLS directly.")
    print(f"Fixed {fixed_count} hidden 'Drug' annotations (like ticagrelor).")
    print(f"Saved brute-force cleaned dataset to: {output_jsonl}")

if __name__ == "__main__":
    clean_dataset_bruteforce()
