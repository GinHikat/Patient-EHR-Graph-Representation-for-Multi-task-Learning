import os
import sys
import json
import pandas as pd
from tqdm import tqdm
import unicodedata
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("_", " ")
    text = text.strip(" .,;!?\"'()[]{}")
    text = re.sub(r'\s+', ' ', text)
    return text

def clean_dataset():
    base_dir = r"d:\Study\Education\Projects\Thesis\data\viettel\vietnamese_ner"
    input_jsonl = os.path.join(base_dir, "unified_qwen_dataset.jsonl")
    output_jsonl = os.path.join(base_dir, "unified_qwen_dataset_cleaned.jsonl")
    mapped_csv = os.path.join(base_dir, "ground_truth_vn_entity.csv")
    
    # 1. Load the mapped CSV to build a fast lookup dictionary for Drug overrides
    print("Loading QuickUMLS ontology mappings...")
    df_mapped = pd.read_csv(mapped_csv)
    df_successful = df_mapped[df_mapped['mapped_cui'].notna() & (df_mapped['mapped_cui'] != "")]
    
    # These semantic types officially mean "Drug" in the UMLS ontology
    drug_semantic_types = {
        "Pharmacologic Substance", "Organic Chemical", "Clinical Drug", 
        "Antibiotic", "Biomedical or Dental Material", "Hormone",
        "Vitamin", "Immunologic Factor"
    }
    
    # Dictionary mapping: cleaned_entity_string -> True (if it's a drug)
    entity_is_drug = {}
    for _, row in df_successful.iterrows():
        raw_ent = str(row['entity'])
        mapped_type = str(row['mapped_type'])
        
        cleaned = clean_text(raw_ent).lower()
        if mapped_type in drug_semantic_types:
            entity_is_drug[cleaned] = True
            
    print(f"Identified {len(entity_is_drug)} unique entities that are actually Drugs according to UMLS.")
    
    # 2. Process the main dataset and override labels
    print(f"Cleaning dataset: {input_jsonl}...")
    fixed_count = 0
    
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
                
                # Check if it was annotated as a Procedure/Treatment but is actually a Drug
                if ent_type == "Procedure/Treatment":
                    cleaned_str = clean_text(raw_str).lower()
                    if cleaned_str in entity_is_drug:
                        ent_dict['type'] = "Drug"
                        has_changes = True
                        fixed_count += 1
                        
            # If we fixed something, update the JSON and write it
            if has_changes:
                messages[2]['content'] = json.dumps(raw_entities, ensure_ascii=False)
                data['messages'] = messages
                
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
            
    print(f"\nSUCCESS! Fixed {fixed_count} hidden 'Drug' annotations.")
    print(f"Saved clean dataset to: {output_jsonl}")

if __name__ == "__main__":
    clean_dataset()
