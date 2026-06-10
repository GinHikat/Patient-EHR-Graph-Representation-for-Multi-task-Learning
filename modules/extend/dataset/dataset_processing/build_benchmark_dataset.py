import os
import sys
import json
import pandas as pd
from tqdm import tqdm
import unicodedata
import re

def clean_text(text):
    text = re.sub(r'[\s]+', ' ', str(text).strip())
    text = re.sub(r'[^\w\s\-\.]', '', text)
    return unicodedata.normalize('NFC', text)

# Mapping UMLS Semantic Types to our 3 Macro Classes
SEMTYPE_MAPPING = {
    # Disease / Symptom
    "Disease or Syndrome": "Disease/Symptom",
    "Sign or Symptom": "Disease/Symptom",
    "Pathologic Function": "Disease/Symptom",
    "Neoplastic Process": "Disease/Symptom",
    "Mental or Behavioral Dysfunction": "Disease/Symptom",
    "Finding": "Disease/Symptom",
    
    # Procedure / Treatment
    "Therapeutic or Preventive Procedure": "Procedure/Treatment",
    "Diagnostic Procedure": "Procedure/Treatment",
    "Health Care Activity": "Procedure/Treatment",
    "Laboratory Procedure": "Procedure/Treatment",
    "Medical Device": "Procedure/Treatment",
    
    # Drug
    "Pharmacologic Substance": "Drug",
    "Organic Chemical": "Drug",
    "Clinical Drug": "Drug",
    "Antibiotic": "Drug",
    "Biomedical or Dental Material": "Drug"
}

def build_benchmark():
    base_dir = r"data\viettel\vietnamese_ner"
    qwen_dataset_path = os.path.join(base_dir, "cleaned_qwen_dataset.jsonl")
    test_split_path = os.path.join(base_dir, "training", "vietnamese", "document_classification", "doc_class_test.jsonl")
    mapped_entities_path = os.path.join(base_dir, "ground_truth_vn_entity.csv")
    output_benchmark_path = os.path.join(base_dir, "gold_standard_benchmark.jsonl")

    # 1. Load the mapped entities into a fast dictionary
    print("Loading mapped entities...")
    df_mapped = pd.read_csv(mapped_entities_path)
    
    # We only care about successfully mapped entities (similarity > 0)
    df_successful = df_mapped[df_mapped['mapped_cui'].notna() & (df_mapped['mapped_cui'] != "")]
    
    entity_to_cui = {}
    entity_to_umls_macro = {}
    for _, row in df_successful.iterrows():
        raw_ent = str(row['entity'])
        cui = str(row['mapped_cui'])
        umls_type = str(row['mapped_type'])
        
        # Clean it exactly the same way we did during mapping
        cleaned = clean_text(raw_ent).lower()
        entity_to_cui[cleaned] = cui
        
        # Map to macro class
        entity_to_umls_macro[cleaned] = SEMTYPE_MAPPING.get(umls_type, None)

    # 2. Load the test split sentences to ensure we only benchmark on the 10% test set
    print("Loading test split sentences...")
    test_sentences = set()
    with open(test_split_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            test_sentences.add(data['text'].strip())
            
    print(f"Loaded {len(test_sentences)} test sentences.")

    # 3. Process the generative NER dataset to build the benchmark
    print("Building Gold Standard Benchmark Dataset...")
    benchmark_data = []
    
    with open(qwen_dataset_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing sentences"):
            data = json.loads(line)
            
            # Extract sentence from the prompt
            messages = data.get('messages', [])
            if len(messages) < 3: continue
            
            sentence = messages[1]['content'].strip()
            
            # ONLY process if this sentence belongs to the Test Split
            if sentence not in test_sentences:
                continue
                
            # Extract the raw generative JSON array from the assistant's response
            try:
                raw_entities = json.loads(messages[2]['content'])
            except:
                continue
                
            # Initialize empty lists for the benchmark
            gold_standard = {
                "Disease/Symptom": set(),
                "Procedure/Treatment": set(),
                "Drug": set()
            }
            
            has_mapped_entity = False
            
            # Map the entities using our dictionary
            for ent_dict in raw_entities:
                raw_str = ent_dict.get('entity', "")
                ent_type = ent_dict.get('type', "")
                
                cleaned_str = clean_text(raw_str).lower()
                
                # If we successfully mapped it earlier, add its CUI
                if cleaned_str in entity_to_cui:
                    cui = entity_to_cui[cleaned_str]
                    umls_macro = entity_to_umls_macro.get(cleaned_str)
                    
                    # USE UMLS GENERALIZED LABEL INSTEAD OF HUMAN LABEL
                    if umls_macro and umls_macro in gold_standard:
                        gold_standard[umls_macro].add(cui)
                        has_mapped_entity = True
            
            # ONLY add the sentence if it has at least one mapped entity!
            # If we leave empty sentences where QuickUMLS failed to map a real disease, 
            # our evaluation script will unfairly punish the model later!
            if has_mapped_entity:
                benchmark_record = {
                    "text": sentence,
                    "gold_entities": {
                        "Disease/Symptom": list(gold_standard["Disease/Symptom"]),
                        "Procedure/Treatment": list(gold_standard["Procedure/Treatment"]),
                        "Drug": list(gold_standard["Drug"])
                    }
                }
                benchmark_data.append(benchmark_record)

    # 4. Save the final benchmark
    with open(output_benchmark_path, 'w', encoding='utf-8') as f:
        for record in benchmark_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print(f"\nSUCCESS! Gold Standard Benchmark saved to: {output_benchmark_path}")
    print(f"Total Test Sentences Processed: {len(benchmark_data)}")

if __name__ == "__main__":
    build_benchmark()
