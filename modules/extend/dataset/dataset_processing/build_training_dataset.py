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

def build_training_dataset():
    base_dir = r"data\viettel\vietnamese_ner"
    qwen_dataset_path = os.path.join(base_dir, "qwen_dataset.jsonl")
    train_split_path = os.path.join(base_dir, "training", "vietnamese", "document_classification", "doc_class_train.jsonl")
    mapped_entities_path = os.path.join(base_dir, "ground_truth_vn_entity.csv")
    output_training_path = os.path.join(base_dir, "gold_standard_training.jsonl")

    # 1. Load the mapped entities into a fast dictionary
    print("Loading mapped entities...")
    df_mapped = pd.read_csv(mapped_entities_path)
    
    # We only care about successfully mapped entities (similarity > 0)
    df_successful = df_mapped[df_mapped['mapped_cui'].notna() & (df_mapped['mapped_cui'] != "")]
    
    entity_to_cui = {}
    for _, row in df_successful.iterrows():
        raw_ent = str(row['entity'])
        cui = str(row['mapped_cui'])
        
        # Clean it exactly the same way we did during mapping
        cleaned = clean_text(raw_ent).lower()
        entity_to_cui[cleaned] = cui

    print(f"Loaded {len(entity_to_cui)} successfully mapped entities.")

    # 2. Load the train split sentences
    print("Loading train split sentences...")
    train_sentences = set()
    with open(train_split_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            train_sentences.add(data['text'].strip())
            
    print(f"Loaded {len(train_sentences)} train sentences.")

    # 3. Process the generative NER dataset to build the training set
    print("Building Gold Standard Training Dataset...")
    training_data = []
    
    with open(qwen_dataset_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing sentences"):
            data = json.loads(line)
            
            # Extract sentence from the prompt
            messages = data.get('messages', [])
            if len(messages) < 3: continue
            
            sentence = messages[1]['content'].strip()
            
            # ONLY process if this sentence belongs to the Train Split
            if sentence not in train_sentences:
                continue
                
            # Extract the raw generative JSON array from the assistant's response
            try:
                raw_entities = json.loads(messages[2]['content'])
            except:
                continue
                
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
                    if ent_type in gold_standard:
                        gold_standard[ent_type].add(cui)
                        has_mapped_entity = True
            
            # ONLY add the sentence to the training dataset if it contains at least one perfectly mapped entity!
            if has_mapped_entity:
                training_record = {
                    "text": sentence,
                    "gold_entities": {
                        "Disease/Symptom": list(gold_standard["Disease/Symptom"]),
                        "Procedure/Treatment": list(gold_standard["Procedure/Treatment"]),
                        "Drug": list(gold_standard["Drug"])
                    }
                }
                training_data.append(training_record)

    # 4. Save the final training dataset
    with open(output_training_path, 'w', encoding='utf-8') as f:
        for record in training_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print(f"\nSUCCESS! Gold Standard Training Dataset saved to: {output_training_path}")
    print(f"Total Sentences with Perfect Entities: {len(training_data)}")

if __name__ == "__main__":
    build_training_dataset()
