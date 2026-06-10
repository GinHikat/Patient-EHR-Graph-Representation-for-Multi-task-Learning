import os
import json
import pandas as pd
from tqdm import tqdm
import unicodedata
import re

def clean_text(text):
    text = re.sub(r'[\s]+', ' ', str(text).strip())
    text = re.sub(r'[^\w\s\-\.]', '', text)
    return unicodedata.normalize('NFC', text)

# UMLS Semantic Type -> Macro Class Mapping
SEMTYPE_MAPPING = {
    "Disease or Syndrome": "Disease/Symptom",
    "Sign or Symptom": "Disease/Symptom",
    "Pathologic Function": "Disease/Symptom",
    "Neoplastic Process": "Disease/Symptom",
    "Mental or Behavioral Dysfunction": "Disease/Symptom",
    "Finding": "Disease/Symptom",
    
    "Therapeutic or Preventive Procedure": "Procedure/Treatment",
    "Diagnostic Procedure": "Procedure/Treatment",
    "Health Care Activity": "Procedure/Treatment",
    "Laboratory Procedure": "Procedure/Treatment",
    "Medical Device": "Procedure/Treatment",
    
    "Pharmacologic Substance": "Drug",
    "Organic Chemical": "Drug",
    "Clinical Drug": "Drug",
    "Antibiotic": "Drug",
    "Biomedical or Dental Material": "Drug"
}

def fix_conll(file_path, entity_to_macro):
    if not os.path.exists(file_path):
        print(f"Skipping {file_path} (not found)")
        return 0
        
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    sentences = []
    current_sentence = []
    
    # Parse the CoNLL into sentences with token info
    for line in lines:
        line = line.strip()
        if not line:
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
            continue
            
        parts = line.split('\t') if '\t' in line else line.split(' ')
        if len(parts) >= 2:
            word, tag = parts[0], parts[-1]
            current_sentence.append([word, tag])
            
    if current_sentence:
        sentences.append(current_sentence)
        
    corrected_count = 0
    
    # Process each sentence to find entities and fix them
    for sentence in sentences:
        i = 0
        while i < len(sentence):
            word, tag = sentence[i]
            if tag.startswith('B-'):
                start_idx = i
                current_type = tag[2:]
                entity_words = [word]
                
                # find the end of the entity
                j = i + 1
                while j < len(sentence) and sentence[j][1] == f"I-{current_type}":
                    entity_words.append(sentence[j][0])
                    j += 1
                    
                entity_text = ' '.join(entity_words)
                cleaned = clean_text(entity_text).lower()
                
                # Check mapping
                if cleaned in entity_to_macro:
                    new_type = entity_to_macro[cleaned]
                    if new_type != current_type:
                        # Overwrite the tags with the correct UMLS macro class
                        sentence[start_idx][1] = f"B-{new_type}"
                        for k in range(start_idx + 1, j):
                            sentence[k][1] = f"I-{new_type}"
                        corrected_count += 1
                        
                i = j
            else:
                i += 1
                
    # 3. Write back to the exact same file
    with open(file_path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            for word, tag in sentence:
                f.write(f"{word}\t{tag}\n")
            f.write("\n")
            
    return corrected_count

def fix_jsonl(file_path, entity_to_macro):
    if not os.path.exists(file_path):
        print(f"Skipping {file_path} (not found)")
        return 0
        
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    corrected_count = 0
    corrected_lines = []
    
    for line in lines:
        data = json.loads(line)
        messages = data.get('messages', [])
        if len(messages) >= 3:
            try:
                raw_entities = json.loads(messages[2]['content'])
                for ent in raw_entities:
                    current_type = ent.get('type', "")
                    cleaned = clean_text(ent.get('entity', "")).lower()
                    if cleaned in entity_to_macro:
                        new_type = entity_to_macro[cleaned]
                        if new_type != current_type:
                            ent['type'] = new_type
                            corrected_count += 1
                messages[2]['content'] = json.dumps(raw_entities, ensure_ascii=False)
            except:
                pass
        corrected_lines.append(json.dumps(data, ensure_ascii=False))
        
    # Write back to the exact same file
    with open(file_path, 'w', encoding='utf-8') as f:
        for c_line in corrected_lines:
            f.write(c_line + '\n')
            
    return corrected_count

def main():
    base_dir = r"data\viettel\vietnamese_ner"
    csv_path = os.path.join(base_dir, "ground_truth_vn_entity.csv")
    
    print("Loading Ground Truth Mappings...")
    df_mapped = pd.read_csv(csv_path)
    df_successful = df_mapped[df_mapped['mapped_cui'].notna() & (df_mapped['mapped_cui'] != "")]
    
    entity_to_macro = {}
    for _, row in df_successful.iterrows():
        raw_ent = str(row['entity'])
        umls_type = str(row['mapped_type'])
        
        cleaned = clean_text(raw_ent).lower()
        macro_class = SEMTYPE_MAPPING.get(umls_type)
        if macro_class:
            entity_to_macro[cleaned] = macro_class
            
    print(f"Loaded {len(entity_to_macro)} verified UMLS mappings.\n")
    
    # Define files to process
    conll_files = [
        os.path.join(base_dir, "training", "vietnamese", "ner_train", "ner_train.conll"),
        os.path.join(base_dir, "training", "vietnamese", "ner_train", "ner_dev.conll"),
        os.path.join(base_dir, "training", "vietnamese", "ner_train", "ner_test.conll")
    ]
    
    jsonl_files = [
        os.path.join(base_dir, "training", "vietnamese", "qwen_finetune", "qwen_train.jsonl"),
        os.path.join(base_dir, "training", "vietnamese", "qwen_finetune", "qwen_dev.jsonl"),
        os.path.join(base_dir, "training", "vietnamese", "qwen_finetune", "qwen_test.jsonl")
    ]
    
    print("Fixing CoNLL Formats...")
    for f in conll_files:
        cnt = fix_conll(f, entity_to_macro)
        print(f"Fixed {cnt} labels in {os.path.basename(f)}")
        
    print("\nFixing JSONL Formats...")
    for f in jsonl_files:
        cnt = fix_jsonl(f, entity_to_macro)
        print(f"Fixed {cnt} labels in {os.path.basename(f)}")
        
    print("\nSUCCESS! All training datasets have been aligned with the UMLS Knowledge Graph.")

if __name__ == "__main__":
    main()
