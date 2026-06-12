import pandas as pd
import json
import os

def main():
    base_dir = "data/viettel/vietnamese_ner/training/vietnamese/document_classification"
    csv_path = "data/viettel/mapping/ground_truth_vn_entity.csv"

    # 1. Load CSV and create mapping dictionary
    df = pd.read_csv(csv_path)
    
    # Create mapping: lowercase entity -> CUI
    # We drop duplicates just in case an entity appears multiple times
    entity_to_cui = {}
    for _, row in df.iterrows():
        entity = str(row['entity']).lower().strip()
        cui = str(row['mapped_cui']).strip()
        # Keep the first mapping we see
        if entity not in entity_to_cui:
            entity_to_cui[entity] = cui

    # 2. Process the datasets
    datasets = ['doc_class_train.jsonl', 'doc_class_dev.jsonl', 'doc_class_test.jsonl']
    
    for filename in datasets:
        input_path = os.path.join(base_dir, filename)
        output_path = os.path.join(base_dir, f"cui_mapped_{filename}")
        
        if not os.path.exists(input_path):
            print(f"Skipping {filename}, not found.")
            continue
            
        print(f"Processing {filename}...")
        mapped_count = 0
        unmapped_count = 0
        
        with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                data = json.loads(line)
                
                text = data.get("text", "")
                labels = data.get("labels", [])
                
                cui_list = []
                for label in labels:
                    lbl_lower = str(label).lower().replace('_', ' ').strip()
                    if lbl_lower in entity_to_cui:
                        cui = entity_to_cui[lbl_lower]
                        if cui not in cui_list:  # Avoid duplicates
                            cui_list.append(cui)
                        mapped_count += 1
                    else:
                        unmapped_count += 1
                
                # Construct the output format matching gold_standard_benchmark.jsonl
                output_data = {
                    "text": text,
                    "gold_entities": {
                        "Disease/Symptom": cui_list,
                        "Procedure/Treatment": [],
                        "Drug": []
                    }
                }
                
                f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                
        print(f"  Finished {filename}: {mapped_count} labels successfully mapped, {unmapped_count} labels could not be found in CSV.")

if __name__ == "__main__":
    main()
