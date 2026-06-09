import os
import sys
import json
import unicodedata
import pandas as pd

project_root = r"d:\Study\Education\Projects\Thesis"
if project_root not in sys.path:
    sys.path.append(project_root)

from modules.dataset_preprocessing.external.uml import spacy_quickumls

# Mapping for semantic types so we can see the macro class
SEMTYPE_MAPPING = {
    # Disease / Symptom
    "Disease or Syndrome": "Disease/Symptom", "Sign or Symptom": "Disease/Symptom", "Pathologic Function": "Disease/Symptom",
    "Neoplastic Process": "Disease/Symptom", "Mental or Behavioral Dysfunction": "Disease/Symptom", "Finding": "Disease/Symptom",
    # Procedure / Treatment
    "Therapeutic or Preventive Procedure": "Procedure/Treatment", "Diagnostic Procedure": "Procedure/Treatment",
    "Health Care Activity": "Procedure/Treatment", "Laboratory Procedure": "Procedure/Treatment", "Medical Device": "Procedure/Treatment",
    # Drug
    "Pharmacologic Substance": "Drug", "Organic Chemical": "Drug", "Clinical Drug": "Drug", "Antibiotic": "Drug", "Biomedical or Dental Material": "Drug"
}

def debug_one_sample():
    benchmark_path = r"d:\Study\Education\Projects\Thesis\data\viettel\vietnamese_ner\gold_standard_benchmark.jsonl"
    
    sample_record = None
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        # Find a sentence that has at least 1 entity in ALL 3 classes
        for line in f:
            record = json.loads(line)
            ge = record['gold_entities']
            if len(ge.get('Disease/Symptom', [])) > 0 and len(ge.get('Procedure/Treatment', [])) > 0 and len(ge.get('Drug', [])) > 0:
                sample_record = record
                break
                
    if not sample_record:
        print("Could not find a single sentence containing all 3 classes simultaneously in the 444 test sentences.")
        return
        
    text = sample_record['text']
    gold_entities = sample_record['gold_entities']
    
    print("="*60)
    print(f"RAW SENTENCE: {text}")
    print("="*60)
    
    print("\n--- GOLD STANDARD ---")
    
    # Load the CSV to translate CUIs to readable terms
    cui_to_term = {}
    csv_path = r"d:\Study\Education\Projects\Thesis\data\viettel\vietnamese_ner\ground_truth_vn_entity.csv"
    if os.path.exists(csv_path):
        df_csv = pd.read_csv(csv_path)
        for _, row in df_csv.iterrows():
            if pd.notna(row['mapped_cui']) and pd.notna(row['mapped_term']):
                cui_to_term[str(row['mapped_cui'])] = str(row['mapped_term'])
                
    for k, v in gold_entities.items():
        if v:
            readable_list = []
            for cui in v:
                term_name = cui_to_term.get(cui, "UNKNOWN")
                readable_list.append(f"{cui} ({term_name})")
            print(f"  [{k}]: {readable_list}")
            
    print("\n--- QUICKUMLS ENGINE (Scanning Raw Text) ---")
    clean_text = unicodedata.normalize('NFC', text)
    df_results = spacy_quickumls(clean_text)
    
    pred_sets = {
        "Disease/Symptom": set(),
        "Procedure/Treatment": set(),
        "Drug": set()
    }
    
    if df_results.empty:
        print("  => QuickUMLS found absolutely nothing!")
    else:
        # Deduplicate to show only the best CUI per matched text string
        df_results = df_results.sort_values(by=['similarity'], ascending=[False])
        df_results = df_results.drop_duplicates(subset=['text'], keep='first')
        
        for _, row in df_results.iterrows():
            macro = SEMTYPE_MAPPING.get(row['type'], "UNKNOWN/IGNORE")
            print(f"  Matched: '{row['text']}' -> CUI: {row['cui']} | Macro: {macro} | Orig_Type: {row['type']}")
            if macro in pred_sets:
                pred_sets[macro].add(row['cui'])

    print("\n--- SINGLE SENTENCE METRICS ---")
    for cls in ["Disease/Symptom", "Procedure/Treatment", "Drug"]:
        gold = set(gold_entities.get(cls, []))
        pred = pred_sets[cls]
        
        tp = len(gold.intersection(pred))
        fp = len(pred - gold)
        fn = len(gold - pred)
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        print(f"[{cls}]")
        print(f"  TP: {tp} | FP: {fp} | FN: {fn}")
        print(f"  Precision: {p:.2f} | Recall: {r:.2f} | F1: {f1:.2f}\n")

if __name__ == "__main__":
    debug_one_sample()
