import os
import sys
import json
import pandas as pd
from tqdm import tqdm
import unicodedata

# Connect to App Backend
project_root = r"d:\Study\Education\Projects\Thesis"
if project_root not in sys.path:
    sys.path.append(project_root)

from modules.dataset_preprocessing.external.uml import spacy_quickumls

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

def evaluate_quickumls():
    benchmark_path = r"d:\Study\Education\Projects\Thesis\data\viettel\vietnamese_ner\gold_standard_benchmark.jsonl"
    
    # Global tracking for TP, FP, FN
    metrics = {
        "Disease/Symptom": {"tp": 0, "fp": 0, "fn": 0},
        "Procedure/Treatment": {"tp": 0, "fp": 0, "fn": 0},
        "Drug": {"tp": 0, "fp": 0, "fn": 0}
    }
    
    print("Loading Gold Standard Benchmark...")
    sentences = []
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        for line in f:
            sentences.append(json.loads(line))
            
    print(f"Loaded {len(sentences)} sentences for evaluation.")
    print("Running QuickUMLS Inference (This will take a minute or two)...\n")
    
    for record in tqdm(sentences, desc="Evaluating QuickUMLS"):
        text = record['text']
        gold_entities = record['gold_entities']
        
        # Convert gold lists to sets for fast intersection
        gold_sets = {
            "Disease/Symptom": set(gold_entities.get("Disease/Symptom", [])),
            "Procedure/Treatment": set(gold_entities.get("Procedure/Treatment", [])),
            "Drug": set(gold_entities.get("Drug", []))
        }
        
        # Initialize predicted sets
        pred_sets = {
            "Disease/Symptom": set(),
            "Procedure/Treatment": set(),
            "Drug": set()
        }
        
        # Clean text exactly as QuickUMLS expects (NFC normalization)
        clean_text = unicodedata.normalize('NFC', text)
        
        # Run QuickUMLS!
        df_results = spacy_quickumls(clean_text)
        
        if not df_results.empty:
            # Sort by similarity descending, so the absolute best match is first
            df_results = df_results.sort_values(by=['similarity'], ascending=[False])
            
            # Filter by the matched term in the sentence to only keep the ONE best CUI per term
            df_results = df_results.drop_duplicates(subset=['text'], keep='first')
            
            for _, row in df_results.iterrows():
                umls_type = row['type']
                cui = row['cui']
                
                # Map the UMLS semantic type to our 3 classes
                macro_class = SEMTYPE_MAPPING.get(umls_type)
                
                if macro_class:
                    pred_sets[macro_class].add(cui)
                    
        # Compare Sets and Update Metrics
        for cls in ["Disease/Symptom", "Procedure/Treatment", "Drug"]:
            gold = gold_sets[cls]
            pred = pred_sets[cls]
            
            tp = len(gold.intersection(pred))
            fp = len(pred - gold)
            fn = len(gold - pred)
            
            metrics[cls]["tp"] += tp
            metrics[cls]["fp"] += fp
            metrics[cls]["fn"] += fn

    print("\n" + "="*50)
    print("QUICKUMLS BASELINE EVALUATION RESULTS")
    print("="*50)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for cls in ["Disease/Symptom", "Procedure/Treatment", "Drug"]:
        tp = metrics[cls]["tp"]
        fp = metrics[cls]["fp"]
        fn = metrics[cls]["fn"]
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"\n[{cls}]")
        print(f"  Precision: {precision:.4f}  (TP: {tp}, FP: {fp})")
        print(f"  Recall:    {recall:.4f}  (FN: {fn})")
        print(f"  F1-Score:  {f1:.4f}")

    print("\n" + "-"*50)
    macro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    macro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    macro_f1 = 2 * (macro_p * macro_r) / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0.0
    
    print(f"OVERALL MACRO (All Classes):")
    print(f"  Precision: {macro_p:.4f}")
    print(f"  Recall:    {macro_r:.4f}")
    print(f"  F1-Score:  {macro_f1:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    evaluate_quickumls()
