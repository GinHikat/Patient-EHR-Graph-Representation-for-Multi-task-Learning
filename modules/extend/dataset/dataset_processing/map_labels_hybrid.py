import pandas as pd
import json
import os
import sys
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add root project dir to path so we can import modules
sys.path.append(os.path.abspath("."))
from modules.models.models import EmbeddingModels

def main():
    base_dir = "data/viettel/vietnamese_ner/training/vietnamese/document_classification"
    csv_path = "data/viettel/mapping/ground_truth_vn_entity.csv"
    db_path = "data/viettel/mapping/mapped_entities_embedded.pkl"

    # Exact Match Dictionary
    print("Loading Exact Match Dictionary...")
    df = pd.read_csv(csv_path)
    entity_to_cui = {}
    for _, row in df.iterrows():
        entity = str(row['entity']).lower().replace('_', ' ').strip()
        cui = str(row['mapped_cui']).strip()
        if entity not in entity_to_cui:
            entity_to_cui[entity] = cui

    datasets = ['doc_class_train.jsonl', 'doc_class_dev.jsonl', 'doc_class_test.jsonl']
    
    # Collect unique unmapped labels
    print("Collecting unmapped labels...")
    all_unmapped = set()
    for filename in datasets:
        input_path = os.path.join(base_dir, filename)
        if not os.path.exists(input_path): continue
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                for label in data.get("labels", []):
                    clean_label = str(label).lower().replace('_', ' ').strip()
                    if clean_label not in entity_to_cui:
                        all_unmapped.add(clean_label)
                        
    unmapped_list = list(all_unmapped)
    print(f"Found {len(unmapped_list)} unique unmapped labels across all datasets.")

    # SapBERT Mapping
    sapbert_to_cui = {}
    if unmapped_list:
        print("\nLoading SapBERT and Database embeddings...")
        sapbert_model_name = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
        embedder = EmbeddingModels(model_choice=sapbert_model_name)
        
        base_df = pd.read_pickle(db_path)
        base_df = base_df.dropna(subset=['mapped_cui']).reset_index(drop=True)
        base_embeddings = np.vstack(base_df['embedding'].values)
        
        print(f"Computing SapBERT embeddings for {len(unmapped_list)} unmapped labels...")
        unmapped_embeddings = embedder.encode_text(unmapped_list, show_progress=True)
        unmapped_embeddings = np.array(unmapped_embeddings)
        
        print("Computing Cosine Similarities...")
        similarity_matrix = cosine_similarity(unmapped_embeddings, base_embeddings)
        
        max_similarities = np.max(similarity_matrix, axis=1)
        best_match_indices = np.argmax(similarity_matrix, axis=1)
        
        threshold = 0.90
        for i, label in enumerate(unmapped_list):
            if max_similarities[i] >= threshold:
                best_cui = base_df.iloc[best_match_indices[i]]['mapped_cui']
                if pd.notna(best_cui) and str(best_cui).strip():
                    sapbert_to_cui[label] = str(best_cui).strip()
                    
        print(f"Successfully mapped {len(sapbert_to_cui)} labels using SapBERT (Threshold >= {threshold}).")

    # Generate Output Files
    print("\nGenerating final mapped datasets...")
    for filename in datasets:
        input_path = os.path.join(base_dir, filename)
        output_path = os.path.join(base_dir, f"cui_mapped_{filename}")
        
        if not os.path.exists(input_path): continue
            
        sentences_kept = 0
        sentences_dropped = 0
        
        with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                if not line.strip(): continue
                data = json.loads(line)
                
                text = data.get("text", "")
                labels = data.get("labels", [])
                
                cui_list = []
                for label in labels:
                    clean_label = str(label).lower().replace('_', ' ').strip()
                    
                    cui = None
                    if clean_label in entity_to_cui:
                        cui = entity_to_cui[clean_label]  # Lexical match
                    elif clean_label in sapbert_to_cui:
                        cui = sapbert_to_cui[clean_label] # SapBERT match
                        
                    if cui and cui not in cui_list:
                        cui_list.append(cui)
                        
                # Only write the sentence if it successfully mapped at least one CUI
                if cui_list:
                    output_data = {
                        "text": text,
                        "gold_entities": cui_list
                    }
                    f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                    sentences_kept += 1
                else:
                    sentences_dropped += 1
                    
        print(f"[{filename}] Kept: {sentences_kept} sentences | Dropped (No CUIs mapped): {sentences_dropped}")
        
    print("\nAll datasets processed successfully!")

if __name__ == "__main__":
    main()
