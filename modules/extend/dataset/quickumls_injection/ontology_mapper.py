import os
import sys
import pandas as pd
import unicodedata
import re
from tqdm import tqdm

# Connect to App Backend
project_root = r"d:\Study\Education\Projects\Thesis"
if project_root not in sys.path:
    sys.path.append(project_root)

from modules.dataset_preprocessing.external.uml import spacy_quickumls

def map_entities():
    base_dir = r"data\viettel\vietnamese_ner"
    input_csv = os.path.join(base_dir, "aggregated_entities.csv")
    output_csv = os.path.join(base_dir, "mapped_entities.csv")
    
    print(f"Loading unmapped entities from: {input_csv}")
    df_raw = pd.read_csv(input_csv)
    
    # Checkpoint logic
    if os.path.exists(output_csv):
        df_mapped = pd.read_csv(output_csv)
        completed_entities = set(df_mapped['entity'].values)
        print(f"Resuming from checkpoint: {len(completed_entities)} already mapped.")
    else:
        completed_entities = set()
        
    # Open file in append mode
    write_header = not os.path.exists(output_csv)
    
    # Build direct fallback dictionaries to catch short words (like "ho") that QuickUMLS ignores
    diag_csv = r"data\viettel\combine\diagnosis_10.csv"
    proc_csv = r"data\viettel\combine\procedure_9.csv"
    diag_df = pd.read_csv(diag_csv, dtype=str)
    proc_df = pd.read_csv(proc_csv, dtype=str)
    
    fallback_dict = {}
    for _, row in diag_df.iterrows():
        if pd.notna(row['name_vi']) and pd.notna(row['id']):
            fallback_dict[str(row['name_vi']).lower().strip()] = {"cui": str(row['id']), "type": "Disease or Syndrome"}
    for _, row in proc_df.iterrows():
        if pd.notna(row['term_vi']) and pd.notna(row['id']):
            fallback_dict[str(row['term_vi']).lower().strip()] = {"cui": str(row['id']), "type": "Therapeutic or Preventive Procedure"}

    print("\nStarting QuickUMLS Mapping Engine...")
    
    with open(output_csv, 'a', encoding='utf-8-sig') as f:
        if write_header:
            f.write("entity,original_type,count,mapped_cui,mapped_term,mapped_type,similarity\n")
            
        # We wrap it in a tqdm progress bar
        for _, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc="Mapping Entities"):
            raw_entity = str(row['entity'])
            
            if raw_entity in completed_entities:
                continue
                
            original_type = row['type']
            count = row['count']
            
            # Clean NER Artifacts (like underscores from VnCoreNLP and trailing punctuation)
            cleaned_entity = raw_entity.replace("_", " ").strip(" .,;!?\"'()[]{}")
            cleaned_entity = re.sub(r'\s+', ' ', cleaned_entity)
            
            # QuickUMLS was built WITHOUT lowercase (-L) and WITHOUT unicode normalization (-U).
            # This means it is strictly case-sensitive and unicode-sensitive.
            variants = set()
            for text_variant in [cleaned_entity, cleaned_entity.capitalize(), cleaned_entity.lower(), cleaned_entity.title()]:
                variants.add(unicodedata.normalize('NFC', text_variant))
                variants.add(unicodedata.normalize('NFD', text_variant))
                
            best_sim = -1.0
            best_match = None
            
            for v in variants:
                df_results = spacy_quickumls(v)
                if not df_results.empty:
                    top = df_results.sort_values(by='similarity', ascending=False).iloc[0]
                    if top['similarity'] > best_sim:
                        best_sim = top['similarity']
                        best_match = top
            
            if best_match is None:
                # Fallback: Direct exact dictionary lookup for short words
                lookup_key = cleaned_entity.lower()
                if lookup_key in fallback_dict:
                    fallback = fallback_dict[lookup_key]
                    f.write(f'"{raw_entity}","{original_type}",{count},"{fallback["cui"]}","{cleaned_entity}","{fallback["type"]}",1.00\n')
                else:
                    f.write(f'"{raw_entity}","{original_type}",{count},"","","",0.0\n')
            else:
                cui = best_match['cui']
                term = best_match['term']
                mapped_type = best_match['type']
                sim = best_match['similarity']
                
                # Write to CSV securely
                f.write(f'"{raw_entity}","{original_type}",{count},"{cui}","{term}","{mapped_type}",{sim:.2f}\n')
                
            # Flush every row to ensure we don't lose data if it crashes
            f.flush()

    print(f"\nSUCCESS! Mapping completed and saved to: {output_csv}")

if __name__ == "__main__":
    map_entities()
