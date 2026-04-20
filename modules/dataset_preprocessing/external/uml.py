import pandas as pd
import numpy as np
from quickumls import QuickUMLS
import spacy
import dotenv
from tqdm import tqdm
dotenv.load_dotenv()

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

quick_umls_path = os.getenv('QUICKUMLS_PATH')
data_dir = os.getenv('DATA_DIR')

nlp = spacy.load("en_ner_bc5cdr_md")
matcher = QuickUMLS(quick_umls_path)
tui_mapping = pd.read_parquet(os.path.join(data_dir, "UML", "META", 'tui_mapping.parquet'))
if 'tui' in tui_mapping.columns:
    tui_mapping = tui_mapping.set_index('tui')

def spacy_quickumls(text):
    
    # SciSpacy
    doc = nlp(text)
    data = [(ent.text, ent.label_) for ent in doc.ents]
    entities = [ent.text for ent in doc.ents]
    df_ent = pd.DataFrame(data, columns=["term", "label"])
    
    # QuickUMLS on FULL TEXT
    results = matcher.match(text)

    flat_data = [item for sublist in results for item in sublist]

    df = pd.DataFrame(flat_data)
    
    # return list(dict.fromkeys(entities)), df_ent

    def get_semantic_type(sem_set):
        if not isinstance(sem_set, (set, list)):
            return None
        # Return the first matching semantic type name
        for tui in sem_set:
            if tui in tui_mapping.index:
                res = tui_mapping.loc[tui, 'sty']
                return res.iloc[0] if isinstance(res, pd.Series) else res
        return None

    df['type'] = df['semtypes'].apply(get_semantic_type)

    df = df[['ngram', 'term', 'cui', 'similarity', 'type']]
    df.columns = ['text', 'term', 'cui', 'similarity', 'type']

    return df, df_ent

# Process MRSTY for CUI-TUI mapping
    mrsty_path = os.path.join(data_dir, "UML", "META", "MRSTY.RRF")

    df_sty = pd.read_csv(
        mrsty_path, 
        sep='|', 
        header=None, 
        names=['cui', 'tui', 'stn', 'sty', 'atui', 'cvf', 'trailing'],
        index_col=False
    )

    df_sty = df_sty[['cui', 'tui', 'sty']]

## Use QuickUML to map between DrugBank Indication to Diagnosis (drug_df is the drugbank dataset)
    output_file = 'drug_diag_results.csv'
    diagnosis_types = ['Disease or Syndrome', 'Injury or Poisoning', 'Finding']

    if os.path.exists(output_file):
        try:
            # Optimization: Only read the one column we need to save RAM
            existing_data = pd.read_csv(output_file, usecols=['original_index'])
            # Ensure indices are integers (or strings) for consistent matching
            processed_indices = set(existing_data['original_index'].dropna().astype(drug_df.index.dtype))
            print(f"Checkpoint found: Resuming from {len(processed_indices)} processed rows.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting fresh.")
            processed_indices = set()
    else:
        processed_indices = set()
        print("No checkpoint found: Starting fresh...")

    for i in tqdm(range(len(drug_df)), desc="Refining Diagnoses"):
        original_idx = drug_df.index[i]
        
        if original_idx in processed_indices:
            continue
            
        row = drug_df.iloc[i].copy()
        text = str(row['indication']) if pd.notnull(row['indication']) else ""
        
        try:
            df_results, _ = spacy_quickumls(text)
            
            if not df_results.empty:
                # Filter by your selected types
                mask = df_results['type'].isin(diagnosis_types)
                df_diag = df_results[mask].copy()
                
                # FIX: Sort similarity DESCENDING (False) so we keep the best matches
                df_diag = df_diag.sort_values(['term', 'similarity'], ascending=[True, False])
                
                # Remove very generic 'Findings' that aren't real diagnoses
                noise_words = ['indicated', 'major', 'various', 'today']
                df_diag = df_diag[~df_diag['term'].str.lower().isin(noise_words)]
                
                df_diag = df_diag.drop_duplicates('term', keep='first')
                
                # Use CUI as requested
                row['related_diagnosis'] = '|'.join(df_diag['cui'].values)
            else:
                row['related_diagnosis'] = ''
                
        except Exception as e:
            # Use tqdm.write so it doesn't break the progress bar
            tqdm.write(f"Error at index {original_idx}: {e}")
            row['related_diagnosis'] = 'ERROR'

        row_to_save = pd.DataFrame([row])
        row_to_save['original_index'] = original_idx 
        
        # Mode 'a' for append, header only written if file is new
        row_to_save.to_csv(
            output_file, 
            mode='a', 
            index=False, 
            sep='\t', 
            quoting=csv.QUOTE_ALL, 
            header=not os.path.exists(output_file)
        )

    print(f"Processing complete! Results saved to {output_file}")

