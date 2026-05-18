import os
import sys
import numpy as np
import pandas as pd

# Setup project root path
project_root = "d:/Study/Education/Projects/Thesis"
if project_root not in sys.path:
    sys.path.append(project_root)

from shared_functions.global_functions import query_neo4j

def main():
    downstream_data_path = os.path.join(project_root, 'data', 'downstream')
    
    print("1. Loading kg_nodes.csv...", flush=True)
    all_nodes = pd.read_csv(os.path.join(downstream_data_path, 'kg_nodes.csv'), dtype={'id': str}, low_memory=False)
    print(f"kg_nodes Shape: {all_nodes.shape}", flush=True)
    
    # 2. Loading target diagnoses from radiology.csv (ONLY loading the target_diagnosis column)
    print("\n2. Loading target diagnoses from radiology.csv...", flush=True)
    radiology_path = os.path.join(project_root, 'data', 'radiology.csv')
    rad_df = pd.read_csv(radiology_path, usecols=['target_diagnosis'])
    
    # Clean list-formatted string columns using fast vectorized operations
    rad_df['target_clean'] = (
        rad_df['target_diagnosis']
        .fillna('[]')
        .astype(str)
        .str.replace('[', '', regex=False)
        .str.replace(']', '', regex=False)
        .str.replace("'", "", regex=False)
        .str.replace(' ', '', regex=False)
    )
    
    # Explode and get unique categories
    flat_cats = rad_df['target_clean'].str.split(',').explode()
    all_rad_categories = set(flat_cats[flat_cats != ''].str.strip().str.lower().dropna().unique())
    print(f"Total unique target CCSR categories in radiology.csv: {len(all_rad_categories)}", flush=True)
    
    # Load CCSR diagnosis mapping table
    print("\n3. Loading CCSR Diagnosis Mapping table...", flush=True)
    map_path = os.path.join(project_root, 'data', 'diagnosis_map.csv')
    map_df = pd.read_csv(map_path, low_memory=False)
    map_df['ccsr_category_str'] = map_df['ccsr_category'].astype(str).str.strip().str.lower()
    map_df['icd_code_str'] = map_df['icd_code'].astype(str).str.strip().str.lower()
    cat_to_icds = map_df.groupby('ccsr_category_str')['icd_code_str'].apply(list).to_dict()
    
    # Map target categories to ICD codes
    all_mapped_icds = set()
    for cat in all_rad_categories:
        icds = cat_to_icds.get(cat.lower(), [])
        all_mapped_icds.update(icds)
        
    print(f"Total specific ICD codes mapped from our target CCSR categories: {len(all_mapped_icds)}", flush=True)
    
    # 4. Check if these mapped ICD codes exist in kg_nodes.csv (i.e., have GAT embeddings)
    all_nodes['id_str'] = all_nodes['id'].astype(str).str.strip().str.lower()
    embedded_ids = set(all_nodes['id_str'].unique())
    
    matching_mapped_icds = all_mapped_icds.intersection(embedded_ids)
    print(f"Mapped ICD codes present in kg_nodes.csv (have GAT embeddings): {len(matching_mapped_icds)} / {len(all_mapped_icds)}", flush=True)
    
    # 5. Check if these matching ICD codes exist in Neo4j and have relationships with other Diagnosis nodes
    if matching_mapped_icds:
        sample_icds = [x.upper() for x in list(matching_mapped_icds)[:10]]
        print(f"\nChecking Neo4j counts for a sample of mapped ICD codes (uppercase): {sample_icds}", flush=True)
        res = query_neo4j("""
            MATCH (d:Diagnosis)
            WHERE d.id IN $icds
            RETURN count(d) AS count
        """, icds=sample_icds)
        print(f"Matching sample nodes with label 'Diagnosis' in Neo4j: {res}", flush=True)
        
        # Check relationships with other Diagnosis nodes in Neo4j
        rel_res = query_neo4j("""
            MATCH (d:Diagnosis)-[r]-(o:Diagnosis)
            WHERE d.id IN $icds AND o <> d
            RETURN type(r) AS rel_type, labels(o) AS dest_labels, count(*) AS count
            LIMIT 10
        """, icds=sample_icds)
        print(f"Relationships of sample ICD nodes in Neo4j with other Diagnosis nodes: {rel_res}", flush=True)

if __name__ == "__main__":
    main()
