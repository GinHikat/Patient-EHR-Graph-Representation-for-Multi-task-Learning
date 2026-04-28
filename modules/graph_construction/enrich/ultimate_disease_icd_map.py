import pandas as pd
import numpy as np
import duckdb

import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from dotenv import load_dotenv
load_dotenv() 
base_data_dir = os.path.join(project_root, 'data')
data_dir = os.getenv('DATA_DIR')

csv_path = os.path.join(base_data_dir, 'nodes.parquet')
sct_root = os.path.join(data_dir, 'Mapping', 'SnomedCT', 'csv')
uml_path = os.path.join(base_data_dir, 'utils', 'uml.csv')

f_ext_map = f'{sct_root}/Refset/Map/iisssccRefset_ExtendedMap.csv'
f_simple_map = f'{sct_root}/Refset/Map/sRefset_SimpleMap.csv'
f_rel = f'{sct_root}/Terminology/relationship.csv'
f_assoc = f'{sct_root}/Refset/Content/cRefset_Association.csv'

# Diagnosis nodes, these are originally ICD format
filtered_df = duckdb.query(f"""
    SELECT * 
    FROM read_parquet('{csv_path}') 
    WHERE labels ILIKE '%Diagnosis%'
""").to_df()
filtered_df = filtered_df[['id', 'labels', 'name']]

# Start mapping ICD to OMIM, HPO and MESH using SnomedCT and UML 
def get_clean_clinical_mappings(icd_list):
    conn = duckdb.connect()
    input_df = pd.DataFrame({'input_icd': [str(x).replace('.', '') for x in icd_list]})
    
    query = f"""
    WITH sct_sources AS (
        SELECT REPLACE(i.CODE, '.', '') as icd_id, CAST(s.CODE AS VARCHAR) as sct_id
        FROM read_csv_auto('{uml_path}') i JOIN read_csv_auto('{uml_path}') s ON i.CUI = s.CUI
        WHERE i.SAB IN ('ICD9CM', 'ICD10') AND s.SAB = 'SNOMEDCT_US'
    ),
    sct_corrections AS (
        SELECT CAST(referencedComponentId AS VARCHAR) as old_id, CAST(targetComponentId AS VARCHAR) as new_id 
        FROM read_csv_auto('{f_assoc}') 
        WHERE active=1 AND refsetId IN ('900000000000527005', '900000000000526001')
    ),
    sct_hierarchy AS (
        SELECT CAST(sourceId AS VARCHAR) as sourceId, CAST(destinationId AS VARCHAR) as parent_id 
        FROM read_csv_auto('{f_rel}') WHERE active=1 AND typeId = '116680003'
    ),
    ext_map AS (
        SELECT CAST(CODE AS VARCHAR) as sct_id, CUI FROM read_csv_auto('{uml_path}') WHERE SAB = 'SNOMEDCT_US'
    ),
    ext_vals AS (
        SELECT CUI, SAB as db, CODE as ext_id FROM read_csv_auto('{uml_path}') WHERE SAB IN ('HPO', 'OMIM', 'MSH')
    ),
    mapping_hub AS (
        SELECT 
            s.icd_id,
            COALESCE(corr.new_id, s.sct_id) as final_sct_id,
            h.parent_id
        FROM sct_sources s
        LEFT JOIN sct_corrections corr ON s.sct_id = corr.old_id
        LEFT JOIN sct_hierarchy h ON COALESCE(corr.new_id, s.sct_id) = h.sourceId
    )
    SELECT 
        src.input_icd as icd_code,
        GROUP_CONCAT(DISTINCT m.final_sct_id) as snomed_ids,
        
        -- HPO (Phenotype)
        GROUP_CONCAT(DISTINCT v.ext_id) FILTER (WHERE v.db = 'HPO') as hpo_direct,
        GROUP_CONCAT(DISTINCT pv.ext_id) FILTER (WHERE pv.db = 'HPO') as hpo_hierarchy,
        
        -- OMIM (Genomics)
        GROUP_CONCAT(DISTINCT v.ext_id) FILTER (WHERE v.db = 'OMIM') as omim_direct,
        GROUP_CONCAT(DISTINCT pv.ext_id) FILTER (WHERE pv.db = 'OMIM') as omim_hierarchy,
        
        -- MeSH (Literature)
        GROUP_CONCAT(DISTINCT v.ext_id) FILTER (WHERE v.db = 'MSH') as mesh_direct,
        GROUP_CONCAT(DISTINCT pv.ext_id) FILTER (WHERE pv.db = 'MSH') as mesh_hierarchy

    FROM input_df src
    LEFT JOIN mapping_hub m ON src.input_icd = m.icd_id
    LEFT JOIN ext_map t ON m.final_sct_id = t.sct_id
    LEFT JOIN ext_vals v ON t.CUI = v.CUI
    LEFT JOIN ext_map pt ON m.parent_id = pt.sct_id
    LEFT JOIN ext_vals pv ON pt.CUI = pv.CUI
    GROUP BY src.input_icd
    """

    return conn.execute(query).df()

if __name__ == "__main__":
    df = get_clean_clinical_mappings(filtered_df['id'])
    print(df.head())

    # Remove rows that have no mapping (in HPO, OMIM, or MESH)
    mapping_columns = [col for col in df.columns if col not in ['icd_code', 'snomed_ids']]
    mapped_only_df = df.dropna(subset=mapping_columns, how='all').copy()

    # Remove the HP: prefix
    hpo = ['hpo_direct', 'hpo_hierarchy']
    mapped_only_df[hpo] = mapped_only_df[hpo].replace('HP:', '', regex=True)

    save_path = os.path.join(base_data_dir, 'utils', 'icd_map.csv')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mapped_only_df.to_csv(save_path, index=False)
    print(f"Successfully saved cleanly mapped ICD data to {save_path}")

    mapped_only_df.to_csv(os.path.join(base_data_dir, 'icd_map.csv'), index=False)