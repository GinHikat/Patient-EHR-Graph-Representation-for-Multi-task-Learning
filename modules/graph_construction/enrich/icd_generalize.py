import pandas as pd
import numpy as np
import json
from collections import defaultdict
import re

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from shared_functions.gg_sheet_drive import *
from shared_functions.global_functions import *
from modules.dataset_preprocessing.utils import *
from enrichment import *

from dotenv import load_dotenv
load_dotenv() 

base_data_dir = os.path.join(project_root, 'data')
icd = os.path.join(base_data_dir, 'ICD_lookup')

# MIMIC files
procedure = read_full('hosp', 'd_icd_procedures')
procedure['long_title'] = procedure['long_title'].str.title()
procedure_10 = procedure[procedure['icd_version'] == 10]
procedure_9 = procedure[procedure['icd_version'] == 9]

diagnosis = read_full('hosp', 'd_icd_diagnoses')

diagnosis['long_title'] = diagnosis['long_title'].str.title()

diagnosis_10 = diagnosis[diagnosis['icd_version'] == 10]
diagnosis_9 = diagnosis[diagnosis['icd_version'] == 9]

# Ontology files
diag_10 = os.path.join(icd, 'ICD_10_diagnosis.csv')
proc_10 = os.path.join(icd, 'ICD_10_procedure.csv')
diag_9 = os.path.join(icd, 'ICD_9_diagnosis.csv')
proc_9 = os.path.join(icd, 'ICD_9_procedure.txt')

diagnosis_10, procedure_10 = load_ccsr_mappings(diag_10, proc_10)
diagnosis_10['ccsr_description'] = diagnosis_10['ccsr_description'].str.title()
procedure_10['ccsr_description'] = procedure_10['ccsr_description'].str.title()

diagnosis_9, procedure_9 = load_ccs_mappings(diag_9, proc_9)
diagnosis_9['ccsr_description'] = diagnosis_9['ccsr_description'].str.title()
procedure_9['ccsr_description'] = procedure_9['ccsr_description'].str.title()

procedure_9['ccsr_category'] = procedure_9['ccsr_category'].apply(lambda x: f'ccsr_proc_{x}')
diagnosis_9['ccsr_category'] = diagnosis_9['ccsr_category'].apply(lambda x: f'ccsr_diag_{x}')

## Procedure 9 
    query = """
        UNWIND $rows AS row

        MATCH (d:Procedure:ICD:Test:MIMIC {id: row.id})
        MERGE (c:Procedure:ICD:Test:MIMC {id: row.ccs_id})

        SET c.name = row.category,
            c.icd_ver = 9

        MERGE (d)-[:CHILD_OF]->(c)
        """

    # Process in batches
    for i in tqdm(range(start_idx, len(procedure_9), BATCH_SIZE), desc="Batch processing"):

        batch = procedure_9.iloc[i:i+BATCH_SIZE]

        rows = []
        for _, row in batch.iterrows():
            rows.append({
                "id": row["icd_code"],
                'ccs_id': row['ccs_category'],
                'category': row['ccs_description']
            })

        dml_ddl_neo4j(
            query,
            progress=False,
            rows=rows
        )

### Diagnosis 9
    query = """
        UNWIND $rows AS row

        MATCH (d:Diagnosis:ICD:Test:MIMIC {id: row.id})
        MERGE (c:Diagnosis:ICD:Test:MIMC {id: row.ccs_id})

        SET c.name = row.category,
            c.icd_ver = 9

        MERGE (d)-[:CHILD_OF]->(c)
        """

    # Process in batches
    for i in tqdm(range(start_idx, len(diagnosis_9), BATCH_SIZE), desc="Batch processing"):

        batch = diagnosis_9.iloc[i:i+BATCH_SIZE]

        rows = []
        for _, row in batch.iterrows():
            rows.append({
                "id": row["icd_code"],
                'ccs_id': row['ccs_category'],
                'category': row['ccs_description']
            })

        dml_ddl_neo4j(
            query,
            progress=False,
            rows=rows
        )

#### Procedure 10
    query = """
        UNWIND $rows AS row

        MATCH (d:Procedure:ICD:Test:MIMIC {id: row.id})
        MERGE (c:Procedure:ICD:Test:MIMC {id: row.ccsr_id})

        SET c.name = row.category,
            c.domain = row.domain,
            c.icd_ver = 10

        MERGE (d)-[:CHILD_OF]->(c)
        """

    # Process in batches
    for i in tqdm(range(start_idx, len(procedure_10), BATCH_SIZE), desc="Batch processing"):

        batch = procedure_10.iloc[i:i+BATCH_SIZE]

        rows = []
        for _, row in batch.iterrows():
            rows.append({
                "id": row["icd_code"],
                'ccsr_id': row['ccsr_category'],
                'category': row['ccsr_description'],
                'domain': row['clinical_domain']
            })

        dml_ddl_neo4j(
            query,
            progress=False,
            rows=rows
        )

#### Diagnosis 10
    query = """
        UNWIND $rows AS row

        MATCH (d:Diagnosis:ICD:Test:MIMIC {id: row.id})
        MERGE (c:Diagnosis:ICD:Test:MIMC {id: row.ccsr_id})

        SET c.name = row.category,
            c.domain = row.domain,
            c.icd_ver = 10

        MERGE (d)-[:CHILD_OF]->(c)
        """

    # Process in batches
    for i in tqdm(range(start_idx, len(diagnosis_10), BATCH_SIZE), desc="Batch processing"):

        batch = diagnosis_10.iloc[i:i+BATCH_SIZE]

        rows = []
        for _, row in batch.iterrows():
            rows.append({
                "id": row["icd_code"],
                'ccsr_id': row['ccsr_category'],
                'category': row['ccsr_description']
            })

        dml_ddl_neo4j(
            query,
            progress=False,
            rows=rows
        )

        