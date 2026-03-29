from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict
import duckdb

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from shared_functions.gg_sheet_drive import *
from modules.dataset_preprocessing.utils import *

# Start by simply parsing dimensional tables

# Path definitions
data_dir = os.getenv('DATA_DIR')
mimic_path = os.path.join(data_dir, 'mimic_iv')
hosp = os.path.join(mimic_path, 'hosp')
icu = os.path.join(mimic_path, 'icu')

start_idx = 0
BATCH_SIZE = 500

# hosp/d_labitems

    d_labitem = read_full('hosp', 'd_labitems')
    title(d_labitem, 'label')
    d_labitem['fluid'] = d_labitem['fluid'].apply(lambda x: 'Other Body Fluid' if x in ['Q', 'Fluid', 'I'] else x)

    query = """
        UNWIND $rows AS row

        MATCH (d:Lab:Item:Test:MIMIC {id: row.id})
        SET d.name = row.label,
            d.fluid = row.fluid,
            d.category = row.category
        """

    # Process in batches
    for i in tqdm(range(start_idx, len(d_labitem), BATCH_SIZE), desc="Batch processing"):

        batch = d_labitem.iloc[i:i+BATCH_SIZE]

        rows = []
        for _, row in batch.iterrows():
            rows.append({
                "id": row["itemid"],
                "label": row["label"],
                "fluid": row["fluid"] if pd.notna(row["fluid"]) else None,
                "category": row["category"] if pd.notna(row["category"]) else None,
            })

        dml_ddl_neo4j(
            query,
            progress=False,
            rows=rows
        )

# hosp/d_icd_diagnoses

    d_icd_diagnoses = read_full('hosp', 'd_icd_diagnoses')
    title(d_icd_diagnoses, 'long_title')
    query = """
        UNWIND $rows AS row

        MERGE (d:Test:Diagnosis:ICD:MIMIC {id: row.code})
        SET d.name = row.long_title,
            d.icd = row.icd,
            d.icd_ver = row.icd_ver
        """

    # Process in batches
    for i in tqdm(range(start_idx, len(d_icd_diagnoses), BATCH_SIZE), desc="Batch processing"):

        batch = d_icd_diagnoses.iloc[i:i+BATCH_SIZE]

        rows = []
        for _, row in batch.iterrows():
            rows.append({
                "code": row["icd_code"],
                "long_title": row["long_title"],
                'icd': row['icd_code'],
                'icd_ver': row['icd_version']
            })

        dml_ddl_neo4j(
            query,
            progress=False,
            rows=rows
        )

# hosp/d_icd_procedures

    d_icd_procedures = read_full('hosp', 'd_icd_procedures')
    title(d_icd_procedures, 'long_title')
    d_icd_procedures = d_icd_procedures.drop_duplicates(subset = 'icd_code')

    query = """
        UNWIND $rows AS row

        MERGE (d:Test:Procedure:ICD:MIMIC {id: row.code})
        SET d.name = row.long_title,
            d.icd = row.icd,
            d.icd_ver = row.icd_ver
        """

    # Process in batches
    for i in tqdm(range(start_idx, len(d_icd_procedures), BATCH_SIZE), desc="Batch processing"):

        batch = d_icd_procedures.iloc[i:i+BATCH_SIZE]

        rows = []
        for _, row in batch.iterrows():
            rows.append({
                "code": row["icd_code"],
                "long_title": row["long_title"],
                'icd': row['icd_code'],
                'icd_ver': row['icd_version']
            })

        dml_ddl_neo4j(
            query,
            progress=False,
            rows=rows
        )

# icu/d_items

    d_items = read_full('icu', 'd_items')
    title(d_items, 'label')
    d_items = d_items.drop(['lownormalvalue', 'highnormalvalue'], axis = 1)
    d_items['similar'] = d_items['label'].str.lower() == d_items['abbreviation'].str.lower()

    query = """
        UNWIND $rows AS row

        MERGE (d:Test:Item:MIMIC:ICU {id: row.itemid})
        SET d.name = row.label,
            d.category = row.category,
            d.alias = row.alias,
            d.related = row.related,
            d.unit = row.unitname,
            d.dtype = row.type
        """

    # Process in batches
    for i in tqdm(range(start_idx, len(d_items), BATCH_SIZE), desc="Batch processing"):

        batch = d_items.iloc[i:i+BATCH_SIZE]

        rows = []
        for _, row in batch.iterrows():
            rows.append({
                "itemid": row["itemid"],
                "label": row["label"],
                "category": row["category"] if pd.notna(row["category"]) else None,
                "alias": row["abbreviation"] if row["similar"] == False else None,
                "unitname": row["unitname"] if pd.notna(row["unitname"]) else None,
                "type": row["param_type"] if pd.notna(row["param_type"]) else None
            })

        dml_ddl_neo4j(
            query,
            progress=False,
            rows=rows
        )

