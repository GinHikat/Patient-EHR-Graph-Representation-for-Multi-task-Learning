import pandas as pd 
import sys, os
from tqdm import tqdm as tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from shared_functions.global_functions import *

# Start by connecting drug first

    drug = pd.read_csv(os.path.join(sider_path, 'drug_lookup.csv'), index_col=0)

    drug['drug_name'] = drug['drug_name'].apply(lambda x: x.title())

    drug.columns = ['id', 'name']

    match_drug = pd.merge(base_drug, drug, on = 'name', how = 'inner')

    query = """
        UNWIND $rows AS row
        MATCH (drug:Drug:Test {id: row.id})
        SET drug.pubchem_id = row.pubchem_id
        """

    rows = [
        {"id": row["id_x"], "pubchem_id": row["id_y"]}
        for _, row in match_drug.iterrows()
    ]

    BATCH_SIZE = 500
    for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Processing"):
        dml_ddl_neo4j(query, progress=False, rows=rows[i:i+BATCH_SIZE])

## Add drugs that's not in the database 

    query = """
        UNWIND $rows AS row
        MERGE (drug:Drug:Test:PubChem {id: row.id})
        SET drug.pubchem_id = row.pubchem_id,
            drug.name = row.name
        """

    rows = [
        {"id": row["id_y"], "pubchem_id": row["id_y"], 'name': row['name']}
        for _, row in match_drug.iterrows()
    ]

    BATCH_SIZE = 500
    for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Processing"):
        dml_ddl_neo4j(query, progress=False, rows=rows[i:i+BATCH_SIZE])

### Do the same with Disease

    sider = pd.read_csv(os.path.join(sider_path, 'drug_disease.csv'), index_col = 0)

    disease = sider[['disease_umls_id', 'disease_name']].drop_duplicates()

    disease['disease_name'] = disease['disease_name'].apply(lambda x: x.title())

    disease.columns = ['uml_id', 'name']

    dis_match = pd.merge(base_dis, disease, on = 'name', how = 'inner')

    query = """
            UNWIND $rows AS row
            MATCH (drug:Disease:Test {id: row.id})
            SET drug.uml_id = row.uml_id
            """

    rows = [
        {"id": row["id"], "uml_id": row["uml_id"]}
        for _, row in dis_match.iterrows()
    ]

    BATCH_SIZE = 500
    for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Processing"):
        dml_ddl_neo4j(query, progress=False, rows=rows[i:i+BATCH_SIZE])

    
    # New nodes
    
    dis_match = pd.merge(base_dis, disease, on = 'name', how = 'right')

    dis_match = dis_match[dis_match['id'].isna()]

    dis_match = dis_match[['uml_id', 'name']]

#### Connect with ADR

    drug_adr = pd.read_csv(os.path.join(sider_path, 'drug_side_effect.csv'), index_col=0)

    adr = drug_adr[['side_effect_id', 'side_effect']]

    adr.columns = ['uml_id', 'name']

    adr = adr.drop_duplicates(subset = 'uml_id')

    adr['name'] = adr['name'].apply(lambda x: x.title())

    # add nodes

        match_dis = pd.merge(base_dis, adr, on = 'name', how = 'inner')

        match_dis['uml_id_x'] = match_dis['uml_id_x'].fillna(match_dis['uml_id_y'])

        nice = match_dis[match_dis['uml_id_x'] == match_dis['uml_id_y']].drop_duplicates(subset = 'id')

        not_nice = match_dis[~match_dis['id'].isin(nice['id'])].drop_duplicates(subset = 'id')

        not_nice = not_nice.drop(['uml_id_x'], axis = 1)
        nice = nice.drop(['uml_id_x'], axis = 1)

        final = pd.concat([nice, not_nice], axis = 0)

        query = """
            UNWIND $rows AS row
            MATCH (drug:Disease:Test {id: row.id})
            SET drug.uml_id = row.uml_id
            """

        rows = [
            {"id": row["id"], "uml_id": row["uml_id_y"]}
            for _, row in final.iterrows()
        ]

        BATCH_SIZE = 500
        for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Processing"):
            dml_ddl_neo4j(query, progress=False, rows=rows[i:i+BATCH_SIZE])

    # add new nodes

        match_dis = pd.merge(base_dis, adr, on = 'name', how = 'right')

        match_dis = match_dis[match_dis['id'].isna()]

        query = """
            UNWIND $rows AS row
            MERGE (drug:Disease:Test:PubChem {id: row.id})
            SET drug.uml_id = row.uml_id,
                drug.name = row.name
            """

        rows = [
            {"id": row["uml_id_y"], "uml_id": row["uml_id_y"], 'name': row['name']}
            for _, row in final.iterrows()
        ]

        BATCH_SIZE = 500
        for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Processing"):
            dml_ddl_neo4j(query, progress=False, rows=rows[i:i+BATCH_SIZE])

    # Connect Drugs with ADR

        drug_adr = pd.read_csv(os.path.join(sider_path, 'drug_side_effect.csv'), index_col=0)

        drug_adr = drug_adr.drop_duplicates(subset = ['drug_id', 'side_effect_id'])

        query = """
            UNWIND $rows AS row
            MATCH (drug: Drug:Test {pubchem_id: row.id})
            MATCH (side:Disease: Test {uml_id: row.se})
            MERGE (drug)-[:CAUSE]->(side)
        """

        rows = [
            {"id": row["drug_id"], "se": row["side_effect_id"]}
            for _, row in drug_adr.iterrows()
        ]

        BATCH_SIZE = 1000

        for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Creating CAUSE relationships"):
            batch = rows[i:i+BATCH_SIZE]
            dml_ddl_neo4j(query, progress=False, rows=batch)