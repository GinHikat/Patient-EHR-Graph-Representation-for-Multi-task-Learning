import pandas as pd
import numpy as np
import sys, os
from tqdm import tqdm as tqdm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from shared_functions.global_functions import *

# Start with ingesting raw Phenotypes
    hpo = pd.read_csv(os.path.join(hpo_path, 'cleaned.csv'))

    list_col = ['alternative_name', 'Is_a', 'snomed_id', 'umls_id']

    def parse_list(x):

        # already list or array
        if isinstance(x, (list, np.ndarray)):
            return list(x)

        # missing value
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return []

        # string 
        if isinstance(x, str):
            return ast.literal_eval(x)

        return []

    for col in list_col:
        hpo[col] = hpo[col].apply(parse_list)

    hpo['description'] = hpo['description'].apply(lambda x: re.sub(r'"\s*\[.*$', '"', str(x)))
    hpo['description'] = hpo['description'].apply(lambda x: x.replace('"', ''))

    # Enforce constraint on the Phenotype
    query = """
    CREATE CONSTRAINT phenotype_id IF NOT EXISTS
    FOR (p:Disease)
    REQUIRE p.id IS UNIQUE
    """

    dml_ddl_neo4j(query)

    # Processing loop
    for _, row in tqdm(hpo.iterrows(),
                    total=len(hpo),
                    desc="Creating nodes + relations"):

        child_id = row["id"]
        parents = row["Is_a"]

        query = """
        MERGE (child:Disease:Test {id: $child_id})
        SET child.name = $name,
            child.description = $description,
            child.alternative_name = $alt_name

        WITH child

        UNWIND $parents AS parent_id
        MERGE (parent:Disease:Test {id: parent_id})
        MERGE (child)-[:CHILD_OF]->(parent)
        """

        dml_ddl_neo4j(
            query,
            progress = False,
            child_id=row["id"],
            name=row["phenotype"],
            description=row["description"],
            alt_name=row["alternative_name"],
            parents=row["Is_a"] if row["Is_a"] else []
        )

## Normalize all name to Title (capitalize all words) and remove Null names

    query = """
    MATCH (p:Test)
    WHERE p.name IS NOT NULL
    SET p.name = apoc.text.capitalizeAll(toLower(p.name))

    MATCH (p:Test)
    WHERE p.name IS NULL
    DETACH DELETE p

    MATCH (n:Disease:Test)
    SET n:HPO;
    """
    dml_ddl_neo4j(query)

### Enrich with attributes from PubMed

    #disease_pubmed is only Diseases in merged_lookup from bc5cdr
    disease_combine = pd.merge(disease_hpo, disease_pubmed, on = 'name', how = 'inner')

    disease_combine = disease_combine.drop_duplicates(subset = 'name')

    disease_combine['MeSH'] = disease_combine['id_y'].apply(lambda x: 1 if x[0] == 'D' else 0)

    # Add to existing HPO label
    query = """
    UNWIND $rows AS row
    MATCH (n {id: row.id_x})
    SET n.MeSH_id  = CASE WHEN row.MeSH = 1 THEN row.id_y ELSE n.MeSH_id END,
        n.OMIM_id  = CASE WHEN row.MeSH = 0 THEN row.id_y ELSE n.OMIM_id END
    """

    rows = [
        {"id_x": row["id_x"], "id_y": row["id_y"], "MeSH": row["MeSH"]}
        for _, row in disease_combine.iterrows()
    ]

    BATCH_SIZE = 500
    for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Adding MeSH/OMIM IDs"):
        dml_ddl_neo4j(query, progress=False, rows=rows[i:i+BATCH_SIZE])
        