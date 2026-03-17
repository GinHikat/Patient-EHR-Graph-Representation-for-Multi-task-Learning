import pandas as pd
import numpy as np
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from shared_functions.global_functions import *

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
FOR (p:Phenotype)
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
    MERGE (child:Phenotype:Test {id: $child_id})
    SET child.name = $name,
        child.description = $description,
        child.alternative_name = $alt_name

    WITH child

    UNWIND $parents AS parent_id
    MERGE (parent:Phenotype:Test {id: parent_id})
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

# Normalize all name to Title (capitalize all words) and remove Null names

query = """
MATCH (p:Test)
WHERE p.name IS NOT NULL
SET p.name = apoc.text.capitalizeAll(toLower(p.name))

MATCH (p:Test)
WHERE p.name IS NULL
DETACH DELETE p

MATCH (n:Phenotypes:Test)
SET n:HPO;
"""
dml_ddl_neo4j(query)