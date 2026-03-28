import os, sys

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from shared_functions.global_functions import *

doid_path = os.path.join(data_dir, 'DOID')

do = pd.read_csv(os.path.join(doid_path, 'DOreports', 'DO.csv'))
lookup = pd.read_csv(os.path.join(doid_path, 'DOreports', 'ref.csv'))

# Connect Nodes
    merge = pd.merge(nodes, lookup, on = 'name', how = 'right')
    merge = merge[merge['id_x'].isna()]
    merge = merge.drop('id_x', axis = 1)
    merge['doid_id'] = merge['doid_id'].apply(lambda x: x.split(':')[1])

    start_idx = 0
    BATCH_SIZE = 500

    query = """
        UNWIND $rows AS row

        MERGE (d:Disease:Test:DO {id: row.id})
        SET d.name = row.name,
            d.uml_id = row.uml_id,
            d.mesh_id = row.mesh_id,
            d.omim_id = row.omim_id,
            d.doid_id = row.doid_id
        """

    # Process in batches
    for i in tqdm(range(start_idx, len(merge), BATCH_SIZE), desc="Batch processing"):

        batch = merge.iloc[i:i+BATCH_SIZE]

        rows = []
        for _, row in batch.iterrows():
            rows.append({
                "id": row["id_y"],
                "name": row["name"],
                "uml_id": row["uml_id"] if pd.notna(row["uml_id"]) else None,
                "mesh_id": row["mesh_id"] if pd.notna(row["mesh_id"]) else None,
                "omim_id": row["omim_id"] if row["omim_id"] else [],
                "doid_id": row["doid_id"] if row["doid_id"] else []
            })

        dml_ddl_neo4j(
            query,
            progress=False,
            rows=rows
        )

    # Add edges
    query = """
        UNWIND $rows AS row

        MATCH (d1:Disease:Test:DO {id: row.parent_id})
        MATCH (d2:Disease:Test:DO {id: row.child_id})

        WITH d1, d2
        WHERE d1.id < d2.id   // prevents duplicate reverse edges

        MERGE (d1)-[:IS_A]->(d2)
        """

    rows = []
    for _, row in do.iterrows():
        rows.append({
            "parent_id": row["parent_id"].split(':')[1],
            "child_id": row["child_id"].split(':')[1]
        })

    BATCH_SIZE = 500
    for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Adding IS_A edges"):
        dml_ddl_neo4j(query, progress=False, rows=rows[i:i+BATCH_SIZE])

## Connect Edges

    name_to_id = lookup.set_index('name')['id']
    do['parent_id'] = do['parents'].map(name_to_id)

    query = """
    UNWIND $rows AS row

    MATCH (d:Test {id: row.id})

    WITH d, row

    UNWIND row.parent_id AS inter_id
    MATCH (d2:Test {id: inter_id})
    MERGE (d)-[:CHILD_OF]->(d2)
    """

    # Process in batches
    for i in tqdm(range(start_idx, len(do), BATCH_SIZE), desc="Batch processing"):

        batch = do.iloc[i:i+BATCH_SIZE]

        rows = []
        for _, row in batch.iterrows():
            rows.append({
                "id": row["id"],
                "parent_id": row["parent_id"] if row["parent_id"] else None
            })

        dml_ddl_neo4j(
            query,
            progress=False,
            rows=rows
        )