
# Match between current and CTD Drugs, if match then add with more attributes

    drug_match = pd.merge(base_drug, chem_df, on = 'name', how = 'inner')

    drug_match['mesh_id'] = drug_match['id_y']

    drug_match['description'] = drug_match['description'].fillna(drug_match['definition'])

    drug_match['aliases'] = drug_match['aliases'].fillna(drug_match['alias'])

    drug_match = drug_match[['id_x', 'aliases', 'description', 'mesh_id', 'name']]

    BATCH_SIZE = 300

    query = """
        UNWIND $rows AS row

        MERGE (d {id: row.id})
        SET d.alias = row.alias,
            d.description = row.description,
            d.mesh_id = row.mesh_id
        """

    for i in tqdm(range(0, len(drug_match), BATCH_SIZE), desc="Batch processing"):

        batch = drug_match.iloc[i:i+BATCH_SIZE]

        rows = []
        for _, row in batch.iterrows():
            rows.append({
                "id": row["id_x"],
                "alias": row["aliases"],
                "description": row["description"] if pd.notna(row["description"]) else None,
                "mesh_id": row["mesh_id"]
            })

        dml_ddl_neo4j(
            query,
            progress=False,
            rows=rows
        )

## If not match, then add new node 

    drug_match = pd.merge(base_drug, chem_df, on = 'name', how = 'right')

    drug_match = drug_match[['name', 'id_y', 'definition', 'alias_y']]

    BATCH_SIZE = 1000

    query = """
        UNWIND $rows AS row

        MERGE (d:Test:Drug:CTD {id: row.id})
        SET d.alias = row.alias,
            d.description = row.description,
            d.mesh_id = row.mesh_id,
            d.name = row.name
        """

    for i in tqdm(range(0, len(drug_match), BATCH_SIZE), desc="Batch processing"):

        batch = drug_match.iloc[i:i+BATCH_SIZE]

        rows = []
        for _, row in batch.iterrows():
            rows.append({
                "id": row["id_y"],
                "alias": row["alias_y"],
                "description": row["definition"] if pd.notna(row["definition"]) else None,
                "mesh_id": row["id_y"],
                'name': row['name']
            })

        dml_ddl_neo4j(
            query,
            progress=False,
            rows=rows
        )

### Continue with Disease, if match then add with more attributes

    dis_match = pd.merge(base_dis, dis_df, on = 'name', how = 'inner')

    dis_match['description'] = dis_match['description'].fillna(dis_match['definition'])

    dis_match['alias_x'] = dis_match['alias_x'].fillna(dis_match['alias_y'])

    dis_match['mesh_id'] = dis_match['mesh_id'].fillna(dis_match['id_y'])

    dis_match = dis_match[['id_x', 'alias_x', 'description', 'mesh_id', 'name', 'DOID', 'OMIM']]

    BATCH_SIZE = 300

    query = """
        UNWIND $rows AS row

        MERGE (d {id: row.id})
        SET d.alias = row.alias,
            d.description = row.description,
            d.mesh_id = row.mesh_id,
            d.omim_id = row.omim_id,
            d.doid_id = row.doid_id
        """

    for i in tqdm(range(0, len(dis_match), BATCH_SIZE), desc="Batch processing"):

        batch = dis_match.iloc[i:i+BATCH_SIZE]

        rows = []
        for _, row in batch.iterrows():
            rows.append({
                "id": row["id_x"],
                "alias": row["alias_x"],
                "description": row["description"] if pd.notna(row["description"]) else None,
                "mesh_id": row["mesh_id"],
                "omim_id": row["OMIM"] if row["OMIM"] else [],
                "doid_id": row["DOID"] if row["DOID"] else []
            })

        dml_ddl_neo4j(
            query,
            progress=False,
            rows=rows
        )

#### If not match, then add new Disease Node

    dis_match = pd.merge(base_dis, dis_df, on = 'name', how = 'right')

    dis_match = dis_match[dis_match['id_x'].isna()]

    dis_match = dis_match.drop(['id_x', 'labels', 'alias_x', 'description', 'mesh_id'], axis = 1)

    BATCH_SIZE = 500

    query = """
        UNWIND $rows AS row

        MERGE (d: Disease: Test: CTD {id: row.id})
        SET d.alias = row.alias,
            d.description = row.description,
            d.mesh_id = row.mesh_id,
            d.omim_id = row.omim_id,
            d.doid_id = row.doid_id,
            d.name = row.name
        """

    for i in tqdm(range(0, len(dis_match), BATCH_SIZE), desc="Batch processing"):

        batch = dis_match.iloc[i:i+BATCH_SIZE]

        rows = []
        for _, row in batch.iterrows():
            rows.append({
                "id": row["id_y"],
                "alias": row["alias_y"],
                "description": row["definition"] if pd.notna(row["definition"]) else None,
                "mesh_id": row["id_y"],
                "omim_id": row["OMIM"] if row["OMIM"] else [],
                "doid_id": row["DOID"] if row["DOID"] else [],
                'name': row['name']
            })

        dml_ddl_neo4j(
            query,
            progress=False,
            rows=rows
        )

##### Start connecting the Drugs and Diseases together by Ontology

    # Start with indexing
    query_neo4j("CREATE INDEX IF NOT EXISTS FOR (n:Disease) ON (n.id)")
    query_neo4j("CREATE INDEX IF NOT EXISTS FOR (n:Disease) ON (n.mesh_id)")

    dis_df['parent'] = dis_df['parent'].apply(lambda x: ast.literal_eval(x))

    query = """
        UNWIND $rows AS row
        MATCH (child {mesh_id: row.id})
        UNWIND row.parents AS parent_id
        MATCH (parent:Disease {mesh_id: parent_id})
        MERGE (child)-[:CHILD_OF]->(parent)
    """

    rows = [
        {"id": row["id"], "parents": row["parent"] if isinstance(row["parent"], list) else []}
        for _, row in dis_df.iterrows()
    ]

    BATCH_SIZE = 500

    for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Creating CHILD_OF relationships"):
        batch = rows[i:i+BATCH_SIZE]
        dml_ddl_neo4j(query, progress=False, rows=batch)

