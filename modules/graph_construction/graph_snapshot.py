import pandas as pd
import sys, os
import re
from tqdm import tqdm
import json
from collections import defaultdict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from shared_functions.global_functions import *
from shared_functions.global_functions import query_neo4j, driver, DATABASE, dml_ddl_neo4j

data_dir = os.path.join(project_root, 'data')

def snapshot_node(namespace='Test'):
    '''
    Take a snapshot of all nodes properties in the specific namespace for backup

    Input: 
        namespace: label of nodes, working as namespace of the Graph to separate between different universes
    Output:
        Save to nodes.csv with node id, node labels and all attributes
    '''

    # Validate label
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", namespace):
        raise ValueError("Invalid label name")

    # Total count for progress bar
    count_query = f"""
    MATCH (n:{namespace})
    RETURN count(n) AS total
    """
    total = query_neo4j(count_query)[0]["total"]

    # Fetch nodes
    query_nodes = f"""
    MATCH (n:{namespace})
    RETURN n.id AS id, labels(n) AS labels, properties(n) AS props
    """
    node_records = query_neo4j(query_nodes)

    # Collect all possible property keys
    all_keys = set()
    for record in tqdm(node_records, desc="Collecting keys", total=total):
        all_keys.update(record["props"].keys())

    all_keys = sorted(all_keys)

    # Normalize nodes
    rows = []
    for record in tqdm(node_records, desc="Processing nodes", total=total):
        row = {
            "id": record["id"],
            "labels": ":".join(record["labels"])
        }

        props = record["props"]
        for key in all_keys:
            row[key] = props.get(key, None)

        rows.append(row)

    # Save CSV
    df_nodes = pd.DataFrame(rows)
    df_nodes.to_csv(os.path.join(data_dir, "nodes.csv"), index=False)

    print("Nodes CSV saved as nodes.csv")

def snapshot_edge(namespace='Test'):
    '''
    Take a snapshot of all edges in the specific namespace for backup

    Input: 
        namespace: label of nodes, working as namespace of the Graph to separate between different universes
    Output:
        Save to edges.csv with head/tail id and edge label
    '''

    # Validate label
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", namespace):
        raise ValueError("Invalid label name")

    # Total count for progress bar
    count_query = f"""
    MATCH (a:{namespace})-[r]->(b:{namespace})
    RETURN count(r) AS total
    """
    total = query_neo4j(count_query)[0]["total"]

    # Fetch edges
    query_edges = f"""
    MATCH (a:{namespace})-[r]->(b:{namespace})
    RETURN 
        coalesce(a.id, elementId(a)) AS source,
        coalesce(b.id, elementId(b)) AS target,
        type(r) AS type,
        properties(r) AS props
    """

    edge_records = query_neo4j(query_edges)

    # Collect all relationship property keys
    all_edge_keys = set()
    for record in tqdm(edge_records, desc="Collecting edge keys", total=total):
        all_edge_keys.update(record["props"].keys())

    all_edge_keys = sorted(all_edge_keys)

    # Normalize edges
    edge_rows = []
    for record in tqdm(edge_records, desc="Processing edges", total=total):
        row = {
            "source": record["source"],
            "target": record["target"],
            "type": record["type"]
        }

        props = record["props"]
        for key in all_edge_keys:
            row[key] = props.get(key, None)

        edge_rows.append(row)

    df_edges = pd.DataFrame(edge_rows)
    df_edges.to_csv(os.path.join(data_dir, "edges.csv"), index=False)

    print("Edges CSV saved as edges.csv")

def graph_recreation(namespace = 'Test'):
    '''
    Recreate the graph with the Nodes and Edges already backed up

    Input: 
        namespace: target namespace
    Output:
        New Graph with all nodes and edges 
    '''
    batch_size = 500

    print(f"Ensuring index and uniqueness constraint on {namespace}(id)...")
    try:
        query_neo4j(f"CREATE INDEX node_id_idx_{namespace} IF NOT EXISTS FOR (n:{namespace}) ON (n.id)")
        query_neo4j(f"CREATE CONSTRAINT node_id_unique_{namespace} IF NOT EXISTS FOR (n:{namespace}) REQUIRE n.id IS UNIQUE")
    except Exception as e:
        print(f"Note on constraints: {e}")

    print("Loading nodes and edges from CSV...")
    # Performance: low_memory=False and dtype=str prevent mixed-type warnings and type mismatch in DB
    nodes = pd.read_csv(os.path.join(data_dir, 'nodes.csv'), low_memory=False, dtype={'id': str})
    edges = pd.read_csv(os.path.join(data_dir, 'edges.csv'), low_memory=False, dtype={'source': str, 'target': str})

    # Granular Node Resume: Group nodes by label combination
    print("Grouping nodes from CSV...")

    node_groups = defaultdict(list)
    
    # Use itertuples for better performance and memory efficiency than to_dict(records)
    for r in tqdm(nodes.itertuples(index=False), total=len(nodes), desc="Grouping nodes"):
        r_dict = r._asdict()
        labels_str = r_dict.pop("labels", "")
        # Filter out NaN/null properties if any to keep DB clean
        clean_props = {k: v for k, v in r_dict.items() if pd.notna(v) and k != 'id'}
        
        # Parse labels and deduplicate
        # Restore parsing from CSV string
        labels_list = [str(l).strip() for l in str(labels_str).split(":") if str(l).strip()]
        labels_list = list(dict.fromkeys(labels_list)) # Deduplicate keeping order

        # 1. Identify components based on CSV structure: [Entity, Test, Database]
        entity_types = ["Drug", "Disease"]
        found_entity_type = next((l for l in labels_list if l in entity_types), "Entity")
        db_labels = [l for l in labels_list if l != namespace and l not in entity_types]
        found_db = db_labels[0] if db_labels else "Source"

        # 2. Reconstruct exactly 3 labels in order: [Entity, Namespace, Database]
        labels_list = [found_entity_type, namespace, found_db]
        
        # We preserve the order in the key for grouping
        labels_key = ":".join(labels_list)
        node_groups[labels_key].append({"id": r_dict['id'], "props": clean_props})

    print("Checking database for node status (exact label matching)...")
    from shared_functions.global_functions import dml_ddl_neo4j
    
    for labels_key, rows in node_groups.items():
        # labels_key is [Type, Test, Database]
        label_clause = ":" + labels_key if labels_key else ""
        
        # Count nodes that have the required labels
        count_query = f"MATCH (n{label_clause}) RETURN count(n) as total"
        res = query_neo4j(count_query)
        db_count = res[0]['total']
        
        if db_count >= len(rows):
            print(f"  - Group ({labels_key}): {len(rows)} nodes verified. Skipping.")
            continue
        
        print(f"  - Group ({labels_key}): {db_count}/{len(rows)} nodes exist. Syncing full group...")
        
        # To enforce order and exactly 3 labels, we clear ALL possible labels first,
        # INCLUDING the namespace itself, so that the order in SET is respected.
        common_labels = f"{namespace}:Drug:Disease:HPO:CTD:PubMed:DrugBank:DB:Source:Entity:Node:Unspecified"
        node_query = f"""
        UNWIND $rows AS row 
        MERGE (n:{namespace}{{id: row.id}}) 
        SET n += row.props 
        REMOVE n:{common_labels}
        SET n{label_clause} 
        RETURN count(*)
        """
        
        for k in tqdm(range(0, len(rows), batch_size), desc=f"Syncing {labels_key}"):
            dml_ddl_neo4j(node_query, progress=False, rows=rows[k:k+batch_size])


    print(f"Final physical node count for '{namespace}': {query_neo4j(f'MATCH (n:{namespace}) RETURN count(n) as t')[0]['t']}")

    # Granular Edge Resume: Check counts by relationship type
    print("Grouping edges from CSV...")
    edge_groups = defaultdict(list)
    for r in tqdm(edges.itertuples(index=False), total=len(edges), desc="Grouping edges"):
        r_dict = r._asdict()
        rel_type = r_dict.pop("type")
        source = r_dict.pop("source")
        target = r_dict.pop("target")
        clean_props = {k: v for k, v in r_dict.items() if pd.notna(v)}
        edge_groups[rel_type].append({"source": source, "target": target, "props": clean_props})

    print("Checking database for edge status...")
    for rel_type, rows in edge_groups.items():
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(rel_type)):
            print(f"  - Skipping invalid relationship type: {rel_type}")
            continue
            
        # Count edges within the namespace context
        count_query = f"MATCH (a:{namespace})-[r:{rel_type}]->(b:{namespace}) RETURN count(r) as total"
        res = query_neo4j(count_query)
        db_count = res[0]['total']
        
        if db_count >= len(rows):
            print(f"  - Type '{rel_type}': {len(rows)} edges verified. Skipping.")
            continue
            
        print(f"  - Type '{rel_type}': {db_count}/{len(rows)} edges exist. Syncing full group...")
        edge_query = f"UNWIND $rows AS row MATCH (a:{namespace} {{id: row.source}}) MATCH (b:{namespace} {{id: row.target}}) MERGE (a)-[r:{rel_type}]->(b) SET r += row.props RETURN count(*)"
        
        for k in tqdm(range(0, len(rows), batch_size), desc=f"Syncing {rel_type}"):
            dml_ddl_neo4j(edge_query, progress=False, rows=rows[k:k+batch_size])

    print("Graph recreation complete.")
    final_edge_count = query_neo4j(f"MATCH (a:{namespace})-[r]->(b:{namespace}) RETURN count(r) as t")[0]['t']
    print(f"Total relationships in database for '{namespace}': {final_edge_count}")

def clear_database():
    '''Fast deletion of all nodes and relationships in the database'''
    print("Deleting all data...")
    query = "MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS"
    
    # CALL { ... } IN TRANSACTIONS requires an implicit transaction.
    # driver.execute_query() uses managed transactions, which will fail.
    try:
        with driver.session(database=DATABASE) as session:
            session.run(query)
        print("Database cleared.")
    except Exception as e:
        print(f"Error clearing database: {e}")
        print("Falling back to standard delete (may be slow/fail on large DBs)...")
        query_neo4j("MATCH (n) DETACH DELETE n")
        print("Database cleared (fallback).")

if __name__ == '__main__':

    snapshot_node()
    snapshot_edge()
    # clear_database()
    # graph_recreation()