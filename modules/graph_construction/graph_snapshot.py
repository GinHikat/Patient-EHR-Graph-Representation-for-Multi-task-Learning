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
        labels_list = [l.strip() for l in labels_str.split(":") if l.strip()] if isinstance(labels_str, str) else []
        labels_list = list(dict.fromkeys(labels_list)) # Deduplicate keeping order

        # ENFORCE EXACTLY 3 LABELS:
        # 1. Ensure namespace is present
        if namespace not in labels_list:
            labels_list.append(namespace)
        
        # 2. Add placeholders if less than 3
        placeholders = ["Entity", "Node", "Unspecified"]
        for p in placeholders:
            if len(labels_list) >= 3: break
            if p not in labels_list:
                labels_list.append(p)
        
        # 3. Truncate if more than 3 (always keep namespace)
        if len(labels_list) > 3:
            # Keep namespace and the first 2 other labels
            other_labels = [l for l in labels_list if l != namespace]
            labels_list = [namespace] + other_labels[:2]

        labels_key = ":".join(sorted(labels_list))
        node_groups[labels_key].append({"id": r_dict['id'], "props": clean_props})

    print("Checking database for node status (exact label matching)...")
    for labels_key, rows in node_groups.items():
        labels_list = labels_key.split(":") if labels_key else []
        label_clause = ""
        if labels_key:
            safe_labels = [lbl for lbl in labels_list if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", lbl)]
            if safe_labels: label_clause = ":" + ":".join(safe_labels)
        
        # Count nodes that have the required labels (flexible on extra labels)
        count_query = f"MATCH (n{label_clause}) RETURN count(n) as total"
        res = query_neo4j(count_query)
        db_count = res[0]['total']
        
        if db_count >= len(rows):
            print(f"  - Group ({labels_key or 'No Label'}): {len(rows)} nodes verified. Skipping.")
            continue
        
        print(f"  - Group ({labels_key or 'No Label'}): {db_count}/{len(rows)} nodes exist. Syncing full group to ensure consistency...")
        
        node_query = f"UNWIND $rows AS row MERGE (n:{namespace}{{id: row.id}}) SET n += row.props {'SET n' + label_clause if label_clause else ''} RETURN count(*)"
        
        for k in tqdm(range(0, len(rows), batch_size), desc=f"Syncing {labels_key or 'No Label'}"):
            query_neo4j(node_query, rows=rows[k:k+batch_size])

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
            query_neo4j(edge_query, rows=rows[k:k+batch_size])

    print("Graph recreation complete.")
    final_edge_count = query_neo4j(f"MATCH (a:{namespace})-[r]->(b:{namespace}) RETURN count(r) as t")[0]['t']
    print(f"Total relationships in database for '{namespace}': {final_edge_count}")

def clear_database():
    '''Fast deletion of all nodes and relationships in the database'''
    print("Deleting all data...")
    query = "MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS"
    query_neo4j(query)
    print("Database cleared.")

if __name__ == '__main__':

    # snapshot_node()
    # snapshot_edge()
    # clear_database()
    graph_recreation()