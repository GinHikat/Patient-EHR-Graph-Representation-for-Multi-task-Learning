import pandas as pd
import sys, os
import re
from tqdm import tqdm

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
    
    nodes = pd.read_csv(os.path.join(data_dir, 'nodes.csv'))
    edges = pd.read_csv(os.path.join(data_dir, 'edges.csv'))

    # Node creations
    records = nodes.to_dict(orient="records")

    query = """
    UNWIND $rows AS row
    MERGE (n {id: row.id})
    SET n += row.props
    WITH n, row
    CALL apoc.create.addLabels(n, row.labels) YIELD node
    RETURN count(*)
    """

    # Prepare batches
    for i in tqdm(range(0, len(records), batch_size), desc="Creating nodes"):
        batch = records[i:i+batch_size]

        # Split props & labels
        rows = []
        for r in batch:
            r = dict(r)
            labels = r.pop("labels", "")
            node_id = r.pop("id")

            rows.append({
                "id": node_id,
                "labels": labels.split(":") if labels else [],
                "props": r
            })

        query_neo4j(query, rows=rows)

    print("All Nodes created")

    # Edges adding
    records = edges.to_dict(orient="records")

    query = """
    UNWIND $rows AS row
    MATCH (a {id: row.source})
    MATCH (b {id: row.target})
    CALL apoc.create.relationship(a, row.type, row.props, b) YIELD rel
    RETURN count(*)
    """

    for i in tqdm(range(0, len(records), batch_size), desc="Creating edges"):
        batch = records[i:i+batch_size]

        rows = []
        for r in batch:
            r = dict(r)
            source = r.pop("source")
            target = r.pop("target")
            rel_type = r.pop("type")

            rows.append({
                "source": source,
                "target": target,
                "type": rel_type,
                "props": r
            })

        query_neo4j(query, rows=rows)

    print("All Edges created")

if __name__ == '__main__':
    snapshot_node()
    # snapshot_edge()