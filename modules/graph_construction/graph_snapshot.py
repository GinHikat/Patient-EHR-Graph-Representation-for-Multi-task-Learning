import pandas as pd
import sys, os
import re
from tqdm import tqdm
import json
from collections import defaultdict, Counter
import itertools
from concurrent.futures import ThreadPoolExecutor

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from shared_functions.global_functions import *
from shared_functions.global_functions import query_neo4j, driver, DATABASE, dml_ddl_neo4j

data_dir = os.path.join(project_root, 'data')

def process_node_batch(batch, all_keys):
    """Normalize a batch of node records."""
    rows = []
    for record in batch:
        row = {
            "id": record["id"],
            "labels": ":".join(record["labels"])
        }
        props = record["props"]
        for key in all_keys:
            row[key] = props.get(key, None)
        rows.append(row)
    return rows

def process_edge_batch(batch, all_edge_keys):
    """Normalize a batch of edge records."""
    edge_rows = []
    for record in batch:
        row = {
            "source": record["source"],
            "target": record["target"],
            "type": record["type"]
        }
        props = record["props"]
        for key in all_edge_keys:
            row[key] = props.get(key, None)
        edge_rows.append(row)
    return edge_rows

def snapshot_node(namespace='Test'):
    '''
    Take a snapshot of all nodes properties in the specific namespace for backup
    Using concurrent streaming batches to avoid memory explosion.

    Input: 
        namespace: label of nodes, working as namespace of the Graph to separate between different universes
    Output:
        Save to nodes.csv with node id, node labels and all attributes
    '''

    # Validate label
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", namespace):
        raise ValueError("Invalid label name")

    # Total count for progress bar
    print(f"Querying node count for '{namespace}'...")
    count_query = f"MATCH (n:{namespace}) RETURN count(n) AS total"
    total = query_neo4j(count_query)[0]["total"]
    
    if total == 0:
        print(f"No nodes found for namespace {namespace}")
        return

    # Efficiently collect all possible property keys using Cypher
    print("Identifying property keys (scanning distinct key sets)...")
    keys_query = f"MATCH (n:{namespace}) RETURN DISTINCT keys(n) AS key_set"
    res = query_neo4j(keys_query)
    all_keys = set()
    for row in res:
        all_keys.update(row["key_set"])
    all_keys = sorted(list(all_keys))
    columns = ["id", "labels"] + all_keys

    # Fetch nodes in a stream
    query_nodes = f"""
    MATCH (n:{namespace})
    RETURN n.id AS id, labels(n) AS labels, properties(n) AS props
    """
    
    output_path = os.path.join(data_dir, "nodes.csv")
    batch_size = 10000
    num_workers = 4

    # Initialize CSV with header
    pd.DataFrame(columns=columns).to_csv(output_path, index=False)

    with driver.session(database=DATABASE) as session:
        result = session.run(query_nodes)
        
        with tqdm(total=total, desc=f"Snapshotting {namespace} Nodes") as pbar:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                while True:
                    # Pull batch from Neo4j
                    batch_data = []
                    for _ in range(batch_size):
                        try:
                            record = next(result)
                            batch_data.append(record.data())
                            pbar.update(1)
                        except StopIteration:
                            break
                    
                    if not batch_data:
                        break
                    
                    # Process and write
                    futures.append(executor.submit(process_node_batch, batch_data, all_keys))
                    
                    # Flush oldest completed future to disk
                    if len(futures) >= num_workers * 2:
                        res_rows = futures.pop(0).result()
                        pd.DataFrame(res_rows, columns=columns).to_csv(output_path, mode='a', header=False, index=False)
                
                # Write remaining batches
                for f in futures:
                    res_rows = f.result()
                    pd.DataFrame(res_rows, columns=columns).to_csv(output_path, mode='a', header=False, index=False)

    print(f"Nodes CSV saved: {output_path}")

def snapshot_edge(namespace='Test'):
    '''
    Take a snapshot of all edges in the specific namespace for backup
    Using concurrent streaming batches to avoid memory explosion.

    Input: 
        namespace: label of nodes, working as namespace of the Graph to separate between different universes
    Output:
        Save to edges.csv with head/tail id and edge label
    '''

    # Validate label
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", namespace):
        raise ValueError("Invalid label name")

    # Total count for progress bar
    print(f"Querying edge count for '{namespace}'...")
    count_query = f"MATCH (a:{namespace})-[r]->(b:{namespace}) RETURN count(r) AS total"
    total = query_neo4j(count_query)[0]["total"]

    if total == 0:
        print(f"No edges found for namespace {namespace}")
        return

    # Efficiently collect all relationship property keys using Cypher
    print("Identifying edge property keys (scanning distinct key sets)...")
    keys_query = f"MATCH (a:{namespace})-[r]->(b:{namespace}) RETURN DISTINCT keys(r) AS key_set"
    res = query_neo4j(keys_query)
    all_edge_keys = set()
    for row in res:
        all_edge_keys.update(row["key_set"])
    all_edge_keys = sorted(list(all_edge_keys))
    columns = ["source", "target", "type"] + all_edge_keys

    # Fetch edges
    query_edges = f"""
    MATCH (a:{namespace})-[r]->(b:{namespace})
    RETURN 
        coalesce(a.id, elementId(a)) AS source,
        coalesce(b.id, elementId(b)) AS target,
        type(r) AS type,
        properties(r) AS props
    """
    
    output_path = os.path.join(data_dir, "edges.csv")
    batch_size = 10000
    num_workers = 4

    # Initialize CSV with header
    pd.DataFrame(columns=columns).to_csv(output_path, index=False)

    with driver.session(database=DATABASE) as session:
        result = session.run(query_edges)
        
        with tqdm(total=total, desc=f"Snapshotting {namespace} Edges") as pbar:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                while True:
                    # Pull batch from Neo4j
                    batch_data = []
                    for _ in range(batch_size):
                        try:
                            record = next(result)
                            batch_data.append(record.data())
                            pbar.update(1)
                        except StopIteration:
                            break
                    
                    if not batch_data:
                        break
                    
                    # Process and write
                    futures.append(executor.submit(process_edge_batch, batch_data, all_edge_keys))
                    
                    # Flush oldest completed future to disk
                    if len(futures) >= num_workers * 2:
                        res_rows = futures.pop(0).result()
                        pd.DataFrame(res_rows, columns=columns).to_csv(output_path, mode='a', header=False, index=False)
                
                # Write remaining batches
                for f in futures:
                    res_rows = f.result()
                    pd.DataFrame(res_rows, columns=columns).to_csv(output_path, mode='a', header=False, index=False)

    print(f"Edges CSV saved: {output_path}")

def graph_recreation(namespace = 'Test', subset = False):
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

    nodes_csv = os.path.join(data_dir, 'nodes.csv')
    edges_csv = os.path.join(data_dir, 'edges.csv')

    selected_ids = None
    if subset:
        print("Subset mode enabled. Selecting 200k nodes balanced by label...")
        
        # Identify connected nodes from edges.csv
        connected_ids = set()
        if os.path.exists(edges_csv):
            est_total_edges = sum(1 for _ in open(edges_csv, encoding='utf-8')) - 1
            for chunk in tqdm(pd.read_csv(edges_csv, usecols=['source', 'target'], dtype=str, chunksize=100000),
                              total=(est_total_edges // 100000) + 1, desc="Identifying connected nodes"):
                connected_ids.update(chunk['source'])
                connected_ids.update(chunk['target'])
        
        # Group all nodes by label and connectivity status
        label_map = defaultdict(lambda: {'connected': [], 'standalone': []})
        est_total_nodes = sum(1 for _ in open(nodes_csv, encoding='utf-8')) - 1
        for chunk in tqdm(pd.read_csv(nodes_csv, usecols=['id', 'labels'], dtype={'id': str}, chunksize=100000),
                          total=(est_total_nodes // 100000) + 1, desc="Analyzing nodes for subset"):
            for r in chunk.itertuples(index=False):
                node_id = str(r.id)
                labels_str = str(r.labels)
                labels_list = [l.strip() for l in labels_str.split(':') if l.strip() and l != namespace]
                primary_label = labels_list[0] if labels_list else 'Unspecified'
                
                if node_id in connected_ids:
                    label_map[primary_label]['connected'].append(node_id)
                else:
                    label_map[primary_label]['standalone'].append(node_id)
        
        # Selection algorithm (Target 200k)
        target_total = 200000
        labels = list(label_map.keys())
        if labels:
            base_quota = target_total // len(labels)
            temp_selected = set()
            conn_taken_per_label = Counter()
            stand_taken_per_label = Counter()
            
            for label in labels:
                nodes = label_map[label]['connected']
                to_take = min(len(nodes), base_quota)
                temp_selected.update(nodes[:to_take])
                conn_taken_per_label[label] = to_take

            # Fill remaining quota with standalone nodes
            for label in labels:
                needed = base_quota - conn_taken_per_label[label]
                if needed > 0:
                    nodes = label_map[label]['standalone']
                    to_take = min(len(nodes), needed)
                    temp_selected.update(nodes[:to_take])
                    stand_taken_per_label[label] = to_take

            # If still under 200k, take remaining nodes (priority: connected)
            if len(temp_selected) < target_total:
                remaining_connected = []
                for label in labels:
                    remaining_connected.extend(label_map[label]['connected'][conn_taken_per_label[label]:])
                to_take = min(len(remaining_connected), target_total - len(temp_selected))
                temp_selected.update(remaining_connected[:to_take])
                # Note: We don't strictly need to track additional connected taken here 
                # because we are just filling to the target.

            # Last resort fill with any remaining standalone
            if len(temp_selected) < target_total:
                remaining_standalone = []
                for label in labels:
                    remaining_standalone.extend(label_map[label]['standalone'][stand_taken_per_label[label]:])
                
                to_take = min(len(remaining_standalone), target_total - len(temp_selected))
                temp_selected.update(remaining_standalone[:to_take])
            
            selected_ids = temp_selected
            print(f"Subset selection complete: {len(selected_ids)} nodes selected.")
            # Clear label_map to free memory
            label_map.clear()
        else:
            print("No labels found, defaulting to full graph.")
            subset = False

    print("Loading nodes from CSV (chunked)...")
    # Performance: chunksize prevents memory explosion for large CSVs
    node_groups = defaultdict(list)
    
    # Estimate total nodes for progress bar (faster than counting lines)
    est_total_nodes = total_nodes = sum(1 for _ in open(nodes_csv, encoding='utf-8')) - 1
    
    for chunk in tqdm(pd.read_csv(nodes_csv, low_memory=False, dtype={'id': str}, chunksize=50000), 
                      total=(est_total_nodes // 50000) + 1, desc="Grouping nodes"):
        for r in chunk.itertuples(index=False):
            if subset and str(r.id) not in selected_ids:
                continue
            r_dict = r._asdict()
            labels_str = r_dict.pop("labels", "")
            # Filter out NaN/null properties
            clean_props = {k: v for k, v in r_dict.items() if pd.notna(v) and k != 'id'}
            
            # Parse labels
            labels_list = [str(l).strip() for l in str(labels_str).split(":") if str(l).strip()]
            labels_list = list(dict.fromkeys(labels_list))

            # Reconstruct labels in order: [Entity, Namespace, Database]
            entity_types = ["Drug", "Disease"]
            found_entity_type = next((l for l in labels_list if l in entity_types), "Entity")
            db_labels = [l for l in labels_list if l != namespace and l not in entity_types]
            found_db = db_labels[0] if db_labels else "Source"
            
            labels_key = f"{found_entity_type}:{namespace}:{found_db}"
            node_groups[labels_key].append({"id": r_dict['id'], "props": clean_props})

    print("Checking database for node status (exact label matching)...")
    
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

    # Granular Edge Resume
    print("Loading edges from CSV (chunked)...")
    edge_groups = defaultdict(list)
    
    # Estimate total edges
    est_total_edges = sum(1 for _ in open(edges_csv, encoding='utf-8')) - 1
    
    for chunk in tqdm(pd.read_csv(edges_csv, low_memory=False, dtype={'source': str, 'target': str}, chunksize=50000),
                      total=(est_total_edges // 50000) + 1, desc="Grouping edges"):
        for r in chunk.itertuples(index=False):
            if subset and (str(r.source) not in selected_ids or str(r.target) not in selected_ids):
                continue
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

    # snapshot_node()
    # snapshot_edge()
    clear_database()
    graph_recreation(subset=True)