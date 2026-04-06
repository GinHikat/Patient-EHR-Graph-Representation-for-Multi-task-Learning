from typing import Optional, List
from core.database import get_db_driver
from core.config import settings
import neo4j.time

def serialize_properties(props: dict) -> dict:
    """Helper to convert Neo4j types to JSON-serializable formats."""
    serialized = {}
    for k, v in props.items():
        if isinstance(v, (neo4j.time.DateTime, neo4j.time.Date)):
            serialized[k] = v.isoformat()
        else:
            serialized[k] = v
    return serialized

def get_edge_types_data(namespace: str = "Test"):
    driver = get_db_driver()
    query = f"MATCH (:`{namespace}`)-[r]-(:`{namespace}`) RETURN DISTINCT type(r) as type"
    with driver.session() as session:
        result = session.run(query)
        return [record["type"] for record in result]

def get_graph_data(limit: int, node_types: Optional[List[str]] = None, edge_types: Optional[List[str]] = None, namespace: str = "Test"):
    driver = get_db_driver()
    
    # 1. Determine labels to balance
    available_labels = get_node_types_data()
    if node_types and "All" not in node_types:
        target_labels = [l for l in available_labels if l in node_types]
    else:
        target_labels = available_labels
        
    # 2. Determine edge types to balance
    all_edge_types = get_edge_types_data(namespace)
    if edge_types:
        target_edge_types = [t for t in all_edge_types if t in edge_types]
    else:
        target_edge_types = all_edge_types
    
    if not target_labels and not target_edge_types:
        return {"nodes": [], "links": []}

    # 3. Calculate quotas for each category to ensure even distribution
    num_node_slots = len(target_labels)
    num_edge_slots = len(target_edge_types)
    total_slots = num_node_slots + num_edge_slots
    
    quota = max(1, int(limit / (num_node_slots + 1.2 * num_edge_slots))) if total_slots > 0 else limit

    nodes = {}
    links = []

    is_filtered = bool(node_types and "All" not in node_types)

    with driver.session() as session:
        # Step A & B: Combined Sampling
        if num_node_slots > 0:
            for label in target_labels:
                lbl_clause = f":`{label}`" if label != "All" else ""
                ns_clause = f":`{namespace}`" if namespace and namespace != "All" else ""
                query = f"MATCH (n{lbl_clause}{ns_clause}) RETURN n LIMIT $quota"
                result = session.run(query, quota=quota)
                for record in result:
                    n = record["n"]
                    if n:
                        eid = n.element_id if hasattr(n, "element_id") else str(getattr(n, "id", n))
                        if eid not in nodes:
                            nodes[eid] = {"id": eid, "labels": list(n.labels), "properties": serialize_properties(dict(n))}

        for etype in target_edge_types:
            where_clause = f"WHERE n:`{namespace}` AND m:`{namespace}`"
            if is_filtered:
                where_clause += " AND any(l IN labels(n) WHERE l IN $target_labels) AND any(l IN labels(m) WHERE l IN $target_labels)"
            
            query = f"""
            MATCH (n)-[r:`{etype}`]-(m)
            {where_clause}
            RETURN n, r, m
            LIMIT $limit
            """
            result = session.run(query, limit=quota * 2 if edge_types else quota, target_labels=target_labels)
            for record in result:
                for key in ["n", "m"]:
                    node = record[key]
                    if node:
                        if is_filtered:
                            node_labels = set(node.labels)
                            if not any(l in node_labels for l in target_labels):
                                continue

                        eid = node.element_id if hasattr(node, "element_id") else str(getattr(node, "id", node))
                        if eid not in nodes:
                            nodes[eid] = {"id": eid, "labels": list(node.labels), "properties": serialize_properties(dict(node))}
                
                r = record["r"]
                if r:
                    start_node = r.start_node if hasattr(r, "start_node") else r.nodes[0]
                    end_node = r.end_node if hasattr(r, "end_node") else r.nodes[1]
                    s_id = start_node.element_id if hasattr(start_node, "element_id") else str(getattr(start_node, "id", start_node))
                    e_id = end_node.element_id if hasattr(end_node, "element_id") else str(getattr(end_node, "id", end_node))
                    
                    if s_id in nodes and e_id in nodes:
                        link = {"source": s_id, "target": e_id, "type": r.type, "properties": serialize_properties(dict(r))}
                        if link not in links:
                            links.append(link)

        # Step C: Enrichment
        if nodes:
            node_ids = list(nodes.keys())
            query = f"""
            MATCH (n:`{namespace}`)-[r]-(m:`{namespace}`)
            WHERE elementId(n) IN $ids AND elementId(m) IN $ids
            {"AND type(r) IN $etypes" if edge_types else ""}
            RETURN r
            LIMIT 5000 
            """
            result = session.run(query, ids=node_ids, etypes=edge_types)
            for record in result:
                r = record["r"]
                start_node = r.start_node if hasattr(r, "start_node") else r.nodes[0]
                end_node = r.end_node if hasattr(r, "end_node") else r.nodes[1]
                s_id = start_node.element_id if hasattr(start_node, "element_id") else str(getattr(start_node, "id", start_node))
                e_id = end_node.element_id if hasattr(end_node, "element_id") else str(getattr(end_node, "id", end_node))
                link = {"source": s_id, "target": e_id, "type": r.type, "properties": serialize_properties(dict(r))}
                if link not in links:
                    links.append(link)

    if len(nodes) > limit:
        node_ids_list = list(nodes.keys())[:limit]
        final_nodes = [nodes[eid] for eid in node_ids_list]
        final_ids = set(node_ids_list)
        final_links = [l for l in links if l["source"] in final_ids and l["target"] in final_ids]
        return {"nodes": final_nodes, "links": final_links}

    return {"nodes": list(nodes.values()), "links": links}

def get_node_by_id(node_id: str, namespace: Optional[str] = None, node_types: Optional[List[str]] = None):
    driver = get_db_driver()
    
    import re
    letters = len(re.findall(r'[a-zA-Z]', node_id))
    digits = len(re.findall(r'[0-9]', node_id))
    op = "CONTAINS" if letters > digits else "STARTS WITH"
    
    is_filtered = bool(node_types and "All" not in node_types and len(node_types) > 0)
    target_labels = node_types if node_types else []

    # REFRACTORED SEARCH: TIERED APPROACH TO HIT INDICES
    ns_clause = f":`{namespace}`" if namespace else ""
    query = f"""
    CALL {{
        // Tier 1: Try exact ID Match (Most efficient, hits unique constraint/index)
        MATCH (n{ns_clause})
        WHERE n.id = $node_id OR elementId(n) = $node_id
        RETURN n LIMIT 1
        
        UNION
        
        // Tier 2: Try common name properties (hits B-Tree index if exists)
        MATCH (n{ns_clause})
        WHERE n.name = $node_id OR n.title = $node_id OR n.Title = $node_id
        RETURN n LIMIT 10
        
        UNION
        
        // Tier 3: Partial Search (Fuzzy fallback, slow full-scan)
        MATCH (n{ns_clause})
        WHERE toLower(toString(n.id)) {op} toLower($node_id)
           OR toLower(toString(n.name)) CONTAINS toLower($node_id) 
           OR toLower(toString(n.title)) CONTAINS toLower($node_id)
        RETURN n LIMIT 20
    }}

    // Post-Search Enrichment: Expand context
    WITH n LIMIT 20
    OPTIONAL MATCH (n)-[r]-(m)
    WHERE NOT $is_filtered OR any(l IN labels(m) WHERE l IN $target_labels)
    RETURN n, r, m
    LIMIT 300
    """
    
    with driver.session() as session:
        result = session.run(query, node_id=node_id, target_labels=target_labels, is_filtered=is_filtered)
        nodes = {}
        links = []
        
        for record in result:
            n = record["n"]
            if n:
                elem_id = n.element_id if hasattr(n, "element_id") else str(getattr(n, "id", str(n)))
                if elem_id not in nodes:
                    nodes[elem_id] = {"id": elem_id, "labels": list(n.labels), "properties": serialize_properties(dict(n))}
            
            m = record.get("m")
            if m:
                m_elem_id = m.element_id if hasattr(m, "element_id") else str(getattr(m, "id", str(m)))
                if m_elem_id not in nodes:
                    nodes[m_elem_id] = {"id": m_elem_id, "labels": list(m.labels), "properties": serialize_properties(dict(m))}
                
            try:
                r = record.get("r")
                if r is not None:
                    start_node = r.start_node if hasattr(r, "start_node") else r.nodes[0]
                    end_node = r.end_node if hasattr(r, "end_node") else r.nodes[1]
                    start_id = start_node.element_id if hasattr(start_node, "element_id") else str(getattr(start_node, "id", str(start_node)))
                    end_id = end_node.element_id if hasattr(end_node, "element_id") else str(getattr(end_node, "id", str(end_node)))
                    links.append({"source": start_id, "target": end_id, "type": r.type, "properties": serialize_properties(dict(r))})
            except Exception:
                pass

        if not nodes:
            return None
        return {"nodes": list(nodes.values()), "links": links}

def get_node_types_data():
    driver = get_db_driver()
    # Optimized: If possible, use call db.labels()
    query = "CALL db.labels() YIELD label RETURN label"
    with driver.session() as session:
        result = session.run(query)
        labels = [record["label"] for record in result if record["label"] not in ("Test", "Entity", "Node")]
        return labels

def get_database_stats_data():
    driver = get_db_driver()
    with driver.session() as session:
        # FASTEST: Use CALL apoc.meta.stats() if available
        try:
            apoc_res = session.run("CALL apoc.meta.stats() YIELD nodeCount, relCount, labels, relTypesCount")
            data = apoc_res.single()
            if data and data["nodeCount"] > 0:
                return {
                    "total_nodes": data["nodeCount"],
                    "node_breakdown": [{"labels": [k], "count": v} for k, v in data["labels"].items()],
                    "total_edges": data["relCount"],
                    "edge_breakdown": [{"type": k, "count": v} for k, v in data["relTypesCount"].items()]
                }
        except Exception:
            pass

        # Fallback: Using faster metadata/label-based counting
        try:
            # 1. Get detailed label combinations for hierarchy
            hierarchy_query = """
            MATCH (n) 
            WITH labels(n) as label_list
            RETURN [l IN label_list WHERE NOT l IN ['Entity', 'Node']] as labels, count(*) as count
            ORDER BY count DESC
            LIMIT 50
            """
            hierarchy_res = session.run(hierarchy_query)
            node_breakdown = []
            total_nodes = 0
            for record in hierarchy_res:
                lbls = record["labels"]
                if not lbls: continue
                count = record["count"]
                node_breakdown.append({"labels": lbls, "count": count})
                # total_nodes is calculated separately to avoid double counting across sets if they were overlaps, 
                # but labels(n) gives the full set, so this summation is actually correct for total.
                total_nodes += count

            # Get absolute total from Test label (representative count store)
            total_nodes_final = session.run("MATCH (n:Test) RETURN count(n) as total").single()["total"]
            if total_nodes_final == 0: # Fallback to global count if :Test is missing
                total_nodes_final = session.run("MATCH (n) RETURN count(n) as total").single()["total"]

            # 2. Get relationship types and counts
            rel_types_res = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
            edge_breakdown = []
            total_edges = 0
            for record in rel_types_res:
                rtype = record["relationshipType"]
                count_res = session.run(f"MATCH ()-[r:`{rtype}`]->() RETURN count(r) as count").single()
                count = count_res["count"] if count_res else 0
                edge_breakdown.append({"type": rtype, "count": count})
                total_edges += count

            return {
                "total_nodes": total_nodes_final,
                "node_breakdown": node_breakdown,
                "total_edges": total_edges,
                "edge_breakdown": edge_breakdown
            }
        except Exception as e:
            print(f"Fallback stats failed: {e}")
            return {
                "total_nodes": 0,
                "node_breakdown": [],
                "total_edges": 0,
                "edge_breakdown": []
            }
