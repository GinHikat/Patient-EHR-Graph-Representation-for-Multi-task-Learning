from core.database import get_db_driver
from core.config import settings

def get_edge_types_data(namespace: str = "Test"):
    driver = get_db_driver()
    query = f"MATCH (:`{namespace}`)-[r]-(:`{namespace}`) RETURN DISTINCT type(r) as type"
    with driver.session() as session:
        result = session.run(query)
        return [record["type"] for record in result]

def get_graph_data(limit: int, node_types: list[str] = None, edge_types: list[str] = None, namespace: str = "Test"):
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
    # If we are filtering by edge type specifically, we focus more on edges.
    if edge_types and not node_types:
        num_node_slots = 0 # Don't just sample random nodes if we want specific edges
        num_edge_slots = len(target_edge_types)
    else:
        num_node_slots = len(target_labels)
        num_edge_slots = len(target_edge_types)
        
    total_slots = num_node_slots + num_edge_slots
    
    quota = int(limit / (num_node_slots + 1.2 * num_edge_slots)) if total_slots > 0 else limit
    quota = max(1, quota)

    nodes = {}
    links = []

    with driver.session() as session:
        # Step A: Sample nodes balanced by label (only if not strictly filtering by edges)
        if num_node_slots > 0:
            for label in target_labels:
                query = f"""
                MATCH (n:`{label}`)
                WHERE n:`{namespace}`
                RETURN n
                LIMIT {quota}
                """
                result = session.run(query)
                for record in result:
                    n = record["n"]
                    if n:
                        eid = n.element_id if hasattr(n, "element_id") else str(getattr(n, "id", n))
                        if eid not in nodes:
                            nodes[eid] = {"id": eid, "labels": list(n.labels), "properties": dict(n)}

        # Step B: Sample edges balanced by type
        for etype in target_edge_types:
            query = f"""
            MATCH (n:`{namespace}`)-[r:`{etype}`]-(m:`{namespace}`)
            RETURN n, r, m
            LIMIT {quota * 2 if edge_types else quota}
            """
            result = session.run(query)
            for record in result:
                # Add nodes
                for key in ["n", "m"]:
                    node = record[key]
                    if node:
                        eid = node.element_id if hasattr(node, "element_id") else str(getattr(node, "id", node))
                        if eid not in nodes:
                            nodes[eid] = {"id": eid, "labels": list(node.labels), "properties": dict(node)}
                
                # Add relationship
                r = record["r"]
                if r:
                    start_node = r.start_node if hasattr(r, "start_node") else r.nodes[0]
                    end_node = r.end_node if hasattr(r, "end_node") else r.nodes[1]
                    s_id = start_node.element_id if hasattr(start_node, "element_id") else str(getattr(start_node, "id", start_node))
                    e_id = end_node.element_id if hasattr(end_node, "element_id") else str(getattr(end_node, "id", end_node))
                    link = {"source": s_id, "target": e_id, "type": r.type}
                    if link not in links:
                        links.append(link)

        # Step C: Enrichment - fetch all edges between the nodes we've already collected
        if nodes:
            node_ids = list(nodes.keys())
            query = f"""
            MATCH (n:`{namespace}`)-[r]-(m:`{namespace}`)
            WHERE elementId(n) IN $ids AND elementId(m) IN $ids
            {"AND type(r) IN $etypes" if edge_types else ""}
            RETURN r
            """
            result = session.run(query, ids=node_ids, etypes=edge_types)
            for record in result:
                r = record["r"]
                start_node = r.start_node if hasattr(r, "start_node") else r.nodes[0]
                end_node = r.end_node if hasattr(r, "end_node") else r.nodes[1]
                s_id = start_node.element_id if hasattr(start_node, "element_id") else str(getattr(start_node, "id", start_node))
                e_id = end_node.element_id if hasattr(end_node, "element_id") else str(getattr(end_node, "id", end_node))
                link = {"source": s_id, "target": e_id, "type": r.type}
                if link not in links:
                    links.append(link)

    # Step D: Final trimming to respect the limit if we went over
    if len(nodes) > limit:
        # Keep a balanced set by taking first few of each if we wanted to be fancy,
        # but simple slicing is usually fine once the collection was balanced.
        node_list = list(nodes.values())[:limit]
        final_ids = {n["id"] for n in node_list}
        final_links = [l for l in links if l["source"] in final_ids and l["target"] in final_ids]
        return {"nodes": node_list, "links": final_links}

    return {"nodes": list(nodes.values()), "links": links}

def get_node_by_id(node_id: str, namespace: str = None):
    driver = get_db_driver()
    # If no specific namespace is provided, search across all nodes.
    # Otherwise, filter by both the namespace label and searching criteria.
    label_part = f"n:`{namespace}`" if namespace else "TRUE"
    
    query = f"""
    MATCH (n)
    WHERE {label_part}
    AND (
       elementId(n) = $node_id 
       OR n.id = $node_id 
       OR toLower(n.name) CONTAINS toLower($node_id) 
       OR toLower(n.title) CONTAINS toLower($node_id)
       OR toLower(n.Title) CONTAINS toLower($node_id)
       OR toLower(n.Title) CONTAINS toLower($node_id)
       OR toLower(n.name) CONTAINS toLower($node_id)
    )
    WITH n LIMIT 10
    OPTIONAL MATCH (n)-[r]-(m)
    RETURN n, r, m
    LIMIT 300
    """
    
    with driver.session() as session:
        result = session.run(query, node_id=node_id)
        nodes = {}
        links = []
        
        for record in result:
            n = record["n"]
            if n:
                elem_id = n.element_id if hasattr(n, "element_id") else str(getattr(n, "id", str(n)))
                nodes[elem_id] = {"id": elem_id, "labels": list(n.labels), "properties": dict(n)}
            
            m = record.get("m")
            if m:
                m_elem_id = m.element_id if hasattr(m, "element_id") else str(getattr(m, "id", str(m)))
                if m_elem_id not in nodes:
                    nodes[m_elem_id] = {"id": m_elem_id, "labels": list(m.labels), "properties": dict(m)}
                
            try:
                r = record.get("r")
                if r is not None:
                    start_node = r.start_node if hasattr(r, "start_node") else r.nodes[0]
                    end_node = r.end_node if hasattr(r, "end_node") else r.nodes[1]
                    start_id = start_node.element_id if hasattr(start_node, "element_id") else str(getattr(start_node, "id", str(start_node)))
                    end_id = end_node.element_id if hasattr(end_node, "element_id") else str(getattr(end_node, "id", str(end_node)))
                    links.append({"source": start_id, "target": end_id, "type": r.type})
            except Exception:
                pass

        if not nodes:
            return None
        return {"nodes": list(nodes.values()), "links": links}

def get_node_types_data():
    driver = get_db_driver()
    # Return labels that have at least one node
    query = """
    MATCH (n) 
    WITH DISTINCT labels(n) AS labels 
    UNWIND labels AS label 
    RETURN DISTINCT label
    """
    with driver.session() as session:
        result = session.run(query)
        labels = [record["label"] for record in result if record["label"] != "Test"]
        return labels

def get_database_stats_data():
    driver = get_db_driver()
    # Breakdown counts by label and relationship type
    query = """
    CALL () {
      MATCH (n:Test)
      WITH count(n) as totalNodes
      MATCH (n:Test)
      UNWIND [l IN labels(n) WHERE l <> 'Test'] as label
      WITH totalNodes, label, count(n) as labelCount
      RETURN totalNodes, collect({type: label, count: labelCount}) as nodeBreakdown
    }
    CALL () {
      MATCH (:Test)-[r]->(:Test)
      WITH count(r) as totalEdges
      MATCH (:Test)-[r]->(:Test)
      WITH totalEdges, type(r) as relType, count(r) as relCount
      RETURN totalEdges, collect({type: relType, count: relCount}) as edgeBreakdown
    }
    RETURN totalNodes, nodeBreakdown[..15] as nodeBreakdown, totalEdges, edgeBreakdown[..15] as edgeBreakdown
    """
    with driver.session() as session:
        result = session.run(query)
        record = result.single()
        if record:
            return {
                "total_nodes": record["totalNodes"],
                "node_breakdown": record["nodeBreakdown"],
                "total_edges": record["totalEdges"],
                "edge_breakdown": record["edgeBreakdown"]
            }
        return {"total_nodes": 0, "node_breakdown": [], "total_edges": 0, "edge_breakdown": []}
