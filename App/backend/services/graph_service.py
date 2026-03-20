from core.database import get_db_driver
from core.config import settings

def get_graph_data(limit: int, node_types: list[str] = None, namespace: str = "Test"):
    driver = get_db_driver()
    
    # Filter by multiple selected types
    if node_types and "All" not in node_types:
        # Use simple label matching for multiple labels
        # Neo4j: MATCH (n:Test) WHERE any(l IN labels(n) WHERE l IN ['Disease', 'Drug'])
        # Or better: MATCH (n:Type1:Test) OR (n:Type2:Test)
        # Using WHERE n:Label1 OR n:Label2 is more efficient
        label_filters = " OR ".join([f"n:`{t}`" for t in node_types])
        query = f"""
        MATCH (n:`{namespace}`)
        WHERE {label_filters}
        OPTIONAL MATCH (n)-[r]-(m:`{namespace}`)
        RETURN n, r, m
        LIMIT $limit
        """
    else:
        query = f"""
        MATCH (n:`{namespace}`)
        OPTIONAL MATCH (n)-[r]-(m:`{namespace}`)
        RETURN n, r, m
        LIMIT $limit
        """
        
    with driver.session() as session:
        result = session.run(query, limit=limit)
        nodes = {}
        links = []
        
        for record in result:
            n = record["n"]
            if n:
                elem_id = n.element_id if hasattr(n, "element_id") else str(getattr(n, "id", str(n)))
                if elem_id not in nodes:
                    nodes[elem_id] = {"id": elem_id, "labels": list(n.labels), "properties": dict(n)}
            
            m = record.get("m")
            if m:
                m_elem_id = m.element_id if hasattr(m, "element_id") else str(getattr(m, "id", str(m)))
                if m_elem_id not in nodes:
                    nodes[m_elem_id] = {"id": m_elem_id, "labels": list(m.labels), "properties": dict(m)}
                
            try:
                r = record.get("r")
                if r is not None:
                    # Using nodes[0] and nodes[1] to represent start and end
                    start_node = r.start_node if hasattr(r, "start_node") else r.nodes[0]
                    end_node = r.end_node if hasattr(r, "end_node") else r.nodes[1]
                    
                    start_id = start_node.element_id if hasattr(start_node, "element_id") else str(getattr(start_node, "id", str(start_node)))
                    end_id = end_node.element_id if hasattr(end_node, "element_id") else str(getattr(end_node, "id", str(end_node)))
                    
                    link = {"source": start_id, "target": end_id, "type": r.type}
                    if link not in links:
                        links.append(link)
            except Exception as e:
                import traceback
                print("Error on link processing:")
                traceback.print_exc()

        return {"nodes": list(nodes.values()), "links": links}

def get_node_by_id(node_id: str, namespace: str = "Test"):
    driver = get_db_driver()
    # Find the target nodes first, then fetch all direct neighbors
    query = f"""
    MATCH (n:`{namespace}`)
    WHERE elementId(n) = $node_id 
       OR n.id = $node_id 
       OR toLower(n.name) = toLower($node_id) 
       OR toLower(n.title) = toLower($node_id)
       OR toLower(n.Title) = toLower($node_id)
    WITH n LIMIT 5
    OPTIONAL MATCH (n)-[r]-(m)
    RETURN n, r, m
    LIMIT 200
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
    # Using faster subqueries for counts
    query = """
    CALL {
      MATCH (n:Test) RETURN count(n) as nodeCount
    }
    CALL {
      MATCH (:Test)-[r]->(:Test) RETURN count(r) as edgeCount
    }
    RETURN nodeCount, edgeCount
    """
    with driver.session() as session:
        result = session.run(query)
        record = result.single()
        if record:
            return record["nodeCount"], record["edgeCount"]
        return 0, 0
