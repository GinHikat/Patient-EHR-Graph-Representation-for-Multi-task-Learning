from core.database import get_db_driver

def get_graph_data(limit: int, node_type: str = None, namespace: str = "Test"):
    driver = get_db_driver()
    if node_type and node_type != "All":
        query = f"""
        MATCH (n:`{namespace}`:`{node_type}`)
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

def get_node_types_data():
    driver = get_db_driver()
    query = "CALL db.labels()"
    with driver.session() as session:
        result = session.run(query)
        labels = [record["label"] for record in result]
        return labels
