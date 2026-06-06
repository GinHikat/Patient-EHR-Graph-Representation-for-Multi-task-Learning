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
    query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType as type"
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
        
    num_node_slots = len(target_labels)
    num_edge_slots = len(target_edge_types)
    total_slots = num_node_slots + num_edge_slots
    quota = max(1, int(limit / (num_node_slots + 1.2 * num_edge_slots))) if total_slots > 0 else limit

    nodes = {}
    links = []
    
    with driver.session() as session:
        # A. UNIFIED BATCHED NODE SAMPLING (Single Round-Trip)
        if num_node_slots > 0:
            subqueries = []
            for label in target_labels:
                lbl_clause = f":`{label}`" if label != "All" else ""
                # Fast single-label index scan
                subqueries.append(f"MATCH (n{lbl_clause}) RETURN n LIMIT {quota}")
            
            unified_query = "\nUNION\n".join(subqueries)
            if unified_query:
                result = session.run(unified_query)
                for record in result:
                    n = record["n"]
                    if n:
                        eid = n.element_id if hasattr(n, "element_id") else str(getattr(n, "id", n))
                        if eid not in nodes:
                            nodes[eid] = {"id": eid, "labels": list(n.labels), "properties": serialize_properties(dict(n))}

        # B. UNIFIED BATCHED EDGE SAMPLING (Single Round-Trip)
        if num_edge_slots > 0:
            subqueries = []
            is_filtered = bool(node_types and "All" not in node_types)
            target_labels_set = set(target_labels)
            
            for etype in target_edge_types:
                # Directed fast index matching without slow double label checks
                subqueries.append(f"MATCH (n)-[r:`{etype}`]->(m) RETURN n, r, m LIMIT {quota * 2 if edge_types else quota}")
            
            unified_query = "\nUNION\n".join(subqueries)
            if unified_query:
                result = session.run(unified_query)
                for record in result:
                    n_node = record["n"]
                    m_node = record["m"]
                    r_rel = record["r"]
                    
                    # Store nodes
                    for node in (n_node, m_node):
                        if node:
                            if is_filtered and not any(l in node.labels for l in target_labels_set):
                                continue
                            eid = node.element_id if hasattr(node, "element_id") else str(getattr(node, "id", node))
                            if eid not in nodes:
                                nodes[eid] = {"id": eid, "labels": list(node.labels), "properties": serialize_properties(dict(node))}
                    
                    # Store relationship
                    if r_rel:
                        start_node = r_rel.start_node if hasattr(r_rel, "start_node") else r_rel.nodes[0]
                        end_node = r_rel.end_node if hasattr(r_rel, "end_node") else r_rel.nodes[1]
                        s_id = start_node.element_id if hasattr(start_node, "element_id") else str(getattr(start_node, "id", start_node))
                        e_id = end_node.element_id if hasattr(end_node, "element_id") else str(getattr(end_node, "id", end_node))
                        
                        if s_id in nodes and e_id in nodes:
                            link = {"source": s_id, "target": e_id, "type": r_rel.type, "properties": serialize_properties(dict(r_rel))}
                            if link not in links:
                                links.append(link)

        # C. ENRICHMENT SCAN (Interconnecting Sampled Entities)
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

    # Trim to strict client limits
    if len(nodes) > limit:
        node_ids_list = list(nodes.keys())[:limit]
        final_nodes = [nodes[eid] for eid in node_ids_list]
        final_ids = set(node_ids_list)
        final_links = [l for l in links if l["source"] in final_ids and l["target"] in final_ids]
        return {"nodes": final_nodes, "links": final_links}

    return {"nodes": list(nodes.values()), "links": links}

def get_node_by_id(node_id: str, namespace: Optional[str] = None, node_types: Optional[List[str]] = None):
    driver = get_db_driver()
    
    # Strip "HP:" prefix if present for HPO searches
    node_id_clean = node_id.strip()
    if node_id_clean.upper().startswith("HP:"):
        node_id_clean = node_id_clean[3:]
        
    import re
    letters = len(re.findall(r'[a-zA-Z]', node_id_clean))
    digits = len(re.findall(r'[0-9]', node_id_clean))
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
    WITH n LIMIT 20

    // Multi-hop expansion for Patients and Admissions, showing full clinical timeline
    CALL {{
        WITH n
        RETURN n AS source, null AS rel, null AS target
        
        UNION
        
        WITH n
        WITH n WHERE "Patient" IN labels(n)
        MATCH (n)-[r1:ADMISSION]->(a:Admission)
        RETURN n AS source, r1 AS rel, a AS target
        
        UNION
        
        WITH n
        WITH n WHERE "Patient" IN labels(n)
        MATCH (n)-[:ADMISSION]->(a:Admission)-[r2]->(m)
        WHERE NOT $is_filtered OR any(l IN labels(m) WHERE l IN $target_labels)
        RETURN a AS source, r2 AS rel, m AS target

        UNION
        
        WITH n
        WITH n WHERE "Patient" IN labels(n)
        MATCH (n)-[r1:HAS_OUTPATIENT_NOTE]->(o:Outpatient)
        RETURN n AS source, r1 AS rel, o AS target
        
        UNION
        
        WITH n
        WITH n WHERE "Patient" IN labels(n)
        MATCH (n)-[:HAS_OUTPATIENT_NOTE]->(o:Outpatient)-[r2]->(m)
        WHERE NOT $is_filtered OR any(l IN labels(m) WHERE l IN $target_labels)
        RETURN o AS source, r2 AS rel, m AS target

        UNION

        WITH n
        WITH n WHERE "Admission" IN labels(n)
        MATCH (p:Patient)-[r1:ADMISSION]->(n)
        RETURN p AS source, r1 AS rel, n AS target
        
        UNION
        
        WITH n
        WITH n WHERE "Admission" IN labels(n)
        MATCH (n)-[r2]->(m)
        WHERE NOT $is_filtered OR any(l IN labels(m) WHERE l IN $target_labels)
        RETURN n AS source, r2 AS rel, m AS target

        UNION

        // Generic fallback for any other node types (1-hop)
        WITH n
        WITH n WHERE NOT "Patient" IN labels(n) AND NOT "Admission" IN labels(n)
        MATCH (n)-[r]-(m)
        WHERE NOT $is_filtered OR any(l IN labels(m) WHERE l IN $target_labels)
        RETURN n AS source, r AS rel, m AS target
    }}
    RETURN source AS n, rel AS r, target AS m
    LIMIT 1000
    """
    
    with driver.session() as session:
        result = session.run(query, node_id=node_id_clean, target_labels=target_labels, is_filtered=is_filtered)
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
                # Fetch hierarchical node combinations instead of APOC's flattened counts to restore UI Hierarchy
                hierarchy_query = """
                MATCH (n) 
                WITH labels(n) as label_list
                RETURN [l IN label_list WHERE NOT l IN ['Entity', 'Node']] as labels, count(*) as count
                ORDER BY count DESC
                LIMIT 50
                """
                hierarchy_res = session.run(hierarchy_query)
                node_breakdown = []
                for record in hierarchy_res:
                    if record["labels"]:
                        node_breakdown.append({"labels": record["labels"], "count": record["count"]})

                return {
                    "total_nodes": data["nodeCount"],
                    "node_breakdown": node_breakdown,
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

def is_valid_id(val) -> bool:
    if not val:
        return False
    val_str = str(val).strip()
    if val_str in ("", "[]", "['[]']", "none", "null", "NaN", "undefined"):
        return False
    return True

def get_cui_subgraph_data(
    cui: str,
    namespace: str = "Test",
    rxnorm: Optional[str] = None,
    snomed: Optional[str] = None,
    mesh: Optional[str] = None,
    drugbank: Optional[str] = None,
    omim: Optional[str] = None,
    icd9: Optional[str] = None,
    icd10: Optional[str] = None,
    loinc: Optional[str] = None,
    pubchem: Optional[str] = None,
    pubmed: Optional[str] = None,
    hpo: Optional[str] = None
) -> dict:
    """
    Retrieves the subgraph for a given CUI or alternative database identifiers.
    Checks priorities one-by-one; once a matching central node is found,
    the search is skipped and only that node's 1-hop edges are retrieved.
    """
    driver = get_db_driver()
    
    # Priority list of search terms and their corresponding Neo4j property names
    search_priorities = [
        (cui, "cui"),
        (drugbank, "drugbank_id"),
        (mesh, "mesh_id"),
        (rxnorm, "rxnorm"),
        (rxnorm, "rxnorm_id"),
        (snomed, "snomed_id"),
        (omim, "omim_id"),
        (icd9, "icd9"),
        (icd10, "icd10"),
        (loinc, "loinc"),
        (pubchem, "pubchem_id"),
        (pubmed, "pubmed_id"),
        (hpo, "hpo_id")
    ]
    
    central_node_eid = None
    
    with driver.session() as session:
        # Step 1: Find the central node one-by-one by priority
        for val, field in search_priorities:
            if not is_valid_id(val):
                continue
                
            val_clean = str(val).strip()
            # Strip "HP:" prefix for HPO matching
            if field == "hpo_id" and val_clean.upper().startswith("HP:"):
                val_clean = val_clean[3:]
                
            vals_list = [v.strip() for v in val_clean.split(',')]
            
            # Handle potential numeric type matching
            params = {"vals_list": vals_list}
            vals_num_list = []
            for v in vals_list:
                try:
                    val_num = float(v)
                    if val_num.is_integer():
                        val_num = int(val_num)
                    vals_num_list.append(val_num)
                except ValueError:
                    pass
            
            if vals_num_list:
                params["vals_num_list"] = vals_num_list
                
            has_num = len(vals_num_list) > 0

            find_query = f"""
            MATCH (c:`{namespace}`)
            WHERE (c:Disease OR c:Drug OR c:Phenotype OR c:Diagnosis OR c:DiagnosisCategory OR c:Lab OR c:Procedure OR c:Patient OR c:Admission OR c:External OR c:CTD OR c:HPO OR c:DB)
              AND (
                c.id IN $vals_list OR any(x IN $vals_list WHERE c.id = [x])
                {"OR c.id IN $vals_num_list OR any(x IN $vals_num_list WHERE c.id = [x])" if has_num else ""}
              )
            RETURN elementId(c) as eid
            LIMIT 1
            """
            res = session.run(find_query, **params)
            record = res.single()
            if record:
                central_node_eid = record["eid"]
                break  # Skip the rest once 1 is found!
                
        # If no node is found, return empty
        if not central_node_eid:
            return {"nodes": [], "links": []}
            
        # Step 2: Fetch only that node and its connected edges/neighbors within 1 hop
        subgraph_query = f"""
        MATCH (c:`{namespace}`)
        WHERE elementId(c) = $eid
        OPTIONAL MATCH (c)-[r]-(m)
        WHERE NOT "Admission" IN labels(m)
        WITH c, r, m
        OPTIONAL MATCH (m)-[r2]-()
        WITH c, r, m, count(r2) AS degree
        ORDER BY degree DESC
        LIMIT 50
        RETURN c, r, m
        """
        result = session.run(subgraph_query, eid=central_node_eid)
        nodes = {}
        links = []
        for record in result:
            c = record["c"]
            if c is not None:
                eid = c.element_id if hasattr(c, "element_id") else str(getattr(c, "id", str(c)))
                if eid not in nodes:
                    nodes[eid] = {"id": eid, "labels": list(c.labels), "properties": serialize_properties(dict(c))}
            m = record.get("m")
            if m is not None:
                eid_m = m.element_id if hasattr(m, "element_id") else str(getattr(m, "id", str(m)))
                if eid_m not in nodes:
                    nodes[eid_m] = {"id": eid_m, "labels": list(m.labels), "properties": serialize_properties(dict(m))}
            r = record.get("r")
            if r is not None:
                start_node = r.start_node if hasattr(r, "start_node") else r.nodes[0]
                end_node = r.end_node if hasattr(r, "end_node") else r.nodes[1]
                start_id = start_node.element_id if hasattr(start_node, "element_id") else str(getattr(start_node, "id", str(start_node)))
                end_id = end_node.element_id if hasattr(end_node, "element_id") else str(getattr(end_node, "id", str(end_node)))
                links.append({"source": start_id, "target": end_id, "type": r.type, "properties": serialize_properties(dict(r))})
        
        return {"nodes": list(nodes.values()), "links": links}


