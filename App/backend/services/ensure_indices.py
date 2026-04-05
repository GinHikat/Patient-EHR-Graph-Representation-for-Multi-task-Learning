from core.database import get_db_driver

def ensure_indices(namespace: str = "Test"):
    """
    Utility to ensure high-performance indices are created for the graph database.
    Run this after importing large datasets.
    """
    driver = get_db_driver()
    queries = [
        # Primary Namespace Index
        f"CREATE INDEX node_id_idx_{namespace} IF NOT EXISTS FOR (n:`{namespace}`) ON (n.id)",
        
        # Entity Specific Indices for faster label-level matching
        f"CREATE INDEX admission_id_idx IF NOT EXISTS FOR (n:Admission) ON (n.id)",
        f"CREATE INDEX patient_id_idx IF NOT EXISTS FOR (n:Patient) ON (n.id)",
        f"CREATE INDEX procedure_id_idx IF NOT EXISTS FOR (n:Procedure) ON (n.id)",
        
        # Search property indices
        f"CREATE INDEX node_name_idx_{namespace} IF NOT EXISTS FOR (n:`{namespace}`) ON (n.name)",
        f"CREATE INDEX node_title_idx_{namespace} IF NOT EXISTS FOR (n:`{namespace}`) ON (n.title)",
        f"CREATE INDEX node_Title_idx_{namespace} IF NOT EXISTS FOR (n:`{namespace}`) ON (n.Title)"
    ]
    
    with driver.session() as session:
        for q in queries:
            print(f"Executing: {q}")
            try:
                session.run(q)
            except Exception as e:
                print(f"Error creating index: {e}")
    
    print("Optimization: Indices ensured.")

if __name__ == "__main__":
    ensure_indices()
