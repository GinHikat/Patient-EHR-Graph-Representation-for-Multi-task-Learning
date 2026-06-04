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
        f"CREATE INDEX node_Title_idx_{namespace} IF NOT EXISTS FOR (n:`{namespace}`) ON (n.Title)",

        # Disease Label Indices
        "CREATE INDEX disease_id_idx IF NOT EXISTS FOR (n:Disease) ON (n.id)",
        "CREATE INDEX disease_cui_idx IF NOT EXISTS FOR (n:Disease) ON (n.cui)",
        "CREATE INDEX disease_mesh_idx IF NOT EXISTS FOR (n:Disease) ON (n.mesh_id)",
        "CREATE INDEX disease_omim_idx IF NOT EXISTS FOR (n:Disease) ON (n.omim_id)",
        "CREATE INDEX disease_doid_idx IF NOT EXISTS FOR (n:Disease) ON (n.doid)",
        "CREATE INDEX disease_snomed_idx IF NOT EXISTS FOR (n:Disease) ON (n.snomed_id)",
        
        # Drug Label Indices
        "CREATE INDEX drug_id_idx IF NOT EXISTS FOR (n:Drug) ON (n.id)",
        "CREATE INDEX drug_cui_idx IF NOT EXISTS FOR (n:Drug) ON (n.cui)",
        "CREATE INDEX drug_mesh_idx IF NOT EXISTS FOR (n:Drug) ON (n.mesh_id)",
        "CREATE INDEX drug_rxnorm_idx IF NOT EXISTS FOR (n:Drug) ON (n.rxnorm)",
        "CREATE INDEX drug_rxnorm_id_idx IF NOT EXISTS FOR (n:Drug) ON (n.rxnorm_id)",
        "CREATE INDEX drug_drugbank_idx IF NOT EXISTS FOR (n:Drug) ON (n.drugbank_id)",
        "CREATE INDEX drug_pubchem_idx IF NOT EXISTS FOR (n:Drug) ON (n.pubchem_id)",
        
        # Phenotype Label Indices
        "CREATE INDEX phenotype_id_idx IF NOT EXISTS FOR (n:Phenotype) ON (n.id)",
        "CREATE INDEX phenotype_cui_idx IF NOT EXISTS FOR (n:Phenotype) ON (n.cui)",
        "CREATE INDEX phenotype_mesh_idx IF NOT EXISTS FOR (n:Phenotype) ON (n.mesh_id)",
        "CREATE INDEX phenotype_hpo_idx IF NOT EXISTS FOR (n:Phenotype) ON (n.hpo_id)",
        "CREATE INDEX phenotype_snomed_idx IF NOT EXISTS FOR (n:Phenotype) ON (n.snomed_id)"
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
