from core.database import get_db_driver

def check_paths():
    driver = get_db_driver()
    with driver.session() as session:
        # Check path from a Patient node
        q = """
        MATCH (dc:DiagnosisCategory)-[r1]-(d:Diagnosis)
        RETURN type(r1) as rel_type, count(*) as count
        LIMIT 10
        """
        results = session.run(q)
        print("DiagnosisCategory -> Diagnosis:")
        for r in results:
            print(f"  {r['rel_type']}: {r['count']}")
            
        q2 = """
        MATCH (d:Diagnosis)-[r2]-(dr:Drug)
        RETURN type(r2) as rel_type, count(*) as count
        LIMIT 10
        """
        results2 = session.run(q2)
        print("Diagnosis -> Drug:")
        for r in results2:
            print(f"  {r['rel_type']}: {r['count']}")

if __name__ == "__main__":
    check_paths()
