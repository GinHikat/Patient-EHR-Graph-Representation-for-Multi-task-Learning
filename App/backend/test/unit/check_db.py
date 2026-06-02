from core.database import get_db_driver

def check_counts():
    driver = get_db_driver()
    with driver.session() as session:
        # Check standard Test count
        count_test = session.run("MATCH (n:Test) RETURN count(n) as total").single()["total"]
        print(f"Count (n:Test): {count_test}")
        
        # Check all node count
        count_all = session.run("MATCH (n) RETURN count(n) as total").single()["total"]
        print(f"Count (n): {count_all}")
        
        # Check all labels
        labels = session.run("CALL db.labels() YIELD label RETURN label as name").values()
        print(f"Labels: {[l[0] for l in labels]}")
        
        # Check relationship count
        count_rel = session.run("MATCH ()-[r]->() RETURN count(r) as total").single()["total"]
        print(f"Relationships: {count_rel}")

if __name__ == "__main__":
    check_counts()
