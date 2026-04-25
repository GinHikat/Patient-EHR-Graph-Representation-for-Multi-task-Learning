import pandas as pd
import duckdb
import json
from tqdm import tqdm
import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from shared_functions.global_functions import dml_ddl_neo4j, query_neo4j

from dotenv import load_dotenv
load_dotenv()

data_dir = os.getenv('DATA_DIR')
mimic_path = os.path.join(data_dir, 'mimic_iv')
aggregated_csv = os.path.join(mimic_path, 'hosp', 'lab_aggregated_metadata.csv').replace('\\', '/')

start_idx = 0
BATCH_SIZE = 500

def import_labs_to_neo4j():

    # Prepare indexes for faster matching
    print("Preparing Neo4j indices...")
    try:
        query_neo4j("CREATE INDEX lab_id_idx IF NOT EXISTS FOR (n:Lab) ON (n.id)")
        query_neo4j("CREATE INDEX adm_id_idx IF NOT EXISTS FOR (n:Admission) ON (n.id)")
        query_neo4j("CREATE INDEX pat_id_idx IF NOT EXISTS FOR (n:Patient) ON (n.id)")
        time.sleep(2)
    except Exception as e:
        print(f"Index creation note: {e}")

    df = pd.read_csv(aggregated_csv)

    # Import to Neo4j
    cypher_lab = """
        UNWIND $rows AS row
        MATCH (p:Patient:Test:MIMIC {id: row.subject_id})

        MERGE (l:MIMIC:Lab:Test {id: row.lab_id})
        SET l:Result,
            l += row.stats,
            l.charttime = row.charttime,
            l.name = row.lab_id,
            l.patient_id = row.subject_id,
            l.admission_id = row.hadm_id

        WITH l, row, p
        CALL (l, row, p) {
            WITH l, row, p WHERE row.hadm_id IS NOT NULL
            MATCH (a:Admission:Test:MIMIC {id: row.hadm_id})
            MERGE (a)-[:HAS_LAB]->(l)

            UNION

            WITH l, row, p WHERE row.hadm_id IS NULL
            MERGE (p)-[:HAS_LAB]->(l)
        }
    """

    for i in tqdm(range(start_idx, len(df), BATCH_SIZE), desc="Batch processing"):
        batch = df.iloc[i:i+BATCH_SIZE]

        rows = []
        for _, row in batch.iterrows():
            # Parse stats_json dict into flat attributes
            stats = row['stats_json']
            if isinstance(stats, str):
                import json
                stats = json.loads(stats)

            rows.append({
                "lab_id": row["lab_id"],
                "subject_id": row["subject_id"],
                "hadm_id": row["hadm_id"] if pd.notna(row["hadm_id"]) else None,
                "charttime": str(row["charttime"]),
                "stats": stats 
            })

        dml_ddl_neo4j(
            cypher_lab,
            progress=False,
            rows=rows
        )

if __name__ == "__main__":
    import_labs_to_neo4j()
