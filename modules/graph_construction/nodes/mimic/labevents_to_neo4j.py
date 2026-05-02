import pandas as pd
import duckdb
import json
from tqdm import tqdm
import os
import sys
import time
from collections import Counter
import json
import warnings
import logging

logging.getLogger("neo4j").setLevel(logging.ERROR)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
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
    df['stats_json'] = df['stats_json'].apply(lambda x: json.loads(x) if isinstance(x, str) else {})

    # Import to Neo4j
    cypher_lab = """
        UNWIND $rows AS row
        MATCH (p:Patient:Test:MIMIC {id: row.subject_id})

        MATCH (l:MIMIC:Lab:Test {id: row.lab_id})
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

def attribute_truncate():
    '''
    Count frequency of each attribute and remove attributes that appear less than threshold.
    '''
    # Count frequency of each attribute
    excluded = {'id', 'name', 'patient_id', 'admission_id', 'charttime'}
    test_counts = Counter()
    batch_size = 10000

    # Get total count first
    count_result = query_neo4j('MATCH (lab:Result) RETURN count(lab) AS total')
    total = count_result[0]['total']
    print(f"Total lab nodes: {total}")

    with tqdm(total=total, desc="Scanning lab panels", unit="node") as pbar:
        skip = 0
        while True:
            batch = query_neo4j(f'''
                MATCH (lab:Result)
                RETURN keys(lab) AS props
                SKIP {skip} LIMIT {batch_size}
            ''')

            if not batch:
                break

            for row in batch:
                for key in row['props']:
                    if key not in excluded:
                        test_counts[key] += 1

            pbar.update(len(batch))
            skip += batch_size

            if len(batch) < batch_size:
                break

    print(f"\nTotal unique tests: {len(test_counts)}")
    print()

    # Remove irrelevant attributes
    REMOVE = {
        'H', 'L', 'I',           # flag counts, not tests
        'Utx1', 'Utx2', 'Utx3', 'Utx4', 'Utx5', 'Utx6', 'Utx7', 'Utx9', 'Utx10',  # unknown coded tests
        'Hpe1', 'Hpe2', 'Hpe3', 'Hpe4', 'Hpe6', 'Hpe7',  # unknown coded tests
        'Stx2', 'Stx3', 'Stx4', 'Stx5', 'Stx6',           # unknown coded tests
        'Arch-1', 'Pan1', 'Pan2',                           # unknown coded tests
        'Cov8Mc', 'Cov8Ic', 'Cov10', 'Cov11', 'Cov12', 'Cov13', 'Cov12Mc', 'Cov12Ic',  # COVID PCR flags
        'E Gene Ct', 'E Gene Endpt', 'N2 Gene Ct', 'N2 Gene Endpt',  # COVID PCR
        'Flua1', 'Flua2', 'Flub1', 'Flub2', 'Rsv1', 'Rsv2',  # flu/RSV PCR flags
        'Other', 'Other Cells', 'Other Cell',               # catch-all categories
        'Peep', 'Tidal Volume', 'Required O2', 'O2 Flow', 'Oxygen',  # ventilator settings, not labs
        'Temperature',            # vital sign, not a lab
        'Length Of Urine Collection', 'Urine Volume',       # collection metadata
    }

    ALIASES = {
        'Potassium, Whole Blood':        'Potassium',
        'Sodium, Whole Blood':           'Sodium',
        'Chloride, Whole Blood':         'Chloride',
        'Creatinine, Whole Blood':       'Creatinine',
        'Hematocrit, Calculated':        'Hematocrit',
        'Calculated Bicarbonate, Whole Blood': 'Bicarbonate',
        'Rbc':                           'Red Blood Cells',
        'Wbc':                           'White Blood Cells',
        'Wbc Count':                     'White Blood Cells',
        'Wbcp':                          'White Blood Cells',
        'Lymphs':                        'Lymphocytes',
        'Monos':                         'Monocytes',
        'Polys':                         'Neutrophils',
        'Eag':                           '% Hemoglobin A1C',  # estimated average glucose, derived from A1C
    }

    PROMOTE = {
        'Ammonia',           # hepatic encephalopathy marker
        'Carboxyhemoglobin', # carbon monoxide poisoning
        'Methemoglobin',     # methemoglobinemia
        'Acetaminophen',     # overdose workup
        'Ethanol',           # intoxication
        'D-Dimer',           # already in kept list at 24k, just noting it's important
        'Ammonia',
        'Beta Hydroxybutyrate',  # DKA marker
        'Homocysteine',
        'Lithium',           # drug level monitoring
        'Phenobarbital',
        'Carbamazepine',
        'Valproic Acid',     # already kept
        'Methotrexate',
    }

    # Final clean vocabulary
    final_vocab = {}

    for test, count in test_counts.items():
        # Skip noise
        if test in REMOVE:
            continue
        
        # Resolve aliases
        canonical = ALIASES.get(test, test)
        
        # Keep if above threshold OR explicitly promoted
        if count >= 10000 or canonical in PROMOTE:
            if canonical not in final_vocab:
                final_vocab[canonical] = 0
            final_vocab[canonical] += count  # sum counts for aliases

    # Sort by frequency
    final_vocab = dict(sorted(final_vocab.items(), key=lambda x: -x[1]))

    print(f"Final clean vocabulary: {len(final_vocab)} tests")

    lab2idx = {test: idx for idx, test in enumerate(final_vocab.keys())}
    with open('lab_vocab.json', 'w') as f:
        json.dump(lab2idx, f, indent=2)

    print("Saved to lab_vocab.json")

def drop_lab():
    
    with open('lab_vocab.json', 'r') as f:
        lab_vocab = json.load(f)

    excluded = {'id', 'name', 'patient_id', 'admission_id', 'charttime'}
    keep_props = list(set(lab_vocab.keys()) | excluded)
    vocab_only = list(lab_vocab.keys())

    def to_cypher_list(items):
        escaped = [f"'{item}'" for item in items]
        return '[' + ', '.join(escaped) + ']'

    keep_literal  = to_cypher_list(keep_props)
    vocab_literal = to_cypher_list(vocab_only)

    total = query_neo4j('MATCH (lab:Result) RETURN count(lab) AS total')[0]['total']
    print(f"Total Result nodes: {total:,}")

    # Remove non-vocab properties
    print("\nStep 1: Removing non-vocab properties...")
    batch_size = 5000

    with tqdm(total=total, unit="node", desc="Cleaning properties") as pbar:
        skip = 0
        while True:
            result = query_neo4j(f'''
                MATCH (lab:Result)
                WITH lab SKIP {skip} LIMIT {batch_size}
                WITH lab, [k IN keys(lab) WHERE NOT k IN {keep_literal}] AS to_remove
                FOREACH (key IN to_remove | REMOVE lab[key])
                RETURN count(lab) AS processed
            ''')

            processed = result[0]['processed']
            pbar.update(processed)
            skip += batch_size

            if processed < batch_size:
                break

            time.sleep(0.05)

    print("Properties cleaned.")

    # Count empty nodes
    print("\nCounting empty nodes...")
    empty_total = query_neo4j(f'''
        MATCH (lab:Result)
        WHERE none(k IN keys(lab) WHERE k IN {vocab_literal})
        RETURN count(lab) AS total
    ''')[0]['total']
    print(f"Empty nodes to delete: {empty_total:,}")

    # Delete empty nodes after dropping attributes
    if empty_total > 0:
        with tqdm(total=empty_total, unit="node", desc="Deleting empty nodes") as pbar:
            while True:
                result = query_neo4j(f'''
                    MATCH (lab:Result)
                    WHERE none(k IN keys(lab) WHERE k IN {vocab_literal})
                    WITH lab LIMIT {batch_size}
                    DETACH DELETE lab
                    RETURN count(lab) AS deleted
                ''')

                deleted = result[0]['deleted']
                pbar.update(deleted)

                if deleted < batch_size:
                    break

                time.sleep(0.05)
    else:
        print("No empty nodes to delete.")

if __name__ == "__main__":
    drop_lab()
