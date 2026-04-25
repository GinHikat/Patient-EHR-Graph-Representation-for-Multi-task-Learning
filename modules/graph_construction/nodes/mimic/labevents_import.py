import duckdb
import os
import sys

# Paths setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

data_dir = os.getenv('DATA_DIR')
# Fixing the path separator for the SQL query strings
mimic_path = os.path.join(data_dir, 'mimic_iv').replace('\\', '/')
lab_clean_path = os.path.join(mimic_path, 'hosp', 'lab_clean.csv')
output_csv = os.path.join(mimic_path, 'hosp', 'lab_aggregated_metadata.csv')

def aggregate_labs_to_csv():
    con = duckdb.connect()
    
    print(f"Project Root: {project_root}")
    print(f"Reading from: {lab_clean_path}")
    print(f"Beginning aggregation. This may take a while for 8M records...")
    
    # query explanation:
    # 1. clean_stats: Removes duplicate stat values for the same time by taking the average.
    # 2. ranked_labs: calculates the lab_index (1, 2, 3...) per patient ordered by time.
    # 3. Final SELECT: Groups everything by patient and time, 
    #    then creates a JSON string containing all stats tracked at that specific time.
    query = f"""
    COPY (
        WITH clean_stats AS (
            SELECT 
                subject_id, 
                hadm_id, 
                charttime, 
                stat, 
                avg(valuenum) as valuenum
            FROM read_csv_auto('{lab_clean_path}')
            WHERE stat IS NOT NULL
            GROUP BY subject_id, hadm_id, charttime, stat
        ),
        ranked_labs AS (
            SELECT *,
                DENSE_RANK() OVER (PARTITION BY subject_id ORDER BY charttime) as lab_index
            FROM clean_stats
        )
        SELECT 
            subject_id, 
            hadm_id, 
            charttime,
            subject_id || '_lab_' || lab_index as lab_id,
            -- Convert the list of stats and values into a JSON string
            to_json(map(list(stat), list(valuenum))) as stats_json
        FROM ranked_labs
        GROUP BY subject_id, hadm_id, charttime, lab_id
    ) TO '{output_csv}' (HEADER, DELIMITER ',');
    """
    
    try:
        con.execute(query)
        print(f"\nSUCCESS: Aggregated data saved to {output_csv}")
    except Exception as e:
        print(f"\nError during aggregation: {e}")

if __name__ == "__main__":
    aggregate_labs_to_csv()
