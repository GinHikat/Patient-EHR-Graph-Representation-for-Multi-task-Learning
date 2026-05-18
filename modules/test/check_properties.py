import os
import sys

# Setup project root path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from shared_functions.global_functions import query_neo4j

def main():
    print("Loading diagnosis_map.csv...")
    import pandas as pd
    df = pd.read_csv('D:/Study/Education/Projects/Thesis/data/diagnosis_map.csv')
    print("Shape:", df.shape)
    print("Columns:", df.columns)
    print("First few rows:\n", df.head(10))
    print("Unique ccsr_categories count:", df['ccsr_category'].nunique())
    print("Unique icd_codes count:", df['icd_code'].nunique())

if __name__ == "__main__":
    main()
